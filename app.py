import re
import time
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from groq import Groq, RateLimitError
from rag_engine import ShipraRag
from intent_parser import (
    OrderIntent,
    parse_order_intent,
    resolve_clarification_reply,
    resolve_date_range,
)

try:
    from intent_parser import parse_request_route
except ImportError:
    parse_request_route = None
from live_api_executor import execute_live_intent
from live_api_intent import parse_live_api_intent
from live_response_formatter import format_live_result
from shipra_api import (
    ShipraAPI,
    ShipraAPIError,
    format_order_detail,
    format_order_rows,
)

STATUS_IDS = {
    "all": None,
    "delivered": "6",
    "returned": "10",
    "refunded": "7",
    "pending_for_return": "9",
    "order_placed": "1",
    "on_hold": "13",
    "in_transit": "22",
    "cancelled": "17",
    "out_for_delivery": "5",
    # User-approved operational definition: exclude terminal/return statuses
    # Delivered(6), Refunded(7), Exchanged(8), PendingForReturn(9), Returned(10),
    # Cancelled(17), Lost(23), Damage(24), ReturnToOrigin(26).
    "in_progress": "1,2,3,4,5,11,12,13,14,15,16,18,19,20,21,22,25,27,28",
}
PAYMENT_STATUS_IDS = {"all": None, "unpaid": 1, "paid": 2}

st.set_page_config(page_title="Shipra Code Assistant", page_icon="🤖", layout="wide")

@st.cache_resource
def load_engine(): return ShipraRag(Path(__file__).resolve().parent / "data")

def shipra_base_url() -> str:
    return st.secrets.get("SHIPRA_API_BASE_URL", "https://stage-api.shipra.io/api/")

def response_language(question: str) -> str:
    """Choose the response language from an explicit request, then the question."""
    normalized = question.lower()

    # An explicitly requested language always wins, even for an English question.
    instruction = r"(?:answer|reply|explain|response|jawab|samjha(?:o|do)|bata(?:o|do)|mai|mein|me|in)"
    if re.search(rf"(?:\b{instruction}\b.{{0,50}}\b(?:arabic|arab)\b|\b(?:arabic|arab)\b.{{0,50}}\b{instruction}\b|العربية)", normalized):
        return "Arabic"
    if re.search(rf"(?:\b{instruction}\b.{{0,50}}\b(?:roman\s*urdu|urdu)\b|\b(?:roman\s*urdu|urdu)\b.{{0,50}}\b{instruction}\b|اردو)", normalized):
        return "Roman Urdu"
    if re.search(rf"(?:\b{instruction}\b.{{0,50}}\benglish\b|\benglish\b.{{0,50}}\b{instruction}\b)", normalized):
        return "English"

    # Arabic script is answered in Arabic. Roman Urdu is detected from common
    # Urdu words written in Latin letters; English remains the safe default.
    if re.search(r"[\u0600-\u06ff]", question):
        return "Arabic"
    roman_urdu_words = r"\b(kya|kia|kaise|kesy|mujhe|mjhy|hai|hain|kahan|kyun|karna|kro|batao|btao|isko|isy|mera|mery|aap|tum|yeh|wali|wala|aur|or)\b"
    if re.search(roman_urdu_words, normalized):
        return "Roman Urdu"
    return "English"

def answer(question, history):
    is_workflow = bool(
        re.search(
            r"\b(step[ -]?by[ -]?step|workflow|flow|how|kaise|kesy|kis tarhan)\b",
            question,
            re.IGNORECASE,
        )
    )
    evidence = load_engine().retrieve(question, limit=4 if is_workflow else 8)

    if is_workflow:
        trace_query = (
            f"{question} frontend API AxiosInterceptors backend controller "
            "command handler mediator repository"
        )
        trace_evidence = load_engine().retrieve(trace_query, limit=6)
        seen = {
            (
                item.meta.get("file_path"),
                item.meta.get("start_line"),
                item.meta.get("end_line"),
            )
            for item in evidence
        }
        for item in trace_evidence:
            key = (
                item.meta.get("file_path"),
                item.meta.get("start_line"),
                item.meta.get("end_line"),
            )
            if key not in seen:
                evidence.append(item)
                seen.add(key)
            if len(evidence) == 8:
                break
    context_blocks = []
    for number, item in enumerate(evidence, 1):
        meta = item.meta
        context_blocks.append(
            f"[S{number}] FILE: {meta.get('file_path')}\n"
            f"LINES: {meta.get('start_line')}-{meta.get('end_line')}\n"
            f"SYMBOL: {meta.get('symbol')}\n"
            f"CODE:\n{item.text[:1400]}"
        )
    context = "\n\n".join(context_blocks)[:11000]
    language = response_language(question)
    prompt = f'''You are Shipra Code Assistant.
Your required response language is: {language}.
Language rules: English means English only. Arabic means Arabic script only. Roman Urdu means Latin/Roman Urdu only. Never write Hindi or Devanagari script. If the user explicitly requests a language, follow that request even when their question is written in another language.
Only use the supplied indexed Shipra source snapshot. Never invent a file, route, API, UI control, handler, repository call, or business fact.
Every factual statement must end in one or more citations like [S1].
If the source does not prove the answer, clearly say it is not verified by the indexed source snapshot, in the required response language. Do not guess.
For a short list question, answer directly; do not add scenario steps.
For a code-change question, separate verified existing code from proposed code and label every proposed file/path as an assumption.
For workflow, implementation, or "how" questions, give a numbered step-by-step guide. For every step, name the exact verified file and function/symbol, explain what happens next, and include a short exact code excerpt copied only from the supplied source. Put the code excerpt immediately below its step.
Required workflow format: `Step N`, then `File`, then `Function/Symbol`, then the explanation with citations, followed by a fenced code block copied from that same cited source. A workflow answer without verified fenced code excerpts is invalid.
Use at most 5 workflow steps and at most one 3-8 line code excerpt per step. Do not repeat the same source lines in multiple steps. Prefer a complete end-to-end answer over extra detail, and always finish every opened sentence and code fence.
Never reconstruct, complete, improve, or paraphrase code inside a code block. If the exact required lines are not present in the supplied sources, state that the code excerpt is not verified instead of inventing it.
Never put ellipses, placeholder comments, `/* ... */`, `// ...`, or incomplete statements in a code block. Copy only complete contiguous lines that are visible in the supplied source.
When multiple files implement similar flows, keep their behavior separate by filename and function. Do not merge branch-specific loading, notification, badge, navigation, API, or error behavior into a generic claim.
Trace a frontend API call into its verified backend controller/command/handler before describing the backend flow. If that connection is not present in the supplied sources, clearly state the evidence gap.
Never write vague phrases such as "or similar handler". Use only the exact handler and connection proven by the supplied source.
Check each create, update, success, failure, and finally branch independently. Do not claim that a badge, notification, navigation, or loading reset happens in all branches unless every cited branch proves it.
Distinguish between creating or updating a record and implementing its dashboard/section. If the question is genuinely ambiguous and the two answers would differ materially, ask one concise clarification question.
For a requested change, first show the verified existing code, then provide the proposed replacement under a clear "Proposed change" label. Never present proposed code as code already present in Shipra.
Keep each excerpt focused: normally 3-12 lines. Cite the explanation and its excerpt with the matching source marker.
Keep the answer concise. Do not add a "Verified sources", "Sources", or references section. The application renders only the source paths that your answer actually cites.

QUESTION: {question}

SOURCES:\n{context}'''
    client = Groq(api_key=st.secrets["GROQ_API_KEY"], timeout=25, max_retries=0)

    def create_completion():
        return client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Follow the required response language exactly. "
                        "Never output Devanagari/Hindi script."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2200,
        )

    try:
        result = create_completion()
    except RateLimitError:
        time.sleep(4)
        result = create_completion()
    text = result.choices[0].message.content or "Is indexed source snapshot mein jawab generate nahi hua."

    # Malformed Markdown should never break the rest of the Streamlit page.
    if text.count("```") % 2:
        text = text.rstrip() + "\n```"

    # The model may still produce a references heading despite the prompt. Remove it
    # because source rendering below is deterministic and citation-aware.
    text = re.split(r"\n\s*#{1,6}\s*(verified\s+)?sources?\s*\n", text, maxsplit=1, flags=re.IGNORECASE)[0].rstrip()

    # Render only evidence referenced in the generated answer, in citation order.
    cited_numbers = []
    for match in re.finditer(r"\[S(\d+)\]", text, flags=re.IGNORECASE):
        number = int(match.group(1))
        if 1 <= number <= len(evidence) and number not in cited_numbers:
            cited_numbers.append(number)

    if not cited_numbers:
        return text

    sources = "\n".join(
        f"- [S{number}] `{evidence[number - 1].meta.get('file_path')}` — "
        f"lines {evidence[number - 1].meta.get('start_line')}-{evidence[number - 1].meta.get('end_line')}"
        for number in cited_numbers
    )
    return text + "\n\n### Verified sources\n" + sources

def _date_scope(from_date, to_date, language: str) -> str:
    if not from_date or not to_date:
        return "تمام الفترات" if language == "Arabic" else ("all time" if language == "English" else "all time")
    return f"{from_date} se {to_date} tak" if language == "Roman Urdu" else f"{from_date} إلى {to_date}" if language == "Arabic" else f"from {from_date} to {to_date}"

def _connect_message(language: str) -> str:
    if language == "Arabic":
        return "يرجى ربط حساب Shipra من الشريط الجانبي أولاً للوصول إلى بيانات الطلبات المباشرة."
    if language == "Roman Urdu":
        return "Live order data ke liye pehle sidebar se apna Shipra account connect karein."
    return "Connect your Shipra account from the sidebar first to access live order data."

def structured_store_answer(question: str) -> str | None:
    text = question.lower().strip()

    has_store_word = bool(
        re.search(
            r"\b(store|stores|stor)\b|سٹور|متجر|متاجر",
            text,
            re.IGNORECASE,
        )
    )

    has_count_word = bool(
        re.search(
            r"\b(how many|count|total|kitne|kitny|kitna|kitni|kul)\b"
            r"|کتنے|كم|عدد",
            text,
            re.IGNORECASE,
        )
    )

    # Store ka sawal nahi hai to normal order/RAG flow continue hoga.
    if not has_store_word:
        return None

    # Store ke orders wala sawal order flow handle karega.
    if re.search(r"\b(order|orders|parcel|shipment)\b|آرڈر|طلب", text):
        return None

    language = response_language(question)
    auth = st.session_state.get("shipra_auth")

    if not auth:
        if language == "Arabic":
            return "يرجى ربط حساب Shipra أولاً للوصول إلى بيانات المتاجر المباشرة."
        if language == "Roman Urdu":
            return (
                "Live store data ke liye pehle sidebar se "
                "apna Shipra account connect karein."
            )
        return (
            "Connect your Shipra account from the sidebar "
            "to access live store data."
        )

    try:
        api = ShipraAPI(
            shipra_base_url(),
            auth=auth,
        )

        if "shopify" in text:
            channels, updated_auth = api.list_sale_channels(1)
            st.session_state.shipra_auth = updated_auth
            names = [
                row.get("text")
                or row.get("Text")
                or row.get("SaleChannelName")
                or row.get("saleChannelName")
                for row in channels
            ]
            names = [name for name in names if name]

            if not names:
                if language == "Arabic":
                    return "لا توجد قناة Shopify نشطة مرتبطة بحساب Shipra الحالي."
                if language == "Roman Urdu":
                    return "Current Shipra account ke sath koi active Shopify channel connected nahi hai."
                return "There is no active Shopify channel connected to the current Shipra account."

            if language == "Arabic":
                heading = f"نعم، توجد **{len(names)} قناة Shopify نشطة**:"
            elif language == "Roman Urdu":
                heading = f"Ji haan, **{len(names)} active Shopify channels** connected hain:"
            else:
                label = "channel" if len(names) == 1 else "channels"
                heading = f"Yes, **{len(names)} active Shopify {label}** are connected:"

            return "\n".join(
                [heading, *[f"{i}. {name}" for i, name in enumerate(names, 1)]]
            )

        if has_count_word:
            count, updated_auth = api.count_stores()

            st.session_state.shipra_auth = updated_auth

            if language == "Arabic":
                return f"وفقاً لواجهة Shipra المباشرة، يوجد حالياً **{count} متجر**."

            if language == "Roman Urdu":
                return f"Live Shipra API ke mutabiq Shipra mein **{count} stores** hain."

            store_label = "store" if count == 1 else "stores"
            return f"According to the live Shipra API, there are **{count} {store_label}** in Shipra."

        selected_store, updated_auth = _resolve_store(api, question)
        st.session_state.shipra_auth = updated_auth

        if selected_store:
            stores, updated_auth = api.list_all_stores()
            st.session_state.shipra_auth = updated_auth
            store = next(
                (
                    row for row in stores
                    if int(row.get("StoreId") or row.get("storeId") or 0)
                    == selected_store["id"]
                ),
                {},
            )
            details = {
                "Store name": store.get("StoreName") or store.get("storeName"),
                "Store code": store.get("StoreCode") or store.get("storeCode"),
                "Company": store.get("StoreCompany") or store.get("storeCompany"),
                "Active": store.get("Active") if "Active" in store else store.get("active"),
                "Connected channels": (
                    store.get("SaleChannelConfigCount")
                    or store.get("saleChannelConfigCount")
                ),
            }
            lines = [f"**{selected_store['name']}** store ki verified information:"]
            lines.extend(
                f"- {key}: {value}"
                for key, value in details.items()
                if value is not None
            )
            return "\n".join(lines)

        stores, updated_auth = api.list_all_stores()

        st.session_state.shipra_auth = updated_auth
        names = [
            store.get("StoreName")
            or store.get("storeName")
            or store.get("Name")
            or store.get("name")
            or f"Store {store.get('StoreId') or store.get('storeId') or ''}"
            for store in stores
        ]

        if language == "Arabic":
            heading = f"يوجد **{len(names)} متجر** في Shipra:"
        elif language == "Roman Urdu":
            heading = f"Shipra mein **{len(names)} stores** hain:"
        else:
            heading = f"Shipra has **{len(names)} stores**:"

        return "\n".join([heading, *[f"{i}. {name}" for i, name in enumerate(names, 1)]])

    except ShipraAPIError as exc:
        return f"Shipra store request stopped safely: `{exc}`"

    
def _clarification_message(intent: OrderIntent) -> str:
    if intent.clarification:
        return intent.clarification
    if intent.language == "Arabic":
        return "يرجى توضيح حالة الطلب والفترة الزمنية المطلوبة."
    if intent.language == "Roman Urdu":
        return "Order status aur date range clear karein taa-ke exact result diya ja sake."
    return "Please clarify the order status and date range so I can return an exact result."


def _normalize_store_text(value: str) -> str:
    value = re.sub(r"\(\s*default\s*\)", "", value, flags=re.IGNORECASE)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _resolve_store(api: ShipraAPI, question: str):
    stores, updated_auth = api.list_all_stores()
    question_text = f" {_normalize_store_text(question)} "
    exact_matches = []
    fuzzy_matches = []
    question_words = question_text.split()

    for store in stores:
        store_id = store.get("StoreId") or store.get("storeId")
        store_name = store.get("StoreName") or store.get("storeName")
        normalized_name = _normalize_store_text(str(store_name or ""))

        if store_id is not None and normalized_name and f" {normalized_name} " in question_text:
            exact_matches.append((len(normalized_name), int(store_id), str(store_name)))
            continue

        if store_id is None or len(normalized_name) < 4:
            continue

        name_word_count = len(normalized_name.split())
        windows = [
            " ".join(question_words[index:index + name_word_count])
            for index in range(len(question_words) - name_word_count + 1)
        ]
        if name_word_count == 1:
            generic_words = {
                "store", "stores", "shop", "orders", "order",
                "sale", "sales", "channel",
            }
            windows = [window for window in windows if window not in generic_words]
        score = max(
            (SequenceMatcher(None, normalized_name, window).ratio() for window in windows),
            default=0.0,
        )
        if score >= 0.90:
            fuzzy_matches.append((score, int(store_id), str(store_name)))

    if exact_matches:
        _, store_id, store_name = max(exact_matches)
        return {"id": store_id, "name": store_name}, updated_auth

    fuzzy_matches.sort(reverse=True)
    if not fuzzy_matches:
        return None, updated_auth

    # Similar store names ko guess karne ke bajaye ambiguous result block hoga.
    if len(fuzzy_matches) > 1 and fuzzy_matches[0][0] - fuzzy_matches[1][0] < 0.04:
        return None, updated_auth

    _, store_id, store_name = fuzzy_matches[0]
    return {"id": store_id, "name": store_name}, updated_auth


def _resolve_sale_channel(api: ShipraAPI, question: str):
    lookups, updated_auth = api.list_sale_channel_lookups()
    question_text = f" {_normalize_store_text(question)} "
    matches = []

    for lookup in lookups:
        lookup_id = (
            lookup.get("SaleChannelLookupId")
            or lookup.get("saleChannelLookupId")
            or lookup.get("id")
        )
        lookup_name = (
            lookup.get("SaleChannelName")
            or lookup.get("saleChannelName")
            or lookup.get("text")
            or lookup.get("name")
        )
        normalized_name = _normalize_store_text(str(lookup_name or ""))

        if lookup_id is not None and normalized_name and f" {normalized_name} " in question_text:
            matches.append((len(normalized_name), int(lookup_id), str(lookup_name)))

    if not matches:
        return None, updated_auth

    _, lookup_id, lookup_name = max(matches)
    configs, updated_auth = api.list_sale_channels(lookup_id)
    config_ids = [
        int(row.get("id") or row.get("Id"))
        for row in configs
        if row.get("id") or row.get("Id")
    ]

    return {
        "ids": config_ids,
        "name": lookup_name,
    }, updated_auth

def _status_label(status: str, payment_status: str, count=None) -> str:
    singular = count == 1
    if payment_status == "unpaid":
        return "unpaid order" if singular else "unpaid orders"
    if payment_status == "paid":
        return "paid order" if singular else "paid orders"
    english = {
        "all": ("order", "orders"),
        "delivered": ("delivered order", "delivered orders"),
        "in_progress": ("dashboard in-progress order", "dashboard in-progress orders"),
        "returned": ("returned order", "returned orders"),
        "refunded": ("refunded order", "refunded orders"),
        "pending_for_return": ("order pending for return", "orders pending for return"),
        "order_placed": ("order-placed order", "order-placed orders"),
        "on_hold": ("on-hold order", "on-hold orders"),
        "in_transit": ("in-transit order", "in-transit orders"),
        "cancelled": ("cancelled order", "cancelled orders"),
        "out_for_delivery": ("out-for-delivery order", "out-for-delivery orders"),
    }
    return english[status][0 if singular else 1]

def _format_store_order_counts(
    data: dict,
    language: str,
) -> str:
    total_count = int(
        data.get("totalCount") or 0
    )

    stores = data.get("stores") or []

    if language == "Arabic":
        lines = [
            (
                f"إجمالي الطلبات هو "
                f"**{total_count}**:"
            )
        ]
    elif language == "Roman Urdu":
        lines = [
            (
                f"Live Shipra API ke mutabiq "
                f"total **{total_count} orders** hain:"
            )
        ]
    else:
        lines = [
            (
                f"According to the live Shipra API, "
                f"there are **{total_count} orders**:"
            )
        ]

    for index, store in enumerate(
        stores,
        start=1,
    ):
        store_name = store.get(
            "storeName",
            "Unknown store",
        )

        order_count = int(
            store.get("orderCount") or 0
        )

        if language == "Arabic":
            lines.append(
                f"{index}. {store_name}: "
                f"**{order_count} طلب**"
            )
        elif language == "Roman Urdu":
            lines.append(
                f"{index}. {store_name}: "
                f"**{order_count} orders**"
            )
        else:
            label = (
                "order"
                if order_count == 1
                else "orders"
            )

            lines.append(
                f"{index}. {store_name}: "
                f"**{order_count} {label}**"
            )

    return "\n".join(lines)

def _validated_live_result(data, intent: OrderIntent):
    expected_ids = STATUS_IDS[intent.status]
    if expected_ids is not None:
        allowed = {int(value) for value in expected_ids.split(",")}
        for row in data.get("rows") or []:
            raw = row.get("CarrierTrackingStatusId", row.get("carrierTrackingStatusId"))
            name = str(row.get("CarrierTrackingStatus", row.get("carrierTrackingStatus", ""))).lower().replace(" ", "")
            if raw is None:
                exact_names = {"delivered": "delivered", "returned": "returned", "refunded": "refunded"}
                expected_name = exact_names.get(intent.status)
                if expected_name and name != expected_name:
                    raise ShipraAPIError("Shipra returned a row outside the requested tracking-status filter.")
                if intent.status == "in_progress" and name in {"refunded", "exchanged", "returned", "cancelled", "lost", "damage", "returntoorigin"}:
                    raise ShipraAPIError(
                        "Shipra's dashboard in-progress rule includes a terminal/exception status. "
                        "This result was blocked instead of presenting it as an active order."
                    )
                continue
            status_id = int(raw)
            if status_id not in allowed:
                raise ShipraAPIError("Shipra returned a row outside the requested status filter.")
            if intent.status == "in_progress" and status_id in {7, 8, 9, 10, 17, 23, 24, 26}:
                raise ShipraAPIError(
                    "Shipra's dashboard in-progress rule includes a terminal/exception status. "
                    "This result was blocked instead of presenting it as an active order."
                )

    if intent.payment_status != "all":
        for row in data.get("rows") or []:
            name = str(row.get("PaymentStatus", row.get("paymentStatus", ""))).strip().lower()
            if name and name != intent.payment_status:
                raise ShipraAPIError("Shipra returned a row outside the requested payment-status filter.")


def _validate_order_source(data, store_id=None, channel_ids=None):
    allowed_channels = {int(value) for value in (channel_ids or [])}

    for row in data.get("rows") or []:
        if store_id is not None:
            raw_store_id = row.get("StoreId", row.get("storeId"))
            if raw_store_id is None or int(raw_store_id) != int(store_id):
                raise ShipraAPIError(
                    "Shipra returned an order outside the selected store. Result blocked."
                )

        if allowed_channels:
            raw_channel_id = row.get(
                "SaleChannelConfigId",
                row.get("saleChannelConfigId"),
            )
            if raw_channel_id is None or int(raw_channel_id) not in allowed_channels:
                raise ShipraAPIError(
                    "Shipra returned an order outside the selected sale channel. Result blocked."
                )

def structured_live_answer(question: str, history) -> str | None:
    text = " ".join(question.lower().strip().split())

    # Store-only sawal ko order parser kabhi handle nahi karega.
    if re.search(r"\b(store|stores|stor)\b|سٹور|متجر|متاجر", text) and not re.search(
        r"\b(order|orders|parcel|shipment)\b|آرڈر|طلب",
        text,
    ):
        return None

    next_words = {"next", "next page", "agla", "agla page", "اگلا"}
    previous_words = {"previous", "previous page", "back", "pichla", "pichla page", "پچھلا"}
    page_state = st.session_state.get("order_page")
    selected_store_id = None
    selected_store_name = None
    selected_channel_ids = None
    selected_channel_name = None
    grouped_pages = None

    if text in next_words | previous_words and page_state:
        intent = page_state["intent"]
        page = int(page_state["page"])
        selected_store_id = page_state.get("store_id")
        selected_store_name = page_state.get("store_name")
        selected_channel_ids = page_state.get("channel_ids")
        selected_channel_name = page_state.get("channel_name")
        grouped_pages = page_state.get("grouped_pages")
        page += 1 if text in next_words else -1
        page = max(page, 0)
    else:
        page = 0
        intent = None
        if text not in next_words | previous_words:
            st.session_state.pop("order_page", None)

    pending_intent = st.session_state.get("pending_order_intent")
    if intent is None:
        intent = resolve_clarification_reply(pending_intent, question) if isinstance(pending_intent, OrderIntent) else None
    if intent is None:
        try:
            intent = parse_order_intent(st.secrets["GROQ_API_KEY"], question, history[:-1])
        except Exception:
            likely_order = bool(re.search(r"order|parcel|shipment|payment|paid|unpaid|progress|return|آرڈر|طلب", question, re.IGNORECASE))
            if pending_intent or likely_order:
                return "Order request parser is temporarily unavailable. No Shipra API call was made; please retry shortly."
            return None
    if not intent.is_order_query:
        st.session_state.pop("pending_order_intent", None)
        st.session_state.pop("pending_order_question", None)
        return None

    # "All stores' orders" always means a store-separated list. This
    # deterministic rule keeps the result correct even if the LLM parser
    # interprets the wording as a normal all-orders request.
    all_stores_list = bool(
        intent.operation == "list"
        and re.search(
            r"\b(all|every|each|sary|sare|sab|tamam|har)\s+stores?\b",
            text,
            re.IGNORECASE,
        )
        and re.search(
            r"\b(order|orders|parcel|shipment)\b|آرڈر|طلب",
            text,
            re.IGNORECASE,
        )
    )
    if all_stores_list and intent.group_by != "store":
        intent = replace(intent, group_by="store")

    if intent.needs_clarification:
        st.session_state.pending_order_intent = intent
        st.session_state.pending_order_question = question
        return _clarification_message(intent)
    st.session_state.pop("pending_order_intent", None)
    store_question = st.session_state.pop("pending_order_question", None) or question

    auth = st.session_state.get("shipra_auth")
    if not auth:
        return _connect_message(intent.language)
    try:
        from_date, to_date = resolve_date_range(intent)
        api = ShipraAPI(shipra_base_url(), auth=auth)

        store_order_question = bool(
            re.search(r"\b(store|stores|stor)\b|سٹور|متجر|متاجر", store_question, re.IGNORECASE)
            and re.search(r"\b(order|orders|parcel|shipment)\b|آرڈر|طلب", store_question, re.IGNORECASE)
        )

        if selected_store_id is None and text not in next_words | previous_words:
            selected_store, updated_auth = _resolve_store(api, store_question)
            st.session_state.shipra_auth = updated_auth

            if selected_store:
                selected_store_id = selected_store["id"]
                selected_store_name = selected_store["name"]
            else:
                selected_channel, updated_auth = _resolve_sale_channel(api, store_question)
                st.session_state.shipra_auth = updated_auth

                if selected_channel:
                    selected_channel_ids = selected_channel["ids"]
                    selected_channel_name = selected_channel["name"]
                elif store_order_question:
                    aggregate_words = (
                        r"\b(by store|store wise|store-wise|each store|every store|"
                        r"all stores?|har store|sary stores?|sare stores?|sab stores?|"
                        r"tamam stores?|kis store)\b"
                    )
                    if not re.search(aggregate_words, text):
                        return "Store ya sale channel name match nahi hua. Exact name ke sath dobara poochein."

        if selected_channel_name and not selected_channel_ids:
            if intent.operation == "list":
                return f"{selected_channel_name} ke liye koi active connection ya matching order nahi mila."
            return f"Live Shipra API ke mutabiq **{selected_channel_name} ke 0 orders** hain."

        if (
            intent.group_by == "store"
            and intent.operation == "count"
            and selected_store_id is None
            and selected_channel_name is None
        ):
            data, updated_auth = (
                api.count_orders_by_store(
                    from_date=from_date,
                    to_date=to_date,
                    carrier_tracking_status_ids=(
                        STATUS_IDS[intent.status]
                    ),
                    payment_status_id=(
                        PAYMENT_STATUS_IDS[
                            intent.payment_status
                        ]
                    ),
                )
            )

            st.session_state.shipra_auth = (
                updated_auth
            )

            return _format_store_order_counts(
                data,
                intent.language,
            )        
        if intent.operation == "detail":
            if not intent.order_reference:
                return _clarification_message(intent)
            if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", intent.order_reference, re.IGNORECASE):
                payload, updated_auth = api.order_by_id(intent.order_reference)
            else:
                payload, updated_auth = api.order_by_reference(intent.order_reference)
            st.session_state.shipra_auth = updated_auth
            return format_order_detail(payload, language=intent.language)
        if intent.operation == "list":
            if (
                grouped_pages
                or (
                    intent.group_by == "store"
                    and selected_store_id is None
                    and selected_channel_name is None
                )
            ):
                if not grouped_pages:
                    grouped_data, updated_auth = api.count_orders_by_store(
                        from_date=from_date,
                        to_date=to_date,
                        carrier_tracking_status_ids=STATUS_IDS[intent.status],
                        payment_status_id=PAYMENT_STATUS_IDS[intent.payment_status],
                        validate_overall_total=False,
                    )
                    st.session_state.shipra_auth = updated_auth
                    grouped_pages = []
                    for store in grouped_data.get("stores") or []:
                        order_count = int(store.get("orderCount") or 0)
                        for start in range(0, order_count, 50):
                            grouped_pages.append(
                                {
                                    "storeId": int(store["storeId"]),
                                    "storeName": str(store["storeName"]),
                                    "orderCount": order_count,
                                    "start": start,
                                }
                            )

                if not grouped_pages:
                    return "Live Shipra API ke mutabiq kisi store mein matching order nahi mila."

                total_pages = len(grouped_pages)
                if page >= total_pages:
                    return "No more orders. Use Previous to go back."

                current_store = grouped_pages[page]
                data, updated_auth = api.list_orders(
                    from_date,
                    to_date,
                    limit=50,
                    start=current_store["start"],
                    carrier_tracking_status_ids=STATUS_IDS[intent.status],
                    payment_status_id=PAYMENT_STATUS_IDS[intent.payment_status],
                    store_ids=str(current_store["storeId"]),
                )
                st.session_state.shipra_auth = updated_auth
                _validated_live_result(data, intent)
                _validate_order_source(data, store_id=current_store["storeId"])

                st.session_state.order_page = {
                    "intent": intent,
                    "page": page,
                    "total_pages": total_pages,
                    "store_id": None,
                    "store_name": None,
                    "channel_ids": None,
                    "channel_name": None,
                    "grouped_pages": grouped_pages,
                }
                store_count = current_store["orderCount"]
                label = _status_label(intent.status, intent.payment_status, store_count)
                result = f"### Store: {current_store['storeName']}\n\n"
                result += format_order_rows(
                    data,
                    limit=50,
                    language=intent.language,
                    label=f"{current_store['storeName']} {label}",
                    start_index=current_store["start"] + 1,
                )
                result += (
                    f"\n\nStore page **{page + 1} of {total_pages}**. "
                    "Use the buttons below."
                )
                return result

            data, updated_auth = api.list_orders(
                from_date,
                to_date,
                limit=50,
                start=page * 50,
                carrier_tracking_status_ids=STATUS_IDS[intent.status],
                payment_status_id=PAYMENT_STATUS_IDS[intent.payment_status],
                store_ids=(str(selected_store_id) if selected_store_id is not None else None),
                sale_channel_config_ids=(
                    ",".join(str(value) for value in selected_channel_ids)
                    if selected_channel_ids
                    else None
                ),
            )
            st.session_state.shipra_auth = updated_auth
            _validated_live_result(data, intent)
            _validate_order_source(
                data,
                store_id=selected_store_id,
                channel_ids=selected_channel_ids,
            )

            total = int(data.get("count") or 0)
            total_pages = max((total + 49) // 50, 1)
            if page >= total_pages:
                return "No more orders. Type `previous` to go back."

            st.session_state.order_page = {
                "intent": intent,
                "page": page,
                "total_pages": total_pages,
                "store_id": selected_store_id,
                "store_name": selected_store_name,
                "channel_ids": selected_channel_ids,
                "channel_name": selected_channel_name,
            }
            label = _status_label(intent.status, intent.payment_status, total)
            source_name = selected_store_name or selected_channel_name
            if source_name:
                label = f"{source_name} {label}"
            result = format_order_rows(
                data,
                limit=50,
                language=intent.language,
                label=label,
                start_index=(page * 50) + 1,
            )
            result += f"\n\nPage **{page + 1} of {total_pages}**. Use the buttons below."
            return result

        if intent.operation == "count" and (
            selected_store_id is not None or selected_channel_ids
        ):
            data, updated_auth = api.list_orders(
                from_date,
                to_date,
                limit=1,
                start=0,
                carrier_tracking_status_ids=STATUS_IDS[intent.status],
                payment_status_id=PAYMENT_STATUS_IDS[intent.payment_status],
                store_ids=(str(selected_store_id) if selected_store_id is not None else None),
                sale_channel_config_ids=(
                    ",".join(str(value) for value in selected_channel_ids)
                    if selected_channel_ids
                    else None
                ),
            )
            st.session_state.shipra_auth = updated_auth
            _validated_live_result(data, intent)
            _validate_order_source(
                data,
                store_id=selected_store_id,
                channel_ids=selected_channel_ids,
            )
            count = int(data.get("count") or 0)
            label = _status_label(intent.status, intent.payment_status, count)
            scope = _date_scope(from_date, to_date, intent.language)
            source_name = selected_store_name or selected_channel_name

            if intent.language == "Roman Urdu":
                return f"Live Shipra API ke mutabiq {scope} **{source_name} ke {count} {label}** hain."
            if intent.language == "Arabic":
                return f"وفقاً لواجهة Shipra المباشرة، لدى {source_name} **{count}** طلب خلال {scope}."
            return f"According to the live Shipra API, **{source_name} has {count} {label}** {scope}."

        fetch_limit = 1000 if intent.status == "in_progress" else 1
        data, updated_auth = api.search_orders(
            from_date,
            to_date,
            carrier_tracking_status_ids=STATUS_IDS[intent.status],
            payment_status_id=PAYMENT_STATUS_IDS[intent.payment_status],
            fetch_limit=fetch_limit,
        )
        st.session_state.shipra_auth = updated_auth
        if intent.operation == "count" and intent.status == "in_progress" and not data.get("complete"):
            raise ShipraAPIError(
                "The in-progress result exceeded the safe validation limit, so its full status set could not be verified."
            )
        _validated_live_result(data, intent)
        count = data["count"]
        label = _status_label(intent.status, intent.payment_status, count)
        scope = _date_scope(from_date, to_date, intent.language)
        if intent.operation == "count":
            if intent.language == "Roman Urdu":
                return f"Live Shipra API ke mutabiq {scope} **{count} {label}** hain."
            if intent.language == "Arabic":
                return f"وفقاً لواجهة Shipra المباشرة، العدد هو **{count}** خلال {scope}."
            return f"According to the live Shipra API, there are **{count} {label}** {scope}."
        return format_order_rows(data, limit=50, language=intent.language, label=label)
    except (ShipraAPIError, ValueError) as exc:
        return f"Shipra live-data request stopped safely: `{exc}`"

def structured_universal_live_answer(question: str, history) -> str | None:
    try:
        intent = parse_live_api_intent(
            st.secrets["GROQ_API_KEY"],
            question,
            history[:-1],
        )
    except Exception:
        return None

    if not intent.is_live_query:
        return None

    if intent.needs_clarification:
        return intent.clarification or "Please provide the required ID or clarify the requested live data."

    auth = st.session_state.get("shipra_auth")
    if not auth:
        return _connect_message(intent.language)

    try:
        api = ShipraAPI(shipra_base_url(), auth=auth)
        result = execute_live_intent(api, intent)
        st.session_state.shipra_auth = result.auth
        count_only = bool(
            re.search(
                r"\b(how many|count|total|kitne|kitny|kitna|kitni|kul)\b|کتنے|كم|عدد",
                question,
                re.IGNORECASE,
            )
        )
        return format_live_result(
            result,
            intent.language,
            count_only=count_only,
        )
    except (ShipraAPIError, ValueError) as exc:
        return f"Shipra live-data request stopped safely: `{exc}`"

st.title("Shipra Code Assistant")
st.caption("Public guest mode • Chats are temporary and never shared")
if "history" not in st.session_state: st.session_state.history = []
with st.sidebar:
    if st.button("New chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.pop("order_page", None)
        st.session_state.pop("pending_order_intent", None)
        st.session_state.pop("pending_order_question", None)
        st.rerun()
    st.caption("Code answers use the indexed source snapshot.")
    st.divider()
    st.subheader("Connect Shipra")
    if st.session_state.get("shipra_auth"):
        st.success(f"Connected as {st.session_state.shipra_auth.get('username', 'Shipra user')}")
        if st.button("Disconnect Shipra", use_container_width=True):
            st.session_state.pop("shipra_auth", None)
            st.rerun()
    else:
        with st.form("shipra_login", clear_on_submit=True):
            username = st.text_input("Shipra username")
            password = st.text_input("Shipra password", type="password")
            connect = st.form_submit_button("Connect account", use_container_width=True)
        if connect:
            if not username or not password:
                st.error("Username and password are required.")
            else:
                try:
                    st.session_state.shipra_auth = ShipraAPI(shipra_base_url()).login(username, password)
                    st.success("Shipra account connected.")
                    st.rerun()
                except ShipraAPIError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("Could not connect to Shipra API.")
        st.caption("Credentials are used for login only and are not stored in chat history.")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

page_state = st.session_state.get("order_page")
if page_state:
    current_page = int(page_state.get("page", 0))
    total_pages = int(page_state.get("total_pages", 1))
    previous_column, page_column, next_column = st.columns([1, 2, 1])

    with previous_column:
        if st.button(
            "← Previous",
            disabled=current_page <= 0,
            use_container_width=True,
        ):
            response = structured_live_answer(
                "previous",
                st.session_state.history,
            )
            st.session_state.history.append(
                {"role": "assistant", "content": response}
            )
            st.session_state.scroll_to_latest = True
            st.rerun()

    with page_column:
        st.markdown(
            f"<p style='text-align:center'>Page "
            f"<b>{current_page + 1}</b> of <b>{total_pages}</b></p>",
            unsafe_allow_html=True,
        )

    with next_column:
        if st.button(
            "Next →",
            disabled=current_page + 1 >= total_pages,
            use_container_width=True,
        ):
            response = structured_live_answer(
                "next",
                st.session_state.history,
            )
            st.session_state.history.append(
                {"role": "assistant", "content": response}
            )
            st.session_state.scroll_to_latest = True
            st.rerun()

if st.session_state.pop("scroll_to_latest", False):
    components.html(
        """
        <script>
        setTimeout(() => {
            window.frameElement.scrollIntoView({
                behavior: 'smooth',
                block: 'end'
            });

            const doc = window.parent.document;
            const container =
                doc.querySelector('[data-testid="stMain"]') ||
                doc.querySelector('section.main');

            if (container) {
                container.scrollTo({
                    top: container.scrollHeight,
                    behavior: 'smooth'
                });
            } else {
                window.parent.scrollTo({
                    top: doc.body.scrollHeight,
                    behavior: 'smooth'
                });
            }
        }, 200);
        </script>
        """,
        height=0,
    )

if q := st.chat_input("Ask about Shipra code..."):
    navigation_words = {
        "next",
        "next page",
        "agla",
        "agla page",
        "previous",
        "previous page",
        "back",
        "pichla",
        "pichla page",
    }
    if " ".join(q.lower().strip().split()) not in navigation_words:
        st.session_state.pop("order_page", None)

    st.session_state.history.append({"role":"user","content":q})
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching verified source..."):
            try:
                pending = st.session_state.get("pending_order_intent")
                is_navigation = " ".join(q.lower().strip().split()) in navigation_words

                # Clarification aur page navigation ko existing order state handle karegi.
                if pending or is_navigation:
                    out = structured_live_answer(q, st.session_state.history)
                else:
                    try:
                        request_route = parse_request_route(
                            st.secrets["GROQ_API_KEY"],
                            q,
                        )
                    except Exception:
                        request_route = None

                    routed_q = (
                        request_route.normalized_question
                        if request_route and request_route.confidence >= 0.75
                        else q
                    )
                    route = (
                        request_route.route
                        if request_route and request_route.confidence >= 0.75
                        else None
                    )

                    if route == "order_live":
                        out = structured_live_answer(
                            routed_q,
                            st.session_state.history,
                        )
                    elif route == "store_live":
                        out = structured_store_answer(routed_q)
                    elif route == "other_live":
                        out = structured_universal_live_answer(
                            routed_q,
                            st.session_state.history,
                        )
                    elif route == "rag":
                        out = answer(q, st.session_state.history[-8:])
                    else:
                        # Router unavailable ho to purana safe flow fallback rahega.
                        out = structured_store_answer(q)
                        if out is None:
                            out = structured_live_answer(q, st.session_state.history)
                        if out is None:
                            out = structured_universal_live_answer(
                                q,
                                st.session_state.history,
                            )

                    if out is None:
                        out = answer(q, st.session_state.history[-8:])
            except Exception as exc:
                out = (
                    f"Source search failed safely: "
                    f"`{type(exc).__name__}: {str(exc)}`"
                )
        st.markdown(out)
    st.session_state.history.append({"role":"assistant","content":out})
    st.session_state.scroll_to_latest = True
    st.rerun()
