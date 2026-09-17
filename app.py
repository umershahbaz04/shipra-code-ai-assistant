import os
import re
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
from groq import Groq
from rag_engine import ShipraRag
from shipra_api import (
    ShipraAPI,
    ShipraAPIError,
    format_order_detail,
    format_order_rows,
    filter_in_progress_rows,
    live_order_intent,
)

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
    evidence = load_engine().retrieve(question)
    context = ShipraRag.context(evidence)
    language = response_language(question)
    prompt = f'''You are Shipra Code Assistant.
Your required response language is: {language}.
Language rules: English means English only. Arabic means Arabic script only. Roman Urdu means Latin/Roman Urdu only. Never write Hindi or Devanagari script. If the user explicitly requests a language, follow that request even when their question is written in another language.
Only use the supplied indexed Shipra source snapshot. Never invent a file, route, API, UI control, handler, repository call, or business fact.
Every factual statement must end in one or more citations like [S1].
If the source does not prove the answer, clearly say it is not verified by the indexed source snapshot, in the required response language. Do not guess.
For a short list question, answer directly; do not add scenario steps.
For a code-change question, separate verified existing code from proposed code and label every proposed file/path as an assumption.
Keep the answer concise. Do not add a "Verified sources", "Sources", or references section. The application renders only the source paths that your answer actually cites.

QUESTION: {question}

SOURCES:\n{context}'''
    client = Groq(api_key=st.secrets["GROQ_API_KEY"], timeout=25, max_retries=1)
    result = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role":"system", "content":"Follow the required response language exactly. Never output Devanagari/Hindi script."}, {"role":"user", "content":prompt}],
        temperature=0.1,
    )
    text = result.choices[0].message.content or "Is indexed source snapshot mein jawab generate nahi hua."

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

def _live_date_range(question: str) -> tuple[date | None, date | None]:
    today = date.today()
    text = question.lower()
    last_days = re.search(r"\b(?:last|past|pichl[aeiy]*)\s+(\d{1,3})\s+(?:days?|din)\b", text)
    if last_days:
        days = max(1, min(int(last_days.group(1)), 366))
        # Inclusive range: "last 2 days" means today and yesterday.
        return today - timedelta(days=days - 1), today
    if re.search(r"\b(yesterday|kal)\b", text):
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    if re.search(r"\b(today|aaj|aj)\b", text):
        return today, today
    return None, None

def _has_recent_order_context(history) -> bool:
    for message in reversed(history[-6:]):
        if message.get("role") == "user" and re.search(r"\b(order|orders|آرڈر|طلبات|الطلبات)\b", message.get("content", ""), re.IGNORECASE):
            return True
    return False

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

def live_answer(question: str, history) -> str | None:
    intent = live_order_intent(question, order_context=_has_recent_order_context(history[:-1]))
    if not intent:
        return None
    language = response_language(question)
    auth = st.session_state.get("shipra_auth")
    if not auth:
        return _connect_message(language)

    api = ShipraAPI(shipra_base_url(), auth=auth)
    from_date, to_date = _live_date_range(question)
    try:
        if intent["action"] == "count":
            count, updated_auth = api.count_orders(intent["kind"], from_date, to_date)
            st.session_state.shipra_auth = updated_auth
            labels = {
                "total": "orders",
                "delivered": "delivered orders",
                "in_progress": "in-progress orders",
                "returned": "returned orders",
                "regular": "regular orders",
                "fulfillable": "fulfillable orders",
                "to_be_packed": "orders to be packed",
                "to_be_shipped": "orders to be shipped",
            }
            label = labels[intent["kind"]]
            scope = _date_scope(from_date, to_date, language)
            if language == "Arabic":
                arabic_labels = {"total": "طلباً", "delivered": "طلباً تم تسليمه", "in_progress": "طلباً قيد التنفيذ", "returned": "طلباً مرتجعاً"}
                return f"وفقاً لواجهة Shipra المباشرة، يوجد **{count} {arabic_labels.get(intent['kind'], 'طلباً')}** خلال {scope}."
            if language == "Roman Urdu":
                return f"Live Shipra API ke mutabiq {scope} **{count} {label}** hain."
            return f"According to the live Shipra API, there are **{count} {label}** {scope}."

        if intent["action"] == "detail":
            payload, updated_auth = api.order_by_id(intent["order_id"])
            st.session_state.shipra_auth = updated_auth
            return format_order_detail(payload)

        if intent["action"] == "detail_search":
            payload, updated_auth = api.order_by_reference(intent["reference"])
            st.session_state.shipra_auth = updated_auth
            return format_order_detail(payload)

        data, updated_auth = api.list_orders(from_date, to_date)
        st.session_state.shipra_auth = updated_auth
        labels = {"total": "orders", "in_progress": "pending/in-progress orders", "delivered": "delivered orders", "returned": "returned orders"}
        if intent["kind"] == "in_progress":
            exact_count, updated_auth = api.count_orders("in_progress", from_date, to_date)
            st.session_state.shipra_auth = updated_auth
            data["rows"] = filter_in_progress_rows(data.get("rows") or [])
            data["count"] = exact_count
        return format_order_rows(data, limit=50, language=language, label=labels.get(intent["kind"], "orders"))
    except ShipraAPIError as exc:
        return f"Shipra live-data request failed safely: `{exc}`"

st.title("Shipra Code Assistant")
st.caption("Public guest mode • Chats are temporary and never shared")
if "history" not in st.session_state: st.session_state.history = []
with st.sidebar:
    if st.button("New chat", use_container_width=True):
        st.session_state.history = []; st.rerun()
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
if q := st.chat_input("Ask about Shipra code..."):
    st.session_state.history.append({"role":"user","content":q})
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching verified source..."):
            try:
                out = live_answer(q, st.session_state.history)
                if out is None:
                    out = answer(q, st.session_state.history[-8:])
            except Exception as exc: out = f"Source search failed safely: `{type(exc).__name__}`"
        st.markdown(out)
    st.session_state.history.append({"role":"assistant","content":out})
