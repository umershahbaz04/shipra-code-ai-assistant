import os
from pathlib import Path
import re

import streamlit as st
from groq import Groq
from rag_engine import ShipraRag

st.set_page_config(page_title="Shipra Code Assistant", page_icon="🤖", layout="wide")

@st.cache_resource
def load_engine(): return ShipraRag(Path(__file__).resolve().parent / "data")

def response_language(question: str) -> str:
    """Choose the response language from an explicit request, then the question."""
    normalized = question.lower()

    # An explicitly requested language always wins, even for an English question.
    if re.search(r"\b(answer|reply|explain|response|jawab|samjha(?:o|do)|bata(?:o|do))\b.{0,50}\b(arabic|arab|العربية)\b", normalized):
        return "Arabic"
    if re.search(r"\b(answer|reply|explain|response|jawab|samjha(?:o|do)|bata(?:o|do))\b.{0,50}\b(roman\s*urdu|urdu|اردو)\b", normalized):
        return "Roman Urdu"
    if re.search(r"\b(answer|reply|explain|response|jawab|samjha(?:o|do)|bata(?:o|do))\b.{0,50}\benglish\b", normalized):
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

st.title("Shipra Code Assistant")
st.caption("Public guest mode • Chats are temporary and never shared")
if "history" not in st.session_state: st.session_state.history = []
with st.sidebar:
    if st.button("New chat", use_container_width=True):
        st.session_state.history = []; st.rerun()
    st.caption("Answers are grounded in the indexed source snapshot.")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])
if q := st.chat_input("Ask about Shipra code..."):
    st.session_state.history.append({"role":"user","content":q})
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching verified source..."):
            try: out = answer(q, st.session_state.history[-8:])
            except Exception as exc: out = f"Source search failed safely: `{type(exc).__name__}`"
        st.markdown(out)
    st.session_state.history.append({"role":"assistant","content":out})
