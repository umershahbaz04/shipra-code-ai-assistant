import os
from pathlib import Path
import streamlit as st
from groq import Groq
from rag_engine import ShipraRag

st.set_page_config(page_title="Shipra Code Assistant", page_icon="🤖", layout="wide")

@st.cache_resource
def load_engine(): return ShipraRag(Path(__file__).resolve().parent / "data")

def answer(question, history):
    evidence = load_engine().retrieve(question)
    context = ShipraRag.context(evidence)
    prompt = f'''You are Shipra Code Assistant. Answer in the user's language.
Only use the supplied indexed Shipra source snapshot. Never invent a file, route, API, UI control, handler, repository call, or business fact.
Every factual statement must end in one or more citations like [S1].
If the source does not prove the answer, say exactly: "Is indexed source snapshot mein yeh verify nahi hua." Do not guess.
For a short list question, answer directly; do not add scenario steps.
For a code-change question, separate verified existing code from proposed code and label every proposed file/path as an assumption.
Keep the answer concise. Then add a heading `### Verified sources` with the cited paths.

QUESTION: {question}

SOURCES:\n{context}'''
    client = Groq(api_key=st.secrets["GROQ_API_KEY"], timeout=25, max_retries=1)
    result = client.chat.completions.create(model="openai/gpt-oss-20b", messages=[{"role":"user","content":prompt}], temperature=0.1)
    text = result.choices[0].message.content or "Is indexed source snapshot mein jawab generate nahi hua."
    sources = "\n".join(f"- [S{i}] `{e.meta.get('file_path')}` — lines {e.meta.get('start_line')}-{e.meta.get('end_line')}" for i,e in enumerate(evidence,1))
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
