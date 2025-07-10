import time
import streamlit as st
from src.agent.agent_core import ask_rag
from src.utils.security import validate_api_key
from src.utils.rate_limit import allow_request, record_request


# ─── Streamlit Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chat", page_icon="💬", layout="centered"
)

# Validate api key
validate_api_key()

st.title("📄 PDF RAG Chat")
st.markdown("Ask questions about the indexed documents:")

# ─── Initialize Chat History ────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Display Chat History ───────────────────────────────────────────────────
for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ─── Rate-limit gate & single chat_input ────────────────────────────────────
can_ask = allow_request()
if not can_ask:
    wait = 60 - int(time.time() - st.session_state.rl_start)
    st.warning(f"Rate limit hit – try again in {wait}s")
prompt = st.chat_input(
    "Your question…",
    disabled=not can_ask,  # greyed-out while blocked
    key="user_prompt",  # explicit key avoids clashes
)

# ─── Handle the prompt when it exists and limit allows ─────────────────────
if prompt and can_ask:
    record_request()  # count this call
    st.session_state.history.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.spinner("Searching…"):
        reply = ask_rag(prompt) or "No relevant information found."
    st.session_state.history.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)
