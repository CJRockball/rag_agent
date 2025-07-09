import streamlit as st
from src.agent.agent_core import ask_rag

# ─── Streamlit Configuration ────────────────────────────────────────────────────
st.set_page_config(page_title="PDF RAG Chat", page_icon="💬", layout="centered")
st.title("📄 PDF RAG Chat")
st.markdown("Ask questions about the indexed documents:")

# ─── Initialize Chat History ─────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Display Chat History ────────────────────────────────────────────────────────
for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ─── Handle User Input and Response ─────────────────────────────────────────────
if prompt := st.chat_input("Your question…"):
    # Store and display user message
    st.session_state.history.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Generate assistant reply
    with st.spinner("Searching…"):
        reply = ask_rag(prompt) or "No relevant information found. Please rephrase."
    st.session_state.history.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)
