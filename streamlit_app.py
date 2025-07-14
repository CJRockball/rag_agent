import sys
import os
import time
import streamlit as st
from src.utils.security import validate_api_key
from src.utils.rate_limit import allow_request, record_request

__import__("pysqlite3")

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# ─── Streamlit Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="💬",
    layout="centered",
)


# ─── Initialize Environment Variables at Startup ──────────────────────────
@st.cache_resource
def initialize_environment():
    """Initialize environment variables from Streamlit secrets at startup"""
    try:
        # Set API key to environment variable once at startup
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

        # Set other configuration from secrets
        os.environ["CHROMA_DB_PATH"] = st.secrets.get(
            "CHROMA_DB_PATH", "src/utils/vectorstore/db_chroma"
        )
        os.environ["COLLECTION_NAME"] = st.secrets.get(
            "COLLECTION_NAME", "v_db"
        )
        os.environ["DOC_PATH"] = st.secrets.get(
            "DOC_PATH", "pdfs/Tender_Specs_ECDC2024OP0017_V1.pdf"
        )

        st.success("✅ Environment initialized successfully!")
        return True

    except KeyError as e:
        st.error(f"❌ Missing secret: {e}")
        st.info(
            "💡 Please add the required secrets in \
            Streamlit Cloud deployment settings"
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Environment initialization failed: {str(e)}")
        st.stop()


# ─── Database Initialization ───────────────────────────────────────────────
@st.cache_resource
def get_database_connection():
    """Initialize and return cached database connection"""
    from src.utils.load_db import setup_vector_database

    # Now we can use os.environ since it's set at startup
    doc_path = os.environ.get("DOC_PATH")
    db_path = os.environ.get("CHROMA_DB_PATH")
    collection_name = os.environ.get("COLLECTION_NAME")

    with st.spinner("🔄 Initializing database..."):
        db = setup_vector_database(doc_path, db_path, collection_name)
        st.success("✅ Database initialized successfully!")
        return db


# ─── App Initialization ────────────────────────────────────────────────────
# Initialize environment variables first
initialize_environment()

# Validate API key (now reads from os.environ)
validate_api_key()

# Initialize database
db = get_database_connection()

st.title("📄 PDF RAG Chat")
st.markdown("Ask questions about the indexed documents:")

# ─── Chat Interface ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

can_ask = allow_request()
if not can_ask:
    wait = 60 - int(time.time() - st.session_state.rl_start)
    st.warning(f"Rate limit hit – try again in {wait}s")

prompt = st.chat_input("Your question…", disabled=not can_ask)

if prompt and can_ask:
    record_request()
    st.session_state.history.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.spinner("Searching…"):
        # Import here to avoid circular imports
        from src.agent.agent_core import ask_rag_with_connection

        reply = (
            ask_rag_with_connection(prompt, db)
            or "No relevant information found."
        )

    st.session_state.history.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)
