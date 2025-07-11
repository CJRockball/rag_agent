# Enhanced LangGraph PDF RAG Agent with Vector Store Setup

# Essential imports
import os
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
import streamlit as st


# Replace with Streamlit secrets management
def get_google_api_key():
    """Get Google API key from Streamlit secrets"""
    print(st.secrets["CHROMA_DB_PATH"])
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("❌ GOOGLE_API_KEY not found in Streamlit secrets")
        st.stop()


# Set the API key for Google services
os.environ["GOOGLE_API_KEY"] = get_google_api_key()

# Configuration using Streamlit secrets with fallbacks
DOC_PATH = st.secrets.get(
    "DOC_PATH", "pdfs/Tender_Specs_ECDC2024OP0017_V1.pdf"
)
CHROMA_DB_PATH = st.secrets.get(
    "CHROMA_DB_PATH", "src/utils/vectorstore/db_chroma"
)
COLLECTION_NAME = st.secrets.get("COLLECTION_NAME", "v_db")


# State definition for the agent
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Initialize LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
doc_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)


# Vector Database Setup Function
def setup_vector_database(doc_path: str, db_path: str, collection_name: str):
    """Set up vector database with persistence check"""

    # Check if database already exists
    if os.path.exists(db_path):
        print("Database exists - connecting to existing database")
        db = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=doc_embeddings,
        )
    else:
        print("Database does not exist - creating new database")

        # Load and process PDF
        loader = PyPDFLoader(doc_path)
        pages = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)

        # Create vector store from documents
        db = Chroma.from_documents(
            chunks,
            doc_embeddings,
            persist_directory=db_path,
            collection_name=collection_name,
        )
        print("Database created and populated successfully")

    return db


# Initialize the vector database
db = setup_vector_database(DOC_PATH, CHROMA_DB_PATH, COLLECTION_NAME)


# Test the database setup
def test_database():
    """Test if the vector database is working"""
    query = "what is the main deliverable?"
    docs = db.similarity_search(query)
    print(f"Found {len(docs)} relevant documents")
    if docs:
        print(f"First document preview: {docs[0].page_content[:200]}...")
