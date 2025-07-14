# Enhanced LangGraph PDF RAG Agent with Vector Store Setup

# Essential imports
import os
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)


# Test the database setup
def test_database(db_test):
    """Test if the vector database is working"""
    query = "what is the main deliverable?"
    docs = db_test.similarity_search(query)
    print(f"Found {len(docs)} relevant documents")
    if docs:
        print(f"First document preview: {docs[0].page_content[:200]}...")


# Vector Database Setup Function
def setup_vector_database(doc_path: str, db_path: str, collection_name: str):
    """Set up vector database with persistence check"""
    # Initialize LLM and embeddings
    doc_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    # Check if database already exists
    if os.path.exists(db_path):
        print("Database exists - connecting to existing database")
        db = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=doc_embeddings,
        )
        # test_database(db)
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
        # test_database(db)
    return db
