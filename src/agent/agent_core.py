# %%
import os
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import START, END, StateGraph
# importing necessary functions from dotenv library
from dotenv import load_dotenv
# loading variables from .env file
load_dotenv() 

# ─── Configuration ──────────────────────────────────────────────────────────────

CHROMA_DB_PATH = "src/utils/vectorstore/db_chroma"
COLLECTION_NAME = "v_db"


# ─── State Definition ────────────────────────────────────────────────────────────
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# ─── Initialize LLM and Vector Store ─────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
doc_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DB_PATH,
    embedding_function=doc_embeddings,  # Retrieval-only
)


# ─── Agent Functions ─────────────────────────────────────────────────────────────
def retrieve(state: State):
    """Fetch top-3 similar document chunks."""
    docs = db.similarity_search(query=state["question"], k=3)
    return {"context": docs}


def generate(state: State):
    """Generate answer based on retrieved context."""
    # Create proper message objects for ChatGoogleGenerativeAI
    system_prompt = (
        "You are an assistant. Use the context below to answer the question.\n\n"
        f"Context:\n{chr(10).join(d.page_content for d in state['context'])}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"]),
    ]

    response = llm.invoke(messages)
    return {"answer": response}


# ─── Build and Compile Graph ─────────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()


def ask_rag(question: str):
    """Run the RAG agent for a single query."""
    for event in graph.stream({"question": question}, stream_mode="values"):
        if event.get("answer"):
            ans = event["answer"]
            return ans.content if hasattr(ans, "content") else ans
    return None
