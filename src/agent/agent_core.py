import os
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langgraph.graph import START, END, StateGraph

# from langchain_core.rate_limiters import InMemoryRateLimiter

# ─── Configuration ──────────────────────────────────────────────────────────
CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH", "src/utils/vectorstore/db_chroma"
)
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "v_db")


# ─── State Definition ───────────────────────────────────────────────────────
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    db: Chroma  # Properly typed as Chroma instead of generic object


# ─── Initialize LLM ─────────────────────────────────────────────────────────
# Now we can use os.environ directly since it's set at startup
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# rate_limiter = InMemoryRateLimiter(
#         # Super slow! request once every 10 seconds!!
#         requests_per_second=0.1,
#         # Wake up every 100 ms to check whether allowed to make a request,
#         check_every_n_seconds=0.1,
#         max_bucket_size=10,)  # Controls the maximum burst size.)

llm = ChatGoogleGenerativeAI(
    model="Gemini 2.0 Flash-Lite",
    temperature=0.1,
)


# ─── Agent Functions ────────────────────────────────────────────────────────
def retrieve(state: State):
    """Fetch top-3 similar document chunks using provided database"""
    # Now type checker knows state["db"] is a Chroma instance
    docs = state["db"].similarity_search(query=state["question"], k=3)
    return {"context": docs}


def generate(state: State):
    """Generate answer based on retrieved context"""
    system_prompt = (
        "You are an assistant. Use the context below to \
            answer the question.\n\n"
        f"Context: \n{chr(10).join(d.page_content for d in state['context'])}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"]),
    ]

    response = llm.invoke(messages)
    return {"answer": response}


# ─── Build Graph ────────────────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()


def ask_rag_with_connection(question: str, db: Chroma):
    """Run RAG agent with provided database connection"""
    # Also properly type the db parameter here
    for event in graph.stream(
        {"question": question, "db": db}, stream_mode="values"
    ):
        if event.get("answer"):
            ans = event["answer"]
            return ans.content if hasattr(ans, "content") else ans
    return None
