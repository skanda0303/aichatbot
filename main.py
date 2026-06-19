import os
import hashlib
import json
from fastapi import FastAPI
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

DOCS_DIR     = "docs"
HASH_FILE    = "docs_hash.json"   # tracks doc fingerprint to skip re-indexing
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL   = "http://localhost:11434"

# ── Helpers for persistent storage ──────────────────────────────────────────

def get_docs_fingerprint() -> str:
    """Compute an MD5 hash of every file in DOCS_DIR.
    If any file is added, removed, or modified, the hash changes."""
    hasher = hashlib.md5()
    for fname in sorted(os.listdir(DOCS_DIR)):
        fpath = os.path.join(DOCS_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                hasher.update(fname.encode())   # include filename
                hasher.update(f.read())          # include content
    return hasher.hexdigest()

def load_stored_fingerprint() -> str:
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            return json.load(f).get("hash", "")
    return ""

def save_fingerprint(h: str):
    with open(HASH_FILE, "w") as f:
        json.dump({"hash": h}, f)

# ── Vector Store Setup ───────────────────────────────────────────────────────

os.makedirs(DOCS_DIR, exist_ok=True)

embeddings  = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

if os.listdir(DOCS_DIR):
    current_fp = get_docs_fingerprint()
    stored_fp  = load_stored_fingerprint()
    existing_ids = vectorstore.get()["ids"]

    if current_fp == stored_fp and existing_ids:
        # Documents haven't changed — skip re-indexing entirely
        print(f"[OK] Documents unchanged — reusing existing index ({len(existing_ids)} chunks). Skipping ingestion.")
    else:
        # New or modified documents detected — re-index
        print("[INFO] Document changes detected. Re-indexing...")
        docs   = PyPDFDirectoryLoader(DOCS_DIR).load()
        
        # Clean up text extraction artifacts (e.g. word-by-word newlines in download3.pdf)
        for doc in docs:
            doc.page_content = " ".join(doc.page_content.split())
            
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

        # Clear old chunks first to avoid duplicates
        if existing_ids:
            vectorstore.delete(ids=existing_ids)

        vectorstore.add_documents(chunks)
        save_fingerprint(current_fp)   # save new fingerprint after successful index

        sources = set(c.metadata.get("source", "unknown") for c in chunks)
        print(f"[OK] Cleaned and Ingested {len(chunks)} chunks from {len(docs)} page(s).")
        print(f"[OK] Sources indexed: {[os.path.basename(s) for s in sources]}")
else:
    print("[INFO] docs/ is empty — no documents ingested.")

# ── Retrievers ───────────────────────────────────────────────────────────────

# ── MMR Retriever ────────────────────────────────────────────────────────────
# MMR (Maximum Marginal Relevance): diversity-aware so one doc can't monopolize results.
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,         # <-- reduced from 10 to 4 to prevent pulling in unrelated pages
        "fetch_k": 20,  # candidate pool to pick from
        "lambda_mult": 0.8  # <-- increased from 0.6 to 0.8 to focus heavily on exact relevance
    }
)

# ── Retriever Tool ───────────────────────────────────────────────────────────

@tool
def search_docs(query: str) -> str:
    """Search ALL uploaded documents for information relevant to the query.
    Always call this tool before answering any question.
    """
    results = retriever.invoke(query)

    if not results:
        return "No relevant information found in the documents."

    # Deduplicate by page content
    seen, unique_results = set(), []
    for doc in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_results.append(doc)

    sources_found = set(d.metadata.get("source", "unknown") for d in unique_results)
    print(f"[RETRIEVAL] Original: '{query}' | Sources hit: {[os.path.basename(s) for s in sources_found]} | Chunks: {len(unique_results)}")

    formatted = []
    for doc in unique_results:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page   = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)

# ── Agent ────────────────────────────────────────────────────────────────────

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0, think=False)

agent = create_agent(
    model=llm,
    tools=[search_docs],
    system_prompt=(
        "You are a helpful assistant with access to uploaded documents and the chat history.\n\n"
        "How to handle different types of user messages:\n"
        "1. For questions about the conversation itself, the chat history, greetings, or meta-questions "
        "(e.g., 'what did I ask you before?', 'who are you?', 'hello'):\n"
        "   - Answer directly based on the conversation history or context.\n"
        "   - Do NOT try to search the documents or output document-only errors for these conversational queries.\n\n"
        "2. For questions asking about facts, data, people, or topics covered by documents:\n"
        "   - You MUST call the search_docs tool first.\n"
        "   - Your answer must be built EXCLUSIVELY from the text returned by search_docs.\n"
        "   - Do NOT add, infer, or recall anything from your own pre-trained knowledge.\n"
        "   - Do not be overly cautious: if the retrieved text discusses the topic, answer the user's question using it, "
        "even if the spelling or wording in the user's question differs slightly from the document (e.g. 'focussed' vs 'focused').\n"
        "   - If the search_docs results do not contain the answer, respond with: "
        "'The documents do not contain information about this topic.'\n"
        "   - Always cite which document and page your answer comes from."
    ),
)

# ── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Chatbot")

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    history  = SQLChatMessageHistory(session_id=req.session_id, connection_string="sqlite:///memory.db")
    messages = history.messages + [HumanMessage(content=req.message)]

    result = agent.invoke({"messages": messages})
    answer = result["messages"][-1].content

    history.add_message(HumanMessage(content=req.message))
    history.add_message(AIMessage(content=answer))

    return {"session_id": req.session_id, "answer": answer}

@app.get("/")
def root():
    from fastapi.responses import FileResponse
    return FileResponse("index.html")
