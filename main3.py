import os
import time
import hashlib
import json
import asyncio
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.tools import tool

# ── Configuration ──────────────────────────────────────────────────────────────
DOCS_DIR             = "docs"
HASH_FILE            = "docs_hash.json"
MAX_AGENT_ITERATIONS = 6    # max tool-call rounds before forcing a final answer
MAX_WEB_RESULTS      = 5  #  results per search
MAX_HISTORY_MESSAGES = 6    # recent chat messages injected into agent context
RETRIEVER_K          = 12   # chunks per retriever in hybrid search

# ── Document fingerprinting ────────────────────────────────────────────────────
def get_docs_fingerprint() -> str:
    hasher = hashlib.md5()
    for fname in sorted(os.listdir(DOCS_DIR)):
        fpath = os.path.join(DOCS_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                hasher.update(fname.encode())
                hasher.update(f.read())
    return hasher.hexdigest()

def load_stored_fingerprint() -> str:
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            return json.load(f).get("hash", "")
    return ""

def save_fingerprint(h: str):
    with open(HASH_FILE, "w") as f:
        json.dump({"hash": h}, f)

os.makedirs(DOCS_DIR, exist_ok=True)

# ── Vector store & ingestion ───────────────────────────────────────────────────
embeddings  = OllamaEmbeddings(model="bge-m3")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="bge_m3"
)

chunks: list[Document] = []

if os.listdir(DOCS_DIR):
    current_fp   = get_docs_fingerprint()
    stored_fp    = load_stored_fingerprint()
    existing_ids = vectorstore.get()["ids"]

    if current_fp == stored_fp and existing_ids:
        print(f"[OK] Documents unchanged — reusing existing index ({len(existing_ids)} chunks).")
        existing_data = vectorstore.get()
        if existing_data and "documents" in existing_data:
            for doc_text, metadata in zip(existing_data["documents"], existing_data["metadatas"]):
                chunks.append(Document(page_content=doc_text, metadata=metadata))
    else:
        print("[INFO] Document changes detected. Re-indexing...")
        raw_docs = PyPDFDirectoryLoader(DOCS_DIR).load()

        grouped: dict = defaultdict(list)
        for d in raw_docs:
            grouped[d.metadata.get("source", "unknown")].append(
                (d.metadata.get("page", 0), d.page_content)
            )

        merged_docs = []
        for src, pages in sorted(grouped.items()):
            pages.sort(key=lambda x: x[0])
            full_text = " ".join(" ".join(c for _, c in pages).split())
            merged_docs.append(Document(page_content=full_text, metadata={"source": src, "page": "0"}))

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        ).split_documents(merged_docs)

        # Batch document addition to prevent Ollama runner crashes on large batch requests
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            vectorstore.add_documents(chunks[i : i + batch_size])
        save_fingerprint(current_fp)
        print(f"[OK] Ingested {len(chunks)} chunks from {len(merged_docs)} file(s).")
else:
    print("[INFO] docs/ is empty — no documents ingested.")

# ── Hybrid retriever ───────────────────────────────────────────────────────────
if chunks:
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = RETRIEVER_K
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K}
    )
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )
    print("[OK] Hybrid retriever initialized — BM25 0.3 / Vector 0.7")
else:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print("[WARNING] No documents found. Defaulting to vector retriever.")

# ── Redundancy filter ──────────────────────────────────────────────────────────
def _word_set(text: str) -> set:
    return set(w.strip(".,;:()[]●-●*").lower() for w in text.split() if len(w.strip(".,;:()[]●-●*")) > 1)

def filter_redundant_docs(docs: list, threshold: float = 0.85) -> list:
    unique: list[Document] = []
    for doc in docs:
        words = _word_set(doc.page_content)
        redundant = False
        for u in unique:
            u_words = _word_set(u.page_content)
            if not words or not u_words:
                continue
            if len(words & u_words) / min(len(words), len(u_words)) > threshold:
                redundant = True
                break
        if not redundant:
            unique.append(doc)
    return unique

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY", ""),
    temperature=0.2
)

# ── Tool definitions ───────────────────────────────────────────────────────────
@tool
def rag_search(query: str) -> str:
    """Search the uploaded PDF documents for information related to the query.
    Always call this first for any question that might be in the documents.
    Returns 'NO_RELEVANT_DOCS' if nothing useful is found — in that case, use web_search as fallback."""
    if not chunks:
        return "NO_RELEVANT_DOCS: No documents have been uploaded."
    try:
        results = retriever.invoke(query)
    except Exception as e:
        return f"NO_RELEVANT_DOCS: Retrieval error — {e}"

    if not results:
        return "NO_RELEVANT_DOCS: No matching documents found."

    seen, unique = set(), []
    for doc in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)

    filtered = filter_redundant_docs(unique, threshold=0.85)
    if not filtered:
        return "NO_RELEVANT_DOCS: All retrieved documents were redundant."

    formatted = "\n\n---\n\n".join(
        doc.page_content.replace("●", "\n- ")
        for doc in filtered[:8]
    )
    print(f"  [RAG TOOL] Returned {len(filtered[:8])} chunks.")
    return formatted

def _fetch_url(url: str) -> str:
    """Helper function to fetch and clean webpage content."""
    try:
        import urllib.request
        from bs4 import BeautifulSoup
        import re

        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=6) as response:
            html = response.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        # Get text and clean it up
        text = soup.get_text(separator=' ')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Compress multiple newlines/spaces
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Cap to 12000 characters to keep it within context bounds
        if len(text) > 12000:
            text = text[:12000] + "... [truncated]"
        return text
    except Exception as e:
        return f"Failed to fetch webpage: {e}"

@tool
def web_search(query: str) -> str:
    """Search the web for real-time information, latest news, current events, or facts
    not found in the documents. Use as a fallback when rag_search returns NO_RELEVANT_DOCS,
    or directly for queries about current events / live data."""
    try:
        import urllib.request
        import json

        print(f"  [WEB TOOL] Querying Tavily API for: '{query}'")
        
        tavily_url = "https://api.tavily.com/search"
        payload = {
            "api_key": "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",
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "max_results": 5
        }
        headers = {
            "Content-Type": "application/json"
        }
        req = urllib.request.Request(
            tavily_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))

        search_results = res_data.get("results", [])
        
        # Filter out video-based and social media platforms (youtube, tiktok, etc.)
        VIDEO_BLACKLIST = ["youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "tiktok.com", "instagram.com"]
        filtered_results = [
            res for res in search_results
            if not any(blacklisted in res.get("url", "").lower() for blacklisted in VIDEO_BLACKLIST)
        ]
        
        if not filtered_results:
            return "No web results found for this query."

        print(f"  [WEB TOOL] Tavily returned {len(filtered_results)} valid non-video results. Auto-fetching top 4 pages...")

        formatted_sources = []
        for i, res in enumerate(filtered_results[:4]): # Format and fetch top 4 sources
            url = res.get("url", "")
            title = res.get("title", "")
            snippet = res.get("content", "")
            score = res.get("score", 0.0)
            
            # Auto-fetch full page for top 4 sources
            if i < 4 and url:
                print(f"    [WEB TOOL] Auto-fetching page {i+1}: {url}")
                page_content = _fetch_url(url)
                if page_content and not page_content.startswith("Failed to fetch webpage"):
                    snippet = page_content
                    print(f"    [WEB TOOL] Successfully fetched {len(page_content)} characters.")
                else:
                    print(f"    [WEB TOOL] Fetch failed: {page_content[:60]}... Using snippet fallback.")

            formatted_sources.append(
                f"=== SOURCE {i+1} ===\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Source: Tavily Search | Relevance Score: {score}\n"
                f"Content:\n{snippet}"
            )

        return "\n\n---\n\n".join(formatted_sources)
    except Exception as e:
        return f"Web search failed: {e}"

@tool
def fetch_webpage_content(url: str) -> str:
    """Fetch and read the full text content of a specific webpage URL to get detailed, in-depth information
    about a news article, release, or technical documentation. Always use this when the search results/snippets
    do not contain enough detailed information. The tool_input must be a valid URL string starting with http or https."""
    return _fetch_url(url)

@tool
def get_datetime() -> str:
    """Returns the current date, time, and day of the week. Use for any question about today's date,
    current time, what day it is, etc."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')}"

# ── Agent setup ────────────────────────────────────────────────────────────────
agent_tools = [web_search, fetch_webpage_content, get_datetime]

agent_executor = create_agent(
    model=llm,
    tools=agent_tools,
    system_prompt=(
        f"You are a precise, fact-grounded assistant. Current date is {datetime.now().strftime('%A, %B %d, %Y')}.\n\n"
        "SYNTHESIS RULES:\n"
        "1. Answer directly from facts in tool results or pre-fetched RAG context — never tell the user to visit a site.\n"
        "2. Be comprehensive and detailed. Use bullet points or bold text where helpful.\n"
        "3. Trust tool outputs over your training memory.\n"
        "4. Do not repeat identical search queries. However, you are highly encouraged to do multi-step searching or call 'fetch_webpage_content' on multiple URLs to gather complete and detailed timelines when requested.\n"
        "5. If you need detailed/specific info from a webpage URL returned by web_search, call 'fetch_webpage_content' with that exact URL in tool_input.\n"
        "6. CHRONOLOGICAL TIMELINES: When asked for a timeline or chronological order of releases/events, carefully verify release months and dates for each model and list them in strict order from oldest to newest with detailed descriptions.\n"
        "7. WEB FALLBACK: If the 'Pre-fetched RAG context' does not contain the answer, is irrelevant to the query, or is blank, you MUST call the 'web_search' tool to find the information on the web. Never simply say the text does not contain the information."
    )
)

def sanitize_tool_output(text: str) -> str:
    # Clean up excessive whitespace
    text = " ".join(text.split())
    # Cap at 25000 chars — keeps context small enough for the model to respond reliably without losing critical search details
    if len(text) > 25000:
        text = text[:25000] + "... [truncated]"
    return text

# ── Agent loop ─────────────────────────────────────────────────────────────────
async def run_agent(query: str, history_messages: list) -> str:
    print(f"[AGENT] Starting query routing. Query: '{query}'")

    # ── Step 1: Single pre-fetch from RAG ───────────────────
    rag_context = ""
    has_rag_docs = False
    if chunks:
        try:
            t_ret = time.perf_counter()
            retrieved_docs = retriever.invoke(query)
            print(f"[AGENT] Retrieval done in {time.perf_counter() - t_ret:.3f}s — {len(retrieved_docs)} raw docs")
            if retrieved_docs:
                seen, unique = set(), []
                for doc in retrieved_docs:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        unique.append(doc)
                filtered = filter_redundant_docs(unique, threshold=0.85)
                if filtered:
                    has_rag_docs = True
                    formatted_chunks = [doc.page_content.replace("●", "\n- ") for doc in filtered[:3]]
                    rag_context = "\n\n---\n\n".join(formatted_chunks)
                    print(f"[AGENT] Injecting {len(filtered[:3])} RAG chunks directly.")
        except Exception as e:
            print(f"[AGENT] Retrieval error: {e}. Continuing without RAG context.")

    # ── Step 2: Incorporate pre-fetched RAG context into input if present ───
    user_input = query
    if has_rag_docs:
        sanitized_rag = sanitize_tool_output(rag_context)
        user_input = (
            f"Pre-fetched RAG context:\n{sanitized_rag}\n\n"
            f"Now answer the query: {query}"
        )

    # ── Step 3: Run the LangChain AgentExecutor / CompiledStateGraph ───────────
    try:
        inputs = {
            "messages": list(history_messages[-MAX_HISTORY_MESSAGES:]) + [HumanMessage(content=user_input)]
        }
        response = await agent_executor.ainvoke(inputs)
        final_message = response["messages"][-1]
        content = final_message.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            return "".join(text_parts)
        return str(content)
    except Exception as e:
        print(f"[AGENT ERROR] agent_executor execution failed: {e}")
        return f"An error occurred while executing the agent: {e}"

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Agentic RAG Chatbot — Port 8002")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    async def response_generator():
        history = SQLChatMessageHistory(
            session_id=req.session_id,
            connection_string="sqlite:///memory3.db"
        )

        t_start = time.perf_counter()
        try:
            final_answer = await run_agent(req.message, history.messages)
        except Exception as e:
            print(f"[ERROR] run_agent raised an exception: {e}")
            final_answer = "Sorry, I encountered an internal error. Please try again."
        print(f"[TIMER] Total agent time: {time.perf_counter() - t_start:.3f}s")

        # Guard: ensure we always have something to stream
        if not final_answer or not final_answer.strip():
            print("[WARNING] final_answer is empty — sending fallback message.")
            final_answer = "I'm sorry, I wasn't able to generate a response. Please try rephrasing your question."

        print(f"[STREAM] Streaming {len(final_answer)} chars to frontend.")

        # Stream word-by-word so the UI renders progressively
        words = final_answer.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0)  # yield control back to event loop

        history.add_message(HumanMessage(content=req.message))
        history.add_message(AIMessage(content=final_answer))

    return StreamingResponse(response_generator(), media_type="text/plain; charset=utf-8")


@app.post("/clear")
async def clear_chat(req: ChatRequest):
    history = SQLChatMessageHistory(
        session_id=req.session_id,
        connection_string="sqlite:///memory3.db"
    )
    history.clear()
    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main3:app", host="0.0.0.0", port=8003, reload=True)
