import os
import time
import hashlib
import json
import asyncio
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.tools import tool

# ── Configuration ──────────────────────────────────────────────────────────────
DOCS_DIR             = "docs"
HASH_FILE            = "docs_hash.json"
OLLAMA_MODEL         = "qwen2.5:3b"
OLLAMA_URL           = "http://localhost:11434"
MAX_AGENT_ITERATIONS = 5    # max tool-call rounds before forcing a final answer
MAX_WEB_RESULTS      = 5   # DuckDuckGo results per search
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

        if existing_ids:
            vectorstore.delete(ids=existing_ids)
        vectorstore.add_documents(chunks)
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
# num_ctx=6144 is enough for conversation + tool results without excessive overhead
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.2,
    think=True,
    num_ctx=6144,
    num_predict=512,
    format="json",
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


@tool
def web_search(query: str) -> str:
    """Search the web for real-time information, latest news, current events, or facts
    not found in the documents. Use as a fallback when rag_search returns NO_RELEVANT_DOCS,
    or directly for queries about current events / live data."""
    try:
        from ddgs import DDGS
        snippets = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=MAX_WEB_RESULTS):
                snippets.append(f"[{r.get('title', '')}]\n{r.get('body', '')}\nSource: {r.get('href', '')}")
        if not snippets:
            return "No web results found for this query."
        print(f"  [WEB TOOL] Returned {len(snippets)} results.")
        return "\n\n---\n\n".join(snippets)
    except Exception as e:
        return f"Web search failed: {e}"


@tool
def get_datetime() -> str:
    """Returns the current date, time, and day of the week. Use for any question about today's date,
    current time, what day it is, etc."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')}"


# ── Agent setup ────────────────────────────────────────────────────────────────
TOOLS     = [rag_search, web_search, get_datetime]
TOOL_MAP  = {t.name: t for t in TOOLS}

AGENT_SYSTEM_PROMPT = (
    "You are a highly precise, fact-grounded agentic assistant. You make routing decisions and answer "
    "questions using the provided tools. You MUST respond ONLY with a valid JSON object matching the schema below. "
    "Do not include any thinking tags like <think> or prose output outside the JSON. "
    "Your response must be a single, valid JSON object at each step.\n\n"
    "SCHEMA:\n"
    "{\n"
    "  \"thought\": \"Detailed reasoning about what tool to call next and what search query is needed.\",\n"
    "  \"tool\": \"rag_search\" | \"web_search\" | \"get_datetime\" | \"none\",\n"
    "  \"tool_input\": \"Search query or parameter, or empty string.\",\n"
    "  \"final_answer\": \"The final output to the user (only fill this when tool is 'none').\"\n"
    "}\n\n"
    "SEARCH QUERY FORMULATION RULES:\n"
    "- Do NOT pass the raw user query directly to web_search if it is conversational or complex. "
    "Instead, rewrite it into highly targeted, keyword-optimized search queries (e.g. rewrite 'who won the latest NBA Finals and who was named Finals MVP?' into '2026 NBA Finals winner MVP').\n"
    "- For financial data or stock prices, always search for 'stock price today June 2026'.\n"
    "- Include the year 2026 in the search query for any time-sensitive queries to get the latest results.\n\n"
    "CRITICAL GROUNDING & SYNTHESIS RULES (follow strictly):\n"
    "1. SYSTEM DATE: Today is Friday, June 26, 2026. Keep this temporal context in mind.\n"
    "2. NO OUTDATED DATA: Check the dates in search snippets. Do NOT present information from 2024 or 2025 as 'today's' or 'latest' data if 2026 data is available or expected. For example, if a stock price article is from 2025, do not call it 'today's price'. If no 2026 data is found, state the date of the data clearly.\n"
    "3. COHERENT SYNTHESIS & VERIFICATION: Integrate search results logically. Do not just list or dump search results sequentially. Summarize them into a cohesive response. Verify names and details (e.g., do not mix up regular-season MVP with Finals MVP).\n"
    "4. ANSWER DIRECTLY: Never tell the user to 'check a website' (e.g. ESPNcricinfo) to get the answer. You MUST extract the actual scores, numbers, or facts from the tool output and present them directly.\n"
    "5. TECHNICAL PRECISION: Avoid hallucinating technical facts or mixing brand architectures (e.g. do not associate NVIDIA with AMD's RDNA or vice versa; RTX is NVIDIA, RDNA/Radeon is AMD).\n"
    "6. DECISION RULES:\n"
    "  - For any question about Rabindranath Tagore, uploaded document topics, files, or info stored in the documents, call 'rag_search' first.\n"
    "  - If 'rag_search' returns 'NO_RELEVANT_DOCS' or lacks the answer, immediately call 'web_search' as a fallback.\n"
    "  - For general knowledge, external queries, or facts, call 'web_search'.\n"
    "  - For today's date, current time, or current day of the week, call 'get_datetime'. DO NOT use 'get_datetime' for historical birthdays.\n"
)

# ── Tool dispatcher ────────────────────────────────────────────────────────────
async def dispatch_tool(name: str, args: dict) -> str:
    """Execute a tool call asynchronously (runs sync tools in a thread pool)."""
    if name not in TOOL_MAP:
        return f"Unknown tool requested: '{name}'"
    try:
        result = await asyncio.to_thread(TOOL_MAP[name].invoke, args)
        return str(result)
    except Exception as e:
        return f"Tool '{name}' failed: {e}"

# ── Agent loop ─────────────────────────────────────────────────────────────────
async def run_agent(query: str, history_messages: list) -> str:
    """
    JSON-based ReAct-style agent loop:
      1. LLM decides which tool to call by returning structured JSON
      2. Tool executes and returns result
      3. LLM reads result and either calls another tool or produces final_answer
    Max MAX_AGENT_ITERATIONS rounds before forcing a final answer.
    """
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    messages += list(history_messages[-MAX_HISTORY_MESSAGES:])
    messages.append(HumanMessage(content=query))

    for iteration in range(MAX_AGENT_ITERATIONS):
        t = time.perf_counter()
        resp = await llm.ainvoke(messages)
        print(f"[AGENT] LLM call #{iteration + 1}: {time.perf_counter() - t:.3f}s")

        try:
            decision = json.loads(resp.content)
        except Exception as e:
            print(f"[AGENT] JSON parse error: {e}. Raw content: {resp.content}")
            return "I encountered an error formatting the response. Please try again."

        thought = decision.get("thought", "")
        tool_name = decision.get("tool", "none")
        tool_input = decision.get("tool_input", "")
        final_answer = decision.get("final_answer", "")

        print(f"[AGENT] Thought: {thought}")
        print(f"[AGENT] Action: {tool_name} (input: '{tool_input}')")

        if tool_name == "none" or not tool_name:
            if not final_answer:
                final_answer = "I could not find relevant information about this topic."
            return final_answer

        # Execute the chosen tool
        t_tool = time.perf_counter()
        args = {}
        if tool_name in ["rag_search", "web_search"]:
            args = {"query": tool_input}

        result = await dispatch_tool(tool_name, args)
        elapsed = time.perf_counter() - t_tool
        print(f"[AGENT] Tool '{tool_name}' -> {len(result)} chars in {elapsed:.3f}s")

        # Append intermediate thought & output
        messages.append(AIMessage(content=json.dumps(decision)))
        messages.append(HumanMessage(content=f"Tool '{tool_name}' returned: {result}"))

    # Force a final answer if max iterations reached
    print("[AGENT] Max iterations reached — forcing final answer.")
    messages.append(SystemMessage(content="You have reached the maximum number of iterations. Please output your final answer now. Set tool to 'none' and put the response in 'final_answer'."))
    resp = await llm.ainvoke(messages)
    try:
        decision = json.loads(resp.content)
        return decision.get("final_answer", "I could not find relevant information about this topic.")
    except Exception:
        return "I could not find relevant information about this topic."


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Agentic RAG Chatbot — Port 8002")

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
        final_answer = await run_agent(req.message, history.messages)
        print(f"[TIMER] Total agent time: {time.perf_counter() - t_start:.3f}s")

        # Stream word-by-word so the UI renders progressively
        words = final_answer.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

        history.add_message(HumanMessage(content=req.message))
        history.add_message(AIMessage(content=final_answer))

    return StreamingResponse(response_generator(), media_type="text/event-stream")


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
    uvicorn.run("main3:app", host="0.0.0.0", port=8002, reload=True)
