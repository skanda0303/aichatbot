import os
import time
import hashlib
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# ── No reranker imported ────────────────────────────────────────────────────

DOCS_DIR     = "docs"
HASH_FILE    = "docs_hash.json"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_URL   = "http://localhost:11434"

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

embeddings  = OllamaEmbeddings(model="bge-m3")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings, collection_name="bge_m3")

chunks = []

if os.listdir(DOCS_DIR):
    current_fp = get_docs_fingerprint()
    stored_fp  = load_stored_fingerprint()
    existing_ids = vectorstore.get()["ids"]

    if current_fp == stored_fp and existing_ids:
        print(f"[OK] Documents unchanged — reusing existing index ({len(existing_ids)} chunks). Skipping ingestion.")
        existing_data = vectorstore.get()
        if existing_data and "documents" in existing_data:
            for doc_text, metadata in zip(existing_data["documents"], existing_data["metadatas"]):
                chunks.append(Document(page_content=doc_text, metadata=metadata))
    else:
        print("[INFO] Document changes detected. Re-indexing...")
        raw_docs = PyPDFDirectoryLoader(DOCS_DIR).load()

        from collections import defaultdict
        grouped_content = defaultdict(list)
        for d in raw_docs:
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", 0)
            grouped_content[src].append((page, d.page_content))

        merged_docs = []
        for src, page_tuples in sorted(grouped_content.items()):
            page_tuples.sort(key=lambda x: x[0])
            full_text = " ".join(" ".join(content for _, content in page_tuples).split())
            merged_docs.append(Document(page_content=full_text, metadata={"source": src, "page": "0"}))

        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(merged_docs)

        if existing_ids:
            vectorstore.delete(ids=existing_ids)

        vectorstore.add_documents(chunks)
        save_fingerprint(current_fp)
        print(f"[OK] Ingested {len(chunks)} chunks from {len(merged_docs)} file(s).")
else:
    print("[INFO] docs/ is empty — no documents ingested.")

if chunks:
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 16

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 16}
    )

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )
    print("[OK] (No-Reranker) Hybrid retriever initialized — BM25 0.3 / Vector 0.7")
else:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print("[WARNING] No documents found. Defaulting to standard vector retriever.")


def clean_text_words(text: str) -> set:
    return set(w.strip(".,;:()[]●-●*").lower() for w in text.split() if len(w.strip(".,;:()[]●-●*")) > 1)

def filter_redundant_docs(docs: list, threshold: float = 0.85) -> list:
    unique_docs = []
    for doc in docs:
        words = clean_text_words(doc.page_content)
        is_redundant = False
        for u_doc in unique_docs:
            u_words = clean_text_words(u_doc.page_content)
            if not words or not u_words:
                continue
            overlap_coef = len(words.intersection(u_words)) / min(len(words), len(u_words))
            if overlap_coef > threshold:
                is_redundant = True
                break
        if not is_redundant:
            unique_docs.append(doc)
    return unique_docs

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0, think=False, num_ctx=8192, num_predict=1024)

def analyze_and_build_prompt(query: str, confidence: str) -> tuple[int, str]:
    meta_prompt = (
        "You are an expert RAG pipeline orchestrator. Analyze the following user query and produce a JSON object.\n\n"
        f"User Query: {query}\n"
        f"Retrieval Confidence: {confidence}  (HIGH means top document score is strong, LOW means weak relevance)\n\n"
        "Produce a valid JSON object with exactly these keys:\n"
        '  \"k\": integer between 5 and 20 — how many document chunks to retrieve. '
        "Use a higher number for complex, multi-part, or broad questions; lower for simple factual lookups.\n"
        '  \"system_prompt\": string — a complete, detailed system prompt for an LLM that will answer this query '
        "using only retrieved document chunks. The prompt must:\n"
        "    - Instruct the LLM to answer solely from provided chunks (no outside knowledge).\n"
        "    - Specify the ideal response format strictly based on query type: \n"
        "        1. For comparisons or side-by-side analysis, explicitly instruct the LLM to output a clean, neat Markdown table with distinct headers.\n"
        "        2. For descriptive profiles or multi-fact summaries (like 'who is X' or 'what are features of Y'), explicitly instruct the LLM to output a clean, structured bullet-point list.\n"
        "        3. For simple lookups or single factual questions, instruct the LLM to reply with a concise prose paragraph.\n"
        "    - If the user asked for a specific word count or detail level, include an instruction to meet it by "
        "elaborating on and explaining the retrieved facts without adding outside information.\n"
        "    - If confidence is LOW, add a caution to only state what is explicitly in the text.\n"
        "    - End with: if no relevant facts are found in the chunks to answer the query, the LLM must reply ONLY and exactly with the fallback message: 'The documents do not contain information about this topic.' It must NOT attempt to answer using outside knowledge or parametric memory under any circumstances.\n"
        "Output ONLY the raw JSON object, no markdown fences, no extra text."
    )
    try:
        res = llm.invoke(meta_prompt)
        raw = res.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        k = max(5, min(20, int(data.get("k", 8))))
        system_prompt = str(data.get("system_prompt", "")).strip()
        if not system_prompt:
            raise ValueError("Empty system_prompt")
        print(f"[DYNAMIC] k={k}")
        return k, system_prompt
    except Exception as e:
        print(f"[DYNAMIC] LLM analysis failed ({e}), using fallback.")
        return 8, (
            "Answer the user's query using only the facts in the retrieved chunks. "
            "Do not use outside knowledge. "
            "If the answer is not present in the chunks, reply exactly with: "
            "'The documents do not contain information about this topic.'"
        )

def search_docs(query: str, k: int = 8) -> tuple[str, list]:
    """Returns (formatted_context, filtered_docs). No reranking — uses ensemble order."""
    if chunks:
        bm25_retriever.k = k
        vector_retriever.search_kwargs["k"] = k

    t_start = time.perf_counter()
    results = retriever.invoke(query)
    print(f"  [TIMER] Hybrid retrieval took {time.perf_counter() - t_start:.3f}s")

    if not results:
        return "No relevant information found in the documents.", []

    seen, unique_results = set(), []
    for doc in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_results.append(doc)

    # No reranking — keep ensemble order
    for idx, doc in enumerate(unique_results):
        doc.metadata["_original_idx"] = idx

    unique_results.sort(key=lambda d: len(d.page_content), reverse=True)

    t_f = time.perf_counter()
    filtered = filter_redundant_docs(unique_results, threshold=0.85)
    print(f"  [TIMER] Redundancy filtering took {time.perf_counter() - t_f:.3f}s")

    filtered.sort(key=lambda d: d.metadata.get("_original_idx", 0))

    def fmt(docs):
        out = []
        for doc in docs:
            doc.metadata.pop("_original_idx", None)
            content = doc.page_content.replace("●", "\n- ")
            out.append(content)
        return "\n\n---\n\n".join(out)

    return fmt(filtered), filtered

def self_rag_rewrite(query: str) -> str:
    prompt = (
        f"You are a search query optimizer. The search for '{query}' returned no relevant documents.\n"
        "Rewrite this query into a different, broader, or alternative search query using synonyms. "
        "Do NOT answer the question. Output ONLY the new query:"
    )
    try:
        res = llm.invoke(prompt)
        new_query = res.content.strip()
        print(f"[Self-RAG] Rewrote query: '{query}' -> '{new_query}'")
        return new_query
    except Exception as e:
        print(f"[Self-RAG] Failed: {e}")
        return query

def classify_query(query: str, history_messages: list) -> str:
    prompt = (
        "Classify the following user message.\n\n"
        "Answer 'CONVERSATIONAL' ONLY if it is simple chitchat, greetings, or asking about the chat history.\n"
        "Answer 'RAG' for ANY other message.\n\n"
        f"Message: {query}\n\n"
        "Output ONLY the word 'CONVERSATIONAL' or 'RAG':"
    )
    try:
        res = llm.invoke(prompt)
        return "CONVERSATIONAL" if "CONVERSATIONAL" in res.content.strip().upper() else "RAG"
    except Exception as e:
        print(f"[ROUTER] Failed: {e}")
        return "RAG"

def condense_query(query: str, history_messages: list) -> str:
    if not history_messages:
        return query
    history_text = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in history_messages
    )
    prompt = (
        "You are an expert query refiner. Rewrite the follow-up question into a standalone search query "
        "using the conversation history to resolve pronouns and implicit references.\n\n"
        "RULES:\n"
        "1. Resolve pronouns by looking back at the history.\n"
        "2. Do NOT add unrelated topics from history.\n"
        "3. Keep it focused and optimized for vector search. Do NOT answer.\n"
        "4. If already standalone, return it as-is.\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"Follow-up Question: {query}\n\n"
        "Standalone Search Query:"
    )
    try:
        res = llm.invoke(prompt)
        condensed = res.content.strip() or query
        print(f"[CONDENSATION] '{query}' -> '{condensed}'")
        return condensed
    except Exception as e:
        print(f"[CONDENSATION] Failed: {e}")
        return query

def get_conversational_prompt(query: str, history_messages: list) -> str:
    history_text = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in history_messages[-10:]
    )
    return (
        "You are a helpful assistant. Answer the user's question directly based on the conversation history below.\n"
        "If it is a casual message or 'hello', reply back to them in a normal manner. "
        "If they ask 'what did I ask before' or similar, list their previous questions.\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"User message: {query}\n\nAssistant Response:"
    )

app = FastAPI(title="RAG Chatbot — No Reranker (Port 8001)")

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    async def response_generator():
        history = SQLChatMessageHistory(session_id=req.session_id, connection_string="sqlite:///memory2.db")

        t_start = time.perf_counter()
        category = classify_query(req.message, history.messages)
        print(f"[TIMER] Route: {category} (took {time.perf_counter() - t_start:.3f}s)")

        if category == "CONVERSATIONAL":
            messages = [HumanMessage(content=get_conversational_prompt(req.message, history.messages))]
        else:
            t_cond = time.perf_counter()
            condensed = condense_query(req.message, history.messages)
            print(f"[TIMER] Condensation: {time.perf_counter() - t_cond:.3f}s")

            t_ret = time.perf_counter()
            retrieved_context, ranked_docs = search_docs(condensed, k=16)

            # Self-RAG: no reranker score available, so check if context is empty
            if not retrieved_context.strip() or retrieved_context == "No relevant information found in the documents.":
                print("[Self-RAG] No results. Rewriting query...")
                condensed = self_rag_rewrite(condensed)
                retrieved_context, ranked_docs = search_docs(condensed, k=16)

            print(f"[TIMER] Total Retrieval took {time.perf_counter() - t_ret:.3f}s")

            if not retrieved_context.strip() or retrieved_context == "No relevant information found in the documents.":
                answer = "The documents do not contain information about this topic."
                history.add_message(HumanMessage(content=req.message))
                history.add_message(AIMessage(content=answer))
                yield answer
                return
            else:
                t_dyn = time.perf_counter()
                k_final, system_prompt = analyze_and_build_prompt(condensed, "HIGH")
                print(f"[TIMER] Dynamic analysis: {time.perf_counter() - t_dyn:.3f}s (k_final={k_final})")

                if k_final < len(ranked_docs):
                    def fmt(docs):
                        out = []
                        for doc in docs:
                            content = doc.page_content.replace("●", "\n- ")
                            out.append(content)
                        return "\n\n---\n\n".join(out)
                    retrieved_context = fmt(ranked_docs[:k_final])

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Retrieved Text Chunks:\n{retrieved_context}\n\nUser Query: {condensed}")
                ]

        full_response = ""
        t_gen = time.perf_counter()
        first_token = True
        token_count = 0
        async for chunk in llm.astream(messages):
            if first_token:
                print(f"[TIMER] Time to first token: {time.perf_counter() - t_gen:.3f}s")
                first_token = False
            token = chunk.content
            full_response += token
            token_count += 1
            yield token
        t_total = time.perf_counter() - t_gen
        print(f"[TIMER] LLM generation: {t_total:.3f}s ({token_count} tokens, {token_count/t_total:.2f} tok/s)")

        history.add_message(HumanMessage(content=req.message))
        history.add_message(AIMessage(content=full_response))

    return StreamingResponse(response_generator(), media_type="text/event-stream")


@app.post("/clear")
async def clear_chat(req: ChatRequest):
    history = SQLChatMessageHistory(session_id=req.session_id, connection_string="sqlite:///memory2.db")
    history.clear()
    return {"status": "ok"}

@app.get("/")
def root():
    from fastapi.responses import FileResponse
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="0.0.0.0", port=8001, reload=True)
