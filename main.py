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
from sentence_transformers import CrossEncoder

DOCS_DIR     = "docs"
HASH_FILE    = "docs_hash.json"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_URL   = "http://localhost:11434"

def get_docs_fingerprint() -> str:
    """Compute an MD5 hash of every file in DOCS_DIR.
    If any file is added, removed, or modified, the hash changes."""
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
reranker    = CrossEncoder("BAAI/bge-reranker-v2-m3")

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
    bm25_retriever.k = 20
    
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    )
    
    # Using optimal weights based on evaluation (BM25: 0.3, Vector: 0.7)
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )
    print("[OK] Hybrid retriever initialized with BM25 (weight 0.3) and Vector (weight 0.7).")
else:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    print("[WARNING] No documents found. Defaulting to standard vector retriever.")


def clean_text_words(text: str) -> set:
    return set(w.strip(".,;:()[]●-●*").lower() for w in text.split() if len(w.strip(".,;:()[]●-●*")) > 1)

def filter_redundant_docs(docs: list, threshold: float = 0.3) -> list:
    unique_docs = []
    for doc in docs:
        words = clean_text_words(doc.page_content)
        is_redundant = False
        for u_doc in unique_docs:
            u_words = clean_text_words(u_doc.page_content)
            if not words or not u_words:
                continue
            intersection = words.intersection(u_words)
            overlap_coef = len(intersection) / min(len(words), len(u_words))
            if overlap_coef > threshold:
                is_redundant = True
                break
        if not is_redundant:
            unique_docs.append(doc)
    return unique_docs

def search_docs(query: str) -> str:
    """Queries the vector store for information relevant to the search query."""
    t_start = time.perf_counter()
    results = retriever.invoke(query)
    t_retrieve = time.perf_counter() - t_start
    print(f"  [TIMER] Hybrid retrieval took {t_retrieve:.3f}s")

    if not results:
        return "No relevant information found in the documents."

    seen, unique_results = set(), []
    for doc in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_results.append(doc)

    # Rerank using BAAI/bge-reranker-v2-m3
    if unique_results:
        t_start_rerank = time.perf_counter()
        pairs = [[query, doc.page_content] for doc in unique_results]
        scores = reranker.predict(pairs)
        scored_docs = list(zip(scores, unique_results))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        unique_results = [doc for score, doc in scored_docs]
        t_rerank = time.perf_counter() - t_start_rerank
        print(f"  [TIMER] Cross-encoder reranking took {t_rerank:.3f}s")

    # Associate each doc with its original rank index
    for idx, doc in enumerate(unique_results):
        doc.metadata["_original_idx"] = idx

    # Sort by length descending so more complete chunks are kept first by the overlap filter
    unique_results.sort(key=lambda d: len(d.page_content), reverse=True)

    # Filter out highly redundant overlapping chunks
    # threshold=0.85: discard a chunk if 85%+ of its words already appear in a kept chunk
    t_start_filter = time.perf_counter()
    filtered_results = filter_redundant_docs(unique_results, threshold=0.85)
    t_filter = time.perf_counter() - t_start_filter
    print(f"  [TIMER] Redundancy filtering took {t_filter:.3f}s")

    # Sort the filtered results back to their original rank order
    filtered_results.sort(key=lambda d: d.metadata.get("_original_idx", 0))

    formatted = []
    for doc in filtered_results:  # no hard cap — let threshold do the work
        doc.metadata.pop("_original_idx", None)
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page   = doc.metadata.get("page", "?")
        content = doc.page_content.replace("●", "\n- ")
        formatted.append(f"[Source: {source}, Page: {page}]\n{content}")

    return "\n\n---\n\n".join(formatted)

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0, think=False, num_ctx=8192, num_predict=512)

def classify_query(query: str, history_messages: list) -> str:
    """Classifies user query as either CONVERSATIONAL (chitchat/greetings/meta-history)
    or RAG (factual inquiries about documents)."""
    prompt = (
        "Classify the following user message.\n\n"
        "Answer 'CONVERSATIONAL' ONLY if it is simple chitchat, greetings, or asking about the chat history (e.g., 'hello', 'how are you', 'what did I ask before').\n"
        "Answer 'RAG' for ANY other message, especially questions asking for facts, concepts, definitions, or details about any topic (e.g., 'who were major contributors of GDP?', 'what is inflation?').\n\n"
        f"Message: {query}\n\n"
        "Output ONLY the word 'CONVERSATIONAL' or 'RAG':"
    )
    try:
        res = llm.invoke(prompt)
        category = res.content.strip().upper()
        if "CONVERSATIONAL" in category:
            return "CONVERSATIONAL"
        return "RAG"
    except Exception as e:
        print(f"[ROUTER] Failed classification, falling back to RAG: {e}")
        return "RAG"


def condense_query(query: str, history_messages: list) -> str:
    """Rewrites a follow-up query using chat history into a standalone search query."""
    if not history_messages:
        return query

    formatted_history = []
    for msg in history_messages:
        role = "User" if msg.type == "human" else "Assistant"
        formatted_history.append(f"{role}: {msg.content}")
    history_text = "\n".join(formatted_history)

    prompt = (
        "You are an expert query refiner. Your task is to analyze a conversation history and a follow-up question, "
        "then rewrite the follow-up question into a standalone search query that contains all necessary context "
        "to retrieve relevant documents.\n\n"
        "RULES:\n"
        "1. Resolve any pronouns (e.g. 'he', 'his', 'it', 'they', 'their', 'this', 'that', 'its') or implicit references "
        "by looking back at the conversation history to find the correct entity, name, year, or topic.\n"
        "2. Do NOT add details, terms, or topics from the history that are NOT referenced or related to the follow-up question.\n"
        "3. Keep the query focused, clean, and optimized for vector search. Do NOT answer the question; only rewrite it.\n"
        "4. If the follow-up question is already standalone and does not contain any pronouns or unresolved references, "
        "return it exactly as is.\n"
        "5. Ensure that the rephrased query is grammatically and semantically logical. Carefully align verbs with their appropriate nouns "
        "(e.g., a deficit 'widens', inflation 'increases' or 'rises', a prize is 'won', a university is 'founded'). Use this to correctly identify what the pronoun refers to.\n"
        "6. Do not guess or hallucinate specific entities (like 'the RBI' or 'inflation') to replace pronouns if they are not explicitly linked to that action or verb in the history. If a pronoun (like 'they' or 'it') cannot be resolved to a specific entity in the history, but the general topic is known (e.g., the 'fiscal deficit' target from the history), resolve only the known parts (e.g., rephrase as 'Why did they warn the fiscal deficit could widen?') rather than guessing incorrect actors.\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"Follow-up Question: {query}\n\n"
        "Standalone Search Query (return ONLY the query, no intro or explanation):"
    )
    try:
        res = llm.invoke(prompt)
        condensed = res.content.strip()
        if not condensed:
            print(f"[CONDENSATION] Empty standalone query returned. Falling back to original: '{query}'")
            return query
        print(f"[CONDENSATION] Original: '{query}' -> Standalone: '{condensed}'")
        return condensed
    except Exception as e:
        print(f"[CONDENSATION] Failed, using raw query: {e}")
        return query

def get_conversational_prompt(query: str, history_messages: list) -> str:
    formatted_history = []
    for msg in history_messages[-10:]:
        role = "User" if msg.type == "human" else "Assistant"
        formatted_history.append(f"{role}: {msg.content}")
    history_text = "\n".join(formatted_history)

    return (
        "You are a helpful assistant. Answer the user's question directly based on the conversation history below.\n"
        "If it is a casual message or 'hello', reply back to them in a normal manner"
        "If they ask 'what did I ask before' or similar, list their previous questions from the history.\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"User message: {query}\n\n"
        "Assistant Response:"
    )

app = FastAPI(title="RAG Chatbot")

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    async def response_generator():
        history = SQLChatMessageHistory(session_id=req.session_id, connection_string="sqlite:///memory.db")
        t_start = time.perf_counter()
        category = classify_query(req.message, history.messages)
        t_classify = time.perf_counter() - t_start
        print(f"[TIMER] Route selected: {category} for query: '{req.message}' (took {t_classify:.3f}s)")

        if category == "CONVERSATIONAL":
            messages = [HumanMessage(content=get_conversational_prompt(req.message, history.messages))]
        else:
            t_cond_start = time.perf_counter()
            condensed = condense_query(req.message, history.messages)
            t_condense = time.perf_counter() - t_cond_start
            print(f"[TIMER] Query Condensation took {t_condense:.3f}s (Condensed: '{condensed}')")
            
            t_ret_start = time.perf_counter()
            retrieved_context = search_docs(condensed)
            t_retrieval = time.perf_counter() - t_ret_start
            print(f"[TIMER] Total Retrieval & Processing took {t_retrieval:.3f}s")
            
            if retrieved_context == "No relevant information found in the documents.":
                answer = "The documents do not contain information about this topic."
                history.add_message(HumanMessage(content=req.message))
                history.add_message(AIMessage(content=answer))
                yield answer
                return
            else:
                system_prompt = (
                    "System Instructions:\n"
                    "1. Answer the user's query using only the facts in the retrieved chunks. Do not use outside knowledge.\n"
                    "2. If the facts to answer the query are present, state the answer directly and mention the source file and page at the end.\n"
                    "3. If the facts to answer the query are not present in the chunks, reply exactly with: 'The documents do not contain information about this topic.' Do not explain or add anything else.\n"
                    "4. Format lists as clean Markdown bullet points using '-' and include all items from the text."
                )
                human_content = (
                    f"Retrieved Text Chunks:\n{retrieved_context}\n\n"
                    f"User Query: {condensed}"
                )
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_content)
                ]

        full_response = ""
        t_gen_start = time.perf_counter()
        first_token = True
        token_count = 0
        async for chunk in llm.astream(messages):
            if first_token:
                t_first_token = time.perf_counter() - t_gen_start
                print(f"[TIMER] Time to first token: {t_first_token:.3f}s")
                first_token = False
            token = chunk.content
            full_response += token
            token_count += 1
            yield token
        t_gen_total = time.perf_counter() - t_gen_start
        tokens_per_sec = token_count / t_gen_total if t_gen_total > 0 else 0
        print(f"[TIMER] LLM generation took {t_gen_total:.3f}s ({token_count} tokens, {tokens_per_sec:.2f} tok/s)")

        history.add_message(HumanMessage(content=req.message))
        history.add_message(AIMessage(content=full_response))

    return StreamingResponse(response_generator(), media_type="text/event-stream")


@app.post("/clear")
async def clear_chat(req: ChatRequest):
    history = SQLChatMessageHistory(session_id=req.session_id, connection_string="sqlite:///memory.db")
    history.clear()
    return {"status": "ok"}

@app.get("/")
def root():
    from fastapi.responses import FileResponse
    return FileResponse("index.html")
