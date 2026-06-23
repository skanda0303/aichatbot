import os
import hashlib
import json
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

DOCS_DIR     = "docs"
HASH_FILE    = "docs_hash.json"
OLLAMA_MODEL = "qwen3:8b"
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

embeddings  = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

if os.listdir(DOCS_DIR):
    current_fp = get_docs_fingerprint()
    stored_fp  = load_stored_fingerprint()
    existing_ids = vectorstore.get()["ids"]

    if current_fp == stored_fp and existing_ids:
        print(f"[OK] Documents unchanged — reusing existing index ({len(existing_ids)} chunks). Skipping ingestion.")
    else:
        print("[INFO] Document changes detected. Re-indexing...")
        raw_docs = PyPDFDirectoryLoader(DOCS_DIR).load()
        
        from collections import defaultdict
        from langchain_core.documents import Document
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

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

def search_docs(query: str) -> str:
    """Queries the vector store for information relevant to the search query."""
    results = retriever.invoke(query)

    if not results:
        return "No relevant information found in the documents."

    seen, unique_results = set(), []
    for doc in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_results.append(doc)

    formatted = []
    for doc in unique_results:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page   = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0, think=False, num_ctx=8192)

def classify_query(query: str, history_messages: list) -> str:
    """Classifies user query as either CONVERSATIONAL (chitchat/greetings/meta-history)
    or RAG (factual inquiries about documents)."""
    history_context = ""
    if history_messages:
        recent = history_messages[-4:]
        formatted = []
        for msg in recent:
            role = "User" if msg.type == "human" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        history_context = "Recent Conversation History:\n" + "\n".join(formatted) + "\n\n"
        
    prompt = (
        "You are a router that classifies a user's latest message as either 'CONVERSATIONAL' or 'RAG'.\n\n"
        "DEFINITIONS:\n"
        "1. 'CONVERSATIONAL': Greetings, general chitchat, or meta-questions *about* the conversation itself "
        "(e.g., 'hello', 'who are you?', 'what was the first question I asked you?', 'what did we just talk about?').\n"
        "2. 'RAG': Questions asking for facts, reports, data, summaries, or analyses about topics/documents "
        "Note: Factual follow-up questions that refer to previous topics"
        "are RAG queries because they require searching document facts.\n\n"
        f"{history_context}"
        f"Latest User Message to Classify: {query}\n\n"
        "Return ONLY the word 'CONVERSATIONAL' or 'RAG', with no formatting, markdown, or explanation.\n\n"
        "Classification:"
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
        print(f"[CONDENSATION] Original: '{query}' -> Standalone: '{condensed}'")
        return condensed
    except Exception as e:
        print(f"[CONDENSATION] Failed, using raw query: {e}")
        return query

def generate_rag_response(condensed_query: str, retrieved_context: str) -> str:
    """Generates factual response using ONLY the current retrieved context,
    preventing any memory leakage or history-bias bugs."""
    prompt = (
        "You are a factual document-answering assistant. You are provided with a search query "
        "and relevant text chunks retrieved from documents.\n\n"
        "RULES:\n"
        "1. Your answer must be built EXCLUSIVELY from the provided retrieved text chunks.\n"
        "2. Do NOT add, infer, or recall anything from your own pre-trained knowledge.\n"
        "3. Do not be overly cautious: if the retrieved text discusses the topic, answer the query, "
        "even if the spelling or phrasing in the query differs slightly from the document (e.g. 'focussed' vs 'focused').\n"
        "4. When the retrieved chunks contain list items, priorities, key areas, or points, "
        "you MUST present them as a clean Markdown bulleted list (using '-' or '*' on new lines), "
        "even if the source text has them inline or separated by symbols.\n"
        "5. If the retrieved chunks do not contain the answer, respond EXACTLY with: "
        "'The documents do not contain information about this topic.'\n"
        "6. Always state which document and page your answer comes from.\n\n"
        f"Retrieved Text Chunks:\n{retrieved_context}\n\n"
        f"Search Query: {condensed_query}\n\n"
        "Answer:"
    )
    res = llm.invoke(prompt)
    return res.content.strip()

def generate_conversational_response(query: str, history_messages: list) -> str:
    """Answers conversational meta-questions directly from the chat history database."""
    formatted_history = []
    for msg in history_messages[-10:]:
        role = "User" if msg.type == "human" else "Assistant"
        formatted_history.append(f"{role}: {msg.content}")
    history_text = "\n".join(formatted_history)

    prompt = (
        "You are a helpful assistant. Answer the user's question directly based on the conversation history below.\n"
        "If they ask 'what did I ask before' or similar, list their previous questions from the history.\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"User message: {query}\n\n"
        "Assistant Response:"
    )
    res = llm.invoke(prompt)
    return res.content.strip()

app = FastAPI(title="RAG Chatbot")

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    history = SQLChatMessageHistory(session_id=req.session_id, connection_string="sqlite:///memory.db")
    category = classify_query(req.message, history.messages)
    print(f"[ROUTER] Route selected: {category} for query: '{req.message}'")

    if category == "CONVERSATIONAL":
        answer = generate_conversational_response(req.message, history.messages)
    else:
        condensed = condense_query(req.message, history.messages)
        retrieved_context = search_docs(condensed)
        if retrieved_context == "No relevant information found in the documents.":
            answer = "The documents do not contain information about this topic."
        else:
            answer = generate_rag_response(condensed, retrieved_context)

    history.add_message(HumanMessage(content=req.message))
    history.add_message(AIMessage(content=answer))

    return {"session_id": req.session_id, "answer": answer}

@app.post("/clear")
async def clear_chat(req: ChatRequest):
    history = SQLChatMessageHistory(session_id=req.session_id, connection_string="sqlite:///memory.db")
    history.clear()
    return {"status": "ok"}

@app.get("/")
def root():
    from fastapi.responses import FileResponse
    return FileResponse("index.html")
