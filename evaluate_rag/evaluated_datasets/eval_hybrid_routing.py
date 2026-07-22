"""
run_eval_hybrid.py — Hybrid RAG + Web Search Evaluator

Tests the RAG bot's routing intelligence using a custom 100-question hybrid
dataset built from:
  - SciFact (50 questions) → expected route: RAG
  - Natural Questions (50 questions) → expected route: Web Search

The agent MUST prefer RAG first. If it cannot find the answer in RAG,
it falls back to Web Search.

Metrics:
  Routing Accuracy (%)         — Did the agent pick the right tool?
  Confusion Matrix             — TP/FP/FN/TN for routing decisions
  Web Search Trigger Rate (%)  — % of NQ queries correctly routed to web
  Answer Accuracy (EM)         — Exact-match for NQ answers
  Faithfulness                 — Generated answer grounded in context
  Answer Relevancy             — Answer directly addresses the question
  Avg Web Search Latency       — Time from query to final web-routed answer

Output: evaluate_rag/eval_report_hybrid.html
"""

import asyncio
import json
import os
import re
import html
import time
import random
from datetime import datetime

import dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.tools import tool
from sentence_transformers import CrossEncoder

# ── Environment & Config ────────────────────────────────────────────────────
_dir          = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_project_root = os.path.abspath(os.path.join(_dir, ".."))
dotenv.load_dotenv(dotenv_path=os.path.join(_project_root, ".env"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
print(f"[INIT] GOOGLE_API_KEY: {GOOGLE_API_KEY[:20]}...")
print(f"[INIT] TAVILY_API_KEY: {TAVILY_API_KEY[:20]}...")

REPORT_PATH   = os.path.join(_dir, "eval_report_hybrid.html")
CHECKPOINT_PATH = os.path.join(_dir, "eval_progress_checkpoint.json")
DATASET_PATH  = os.path.join(_dir, "hybrid_eval_dataset.json")
EVAL_SIZE_RAG = 50
EVAL_SIZE_WEB = 50
SCIFACT_CHROMA_DB   = os.path.join(_project_root, "chroma_db")
SCIFACT_COLLECTION  = "bge_m3"
JUDGE_MODEL         = "gemini-3.1-flash-lite"


# ── LLM Judge (evaluation only) ─────────────────────────────────────────────
judge_llm = ChatGoogleGenerativeAI(
    model=JUDGE_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0
)


# ── Helper: Text Extractor ───────────────────────────────────────────────────
def _extract_text(content) -> str:
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "thinking" or "thinking" in p:
                    continue
                if "text" in p:
                    parts.append(p["text"])
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


# ── Helper: JSON Parser ──────────────────────────────────────────────────────
def _parse_llm_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    candidates = [
        match.group(0),
        match.group(0).replace("'", '"'),
        re.sub(r",\s*([\]}])", r"\1", match.group(0).replace("'", '"')),
    ]
    for s in candidates:
        try:
            return json.loads(s)
        except Exception:
            continue
    return None


# ── Dataset Generation ───────────────────────────────────────────────────────
def build_hybrid_dataset() -> list[dict]:
    """
    Loads SciFact (50 q) + Natural Questions (50 q), merges, shuffles.
    Saves to DATASET_PATH. Returns the list.
    """
    import datasets as hf_datasets

    print("\n[DATASET] Building hybrid evaluation dataset...")
    hybrid = []

    # ── 1. SciFact → RAG queries ──────────────────────────────────────────
    print("[DATASET] Loading SciFact queries & qrels...")
    sf_queries = hf_datasets.load_dataset("mteb/scifact", "queries", split="queries")
    sf_qrels   = hf_datasets.load_dataset("mteb/scifact", "default", split="test")

    # Build query_id → corpus_id map from qrels
    qrels_map: dict[str, str] = {}
    for row in sf_qrels:
        qid = row.get("query-id") or row.get("_id") or ""
        cid = row.get("corpus-id") or row.get("doc-id") or ""
        if qid and cid:
            qrels_map[qid] = cid

    sf_list = [q for q in sf_queries]
    random.shuffle(sf_list)
    added = 0
    for q in sf_list:
        if added >= EVAL_SIZE_RAG:
            break
        qid = q.get("_id", q.get("id", ""))
        text = q.get("text", q.get("query", ""))
        if not text:
            continue
        hybrid.append({
            "id":             f"rag_{qid}",
            "query":          text,
            "expected_route": "rag",
            "target_doc_id":  qrels_map.get(qid, ""),
            "reference_answer": "",
        })
        added += 1
    print(f"[DATASET] SciFact: {added} queries added.")

    # ── 2. NQ Open → Web queries ────────────────────────────────────────
    print("[DATASET] Loading NQ Open...")
    nq = hf_datasets.load_dataset("nq_open", split="validation", streaming=True)
    added = 0
    for example in nq:
        if added >= EVAL_SIZE_WEB:
            break
        q_text = example.get("question", "")
        if not q_text:
            continue

        ref_answers = example.get("answer", [])
        if not ref_answers:
            continue

        nq_id = f"nq_{added}"

        hybrid.append({
            "id":             f"web_{nq_id}",
            "query":          q_text,
            "expected_route": "web",
            "target_doc_id":  "",
            "reference_answer": ref_answers,  # List of acceptable short answers
        })
        added += 1
    print(f"[DATASET] NQ Open: {added} queries added.")

    # ── Shuffle and save ──────────────────────────────────────────────────
    random.shuffle(hybrid)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(hybrid, f, indent=2, ensure_ascii=False)
    print(f"[DATASET] Saved {len(hybrid)} questions -> {DATASET_PATH}")
    return hybrid


def load_or_build_dataset() -> list[dict]:
    if os.path.exists(DATASET_PATH):
        print(f"[DATASET] Found existing dataset at {DATASET_PATH} — loading...")
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[DATASET] Loaded {len(data)} questions.")
        return data
    return build_hybrid_dataset()


# ── RAG Infrastructure ───────────────────────────────────────────────────────
def build_scifact_retriever() -> tuple:
    """
    Loads the existing SciFact Chroma index (same one used by run_eval.py),
    builds a hybrid BM25+Vector+Reranker retriever, and returns
    (retriever, chunks).
    """
    from evaluate_rag.ingestion import vectorstore as sf_vectorstore

    print("[INIT] Loading SciFact Chroma index...")
    existing = sf_vectorstore.get()
    chunks: list[Document] = []
    if existing and "documents" in existing:
        for text, meta in zip(existing["documents"], existing["metadatas"]):
            chunks.append(Document(page_content=text, metadata=meta or {}))
    print(f"[INIT] Loaded {len(chunks)} chunks from SciFact Chroma.")

    if not chunks:
        raise RuntimeError(
            "SciFact Chroma index is empty. Run `python -m evaluate_rag.index_dataset` first."
        )

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 12
    vec = sf_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 12})
    ensemble = EnsembleRetriever(retrievers=[bm25, vec], weights=[0.3, 0.7])

    print("[INIT] Loading BGE Reranker (GPU)...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    class _RerankedRetriever:
        def invoke(self, query: str) -> list[Document]:
            docs = ensemble.invoke(query)
            if not docs:
                return []
            seen, unique = set(), []
            for d in docs:
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    unique.append(d)
            pairs  = [[query, d.page_content] for d in unique]
            scores = reranker.predict(pairs)
            ranked = [d for d, _ in sorted(zip(unique, scores), key=lambda x: x[1], reverse=True)]
            return ranked[:12]

    retriever = _RerankedRetriever()
    print("[INIT] SciFact hybrid retriever ready.")
    return retriever, chunks


def _filter_redundant(docs: list[Document], threshold=0.85) -> list[Document]:
    unique: list[Document] = []
    for doc in docs:
        words = set(w.lower() for w in doc.page_content.split() if len(w) > 1)
        dup = False
        for u in unique:
            u_words = set(w.lower() for w in u.page_content.split() if len(w) > 1)
            if words and u_words and len(words & u_words) / min(len(words), len(u_words)) > threshold:
                dup = True
                break
        if not dup:
            unique.append(doc)
    return unique


# ── Web Search Helper ────────────────────────────────────────────────────────
def _tavily_search(query: str, max_results: int = 4) -> str:
    import urllib.request
    try:
        payload = json.dumps({
            "api_key":        TAVILY_API_KEY,
            "query":          query,
            "search_depth":   "basic",
            "include_answer": False,
            "max_results":    max_results,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results", [])
        if not results:
            return ""
        parts = []
        for r in results[:4]:
            parts.append(
                f"[Source: {r.get('title', '')} | {r.get('url', '')}]\n"
                f"{r.get('content', '')}"
            )
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Web search failed: {e}"


# ── Agent: Hybrid RAG-first, Web Fallback ───────────────────────────────────
async def run_hybrid_agent(
    query: str,
    retriever,
    chunks: list[Document],
    llm: ChatGoogleGenerativeAI,
) -> dict:
    """
    Runs the RAG-first, web-fallback pipeline.

    Returns a dict with:
      answer        : final answer string
      actual_route  : "rag" | "web" | "both"
      rag_context   : retrieved RAG context (or "")
      web_context   : retrieved web context (or "")
      web_latency   : seconds spent on web search (0.0 if not triggered)
    """
    result = {
        "answer":      "",
        "actual_route": "rag",
        "rag_context": "",
        "web_context": "",
        "web_latency": 0.0,
    }

    # ── Step 1: Try RAG ──────────────────────────────────────────────────
    rag_context = ""
    if chunks:
        docs = retriever.invoke(query)
        if docs:
            filtered = _filter_redundant(docs)[:5]
            rag_context = "\n\n---\n\n".join(
                f"[{d.metadata.get('source', 'doc')}]: {d.page_content}"
                for d in filtered
            )
            result["rag_context"] = rag_context

    # ── Step 2: Ask LLM if RAG is sufficient ────────────────────────────
    # We use a structured prompt: the LLM decides whether to use web search
    rag_decision_prompt = (
        f"You are a precise, fact-grounded assistant. "
        f"Your PRIMARY source is the RAG context below.\n\n"
        f"RAG Context:\n{rag_context if rag_context else '[NO RAG CONTEXT FOUND]'}\n\n"
        f"Question: {query}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. If the RAG context contains a clear, direct answer to the question, answer from it.\n"
        f"2. If the RAG context is empty, irrelevant, or does not contain the answer, "
        f"respond ONLY with the single token: <<NEEDS_WEB_SEARCH>>\n"
        f"3. Do not use outside knowledge. Do not make up facts. Only answer from the RAG context."
    )

    try:
        resp = await llm.ainvoke([HumanMessage(content=rag_decision_prompt)])
        rag_answer = _extract_text(resp.content).strip()
    except Exception as e:
        rag_answer = f"Error: {e}"

    # ── Step 3: Check if Web Fallback needed ────────────────────────────
    needs_web = (
        "<<NEEDS_WEB_SEARCH>>" in rag_answer
        or not rag_context
        or rag_answer.lower().startswith("error:")
    )

    if needs_web:
        # ── Step 4: Web Search ───────────────────────────────────────────
        result["actual_route"] = "web" if not rag_context else "both"
        t_web = time.perf_counter()
        web_context = _tavily_search(query)
        result["web_latency"] = time.perf_counter() - t_web
        result["web_context"] = web_context

        web_prompt = (
            f"You are a precise, fact-grounded assistant.\n\n"
            f"Web Search Results:\n{web_context if web_context else '[NO WEB RESULTS]'}\n\n"
            f"Question: {query}\n\n"
            f"Answer the question directly based only on the web search results above. "
            f"Be concise and factual. Do not make up information."
        )
        try:
            resp2 = await llm.ainvoke([HumanMessage(content=web_prompt)])
            result["answer"] = _extract_text(resp2.content).strip()
        except Exception as e:
            result["answer"] = f"Error: {e}"
    else:
        result["answer"] = rag_answer

    return result


# ── LLM Judge: Faithfulness + Relevancy ─────────────────────────────────────
async def judge_answer(
    query: str,
    context: str,
    generated: str,
) -> dict:
    """Judge faithfulness and answer relevancy given the retrieved context."""
    prompt = f"""You are an objective RAG evaluation judge.

QUESTION: {query}
RETRIEVED CONTEXT:
{context[:3000]}
GENERATED ANSWER:
{generated}

Score each metric 0.0 to 1.0:

FAITHFULNESS: Are ALL claims in the generated answer directly supported by the RETRIEVED CONTEXT above?
  1.0 = every claim is grounded in the context | 0.5 = partial | 0.0 = fabricated or error message

ANSWER_RELEVANCY: Does the generated answer directly address the QUESTION?
  1.0 = fully and directly addresses the question | 0.5 = partially | 0.0 = off-topic or evasive

Respond ONLY with JSON:
{{
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "reasoning": "one sentence"
}}"""
    try:
        resp = await judge_llm.ainvoke([HumanMessage(content=prompt)])
        content = _extract_text(resp.content).strip()
        parsed = _parse_llm_json(content)
        if parsed:
            return {
                "faithfulness":     float(parsed.get("faithfulness", 0.5)),
                "answer_relevancy": float(parsed.get("answer_relevancy", 0.5)),
                "reasoning":        str(parsed.get("reasoning", "")),
            }
    except Exception:
        pass
    return {"faithfulness": 0.5, "answer_relevancy": 0.5, "reasoning": "Judge parse error."}


# ── LLM Judge: Answer Correctness (web queries only) ─────────────────────────
async def judge_answer_correctness(
    query: str,
    generated: str,
    reference: str | list[str],
) -> float:
    """
    Asks the LLM judge whether the generated answer is factually correct
    compared to the reference answer. Used for NQ/web queries only.
    Returns a score 0.0–1.0.
    """
    if not reference or not generated:
        return 0.0
    
    # If reference is a list, format it clearly for the LLM
    ref_str = ", ".join(reference) if isinstance(reference, list) else reference

    prompt = f"""You are an expert fact-checking judge.

QUESTION: {query}
REFERENCE ANSWER (ground truth options): {ref_str}
GENERATED ANSWER: {generated}

Is the generated answer factually correct compared to any of the reference answer options?
The generated answer does not need to be word-for-word identical — it just needs to convey the same correct fact(s).

Respond ONLY with JSON:
{{
  "answer_correctness": 0.0,
  "reasoning": "one sentence"
}}

Where:
  1.0 = generated answer is correct and consistent with the reference
  0.5 = partially correct or vague
  0.0 = factually wrong or completely different from the reference"""
    try:
        resp = await judge_llm.ainvoke([HumanMessage(content=prompt)])
        content = _extract_text(resp.content).strip()
        parsed = _parse_llm_json(content)
        if parsed:
            return float(parsed.get("answer_correctness", 0.0))
    except Exception:
        pass
    return 0.0


# ── Exact Match normalization (SQuAD / NQ convention) ────────────────────────
def _normalize_answer(text: str) -> str:
    """Lowercase, remove articles, strip punctuation, collapse whitespace."""
    import string
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Strip punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(generated: str, reference: str | list[str]) -> bool:
    """Return True if any of the normalized reference options is contained in the normalized generated answer."""
    if not reference:
        return False
    gen_norm = _normalize_answer(generated)
    if isinstance(reference, list):
        return any(_normalize_answer(ref) in gen_norm for ref in reference if ref)
    return _normalize_answer(reference) in gen_norm


# ── HTML Report ──────────────────────────────────────────────────────────────
def save_html_report(
    results: list[dict],
    routing_stats: dict,
    web_stats: dict,
    gen_stats: dict,
):
    """Generate and save the HTML evaluation report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Confusion matrix values
    tp = routing_stats["TP"]  # RAG query → RAG route
    tn = routing_stats["TN"]  # Web query → Web route
    fp = routing_stats["FP"]  # Web query → RAG route (missed web)
    fn = routing_stats["FN"]  # RAG query → Web route (unnecessary web)
    total = tp + tn + fp + fn

    routing_acc = (tp + tn) / total * 100 if total else 0
    web_trigger_rate = routing_stats["web_trigger_rate"] * 100
    avg_web_lat = web_stats["avg_latency"]

    # Pre-compute conditional strings (Python 3.10: no backslashes in f-string expressions)
    routing_card_class = "card-green" if routing_acc >= 75 else ("card-yellow" if routing_acc >= 50 else "card-red")
    rag_precision_str = f"{(tp/(tp+fn)*100):.1f}%" if (tp+fn) else "0.0%"
    web_recall_str    = f"{(tn/(tn+fp)*100):.1f}%" if (tn+fp) else "0.0%"
    total_correct     = tp + tn

    rows_html = ""
    for r in results:
        route_color = "#22c55e" if r["routing_correct"] else "#ef4444"
        route_label = r["actual_route"].upper()
        expected_label = r["expected_route"].upper()
        corr_val = f"{r.get('ans_correctness', 0.0):.2f}" if r["expected_route"] == "web" else "—"
        rows_html += f"""
        <tr>
          <td class="idx">{r['idx']}</td>
          <td class="route-badge" style="color:{route_color}">
            <b>{route_label}</b><br><small>expected: {expected_label}</small>
          </td>
          <td class="query-cell">{html.escape(r['query'])}</td>
          <td class="answer-cell">{html.escape(r['answer'][:300])}{'...' if len(r['answer']) > 300 else ''}</td>
          <td class="ref-cell">{html.escape(r.get('reference_answer', '')[:200])}</td>
          <td class="score">{r['faithfulness']:.2f}</td>
          <td class="score">{r['answer_relevancy']:.2f}</td>
          <td class="score">{corr_val}</td>
          <td class="latency">{r['total_latency']:.2f}s</td>
          <td class="latency">{r['web_latency']:.2f}s</td>
          <td class="reasoning-cell">{html.escape(r.get('reasoning', ''))}</td>
        </tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Hybrid RAG + Web Evaluation Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0f1117; --surface: #1a1d2e; --surface2: #252840;
      --border: #2e3258; --accent: #6366f1; --accent2: #8b5cf6;
      --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
      --text: #e2e8f0; --text-muted: #94a3b8; --text-dim: #64748b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }}

    /* Header */
    .header {{
      background: linear-gradient(135deg, #1a1d2e 0%, #252840 50%, #1e1b4b 100%);
      border-bottom: 1px solid var(--border);
      padding: 2.5rem 3rem;
      position: relative; overflow: hidden;
    }}
    .header::before {{
      content: ''; position: absolute; top: -50%; right: -10%;
      width: 500px; height: 500px;
      background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
      pointer-events: none;
    }}
    .header h1 {{ font-size: 2rem; font-weight: 700; color: #fff; }}
    .header h1 span {{ background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .header .meta {{ margin-top: 0.5rem; color: var(--text-muted); font-size: 0.9rem; }}
    .header .pills {{ display: flex; gap: 0.75rem; margin-top: 1rem; flex-wrap: wrap; }}
    .pill {{
      padding: 0.3rem 0.9rem; border-radius: 999px; font-size: 0.78rem; font-weight: 500;
      border: 1px solid;
    }}
    .pill-blue {{ background: rgba(99,102,241,0.15); border-color: rgba(99,102,241,0.4); color: #818cf8; }}
    .pill-green {{ background: rgba(34,197,94,0.12); border-color: rgba(34,197,94,0.35); color: #4ade80; }}
    .pill-purple {{ background: rgba(139,92,246,0.15); border-color: rgba(139,92,246,0.4); color: #a78bfa; }}

    /* Main layout */
    .container {{ max-width: 1600px; margin: 0 auto; padding: 2rem 3rem; }}

    /* Summary cards */
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.25rem; margin-bottom: 2.5rem; }}
    .card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 14px; padding: 1.5rem;
      position: relative; overflow: hidden;
    }}
    .card::before {{
      content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
    }}
    .card.card-green::before {{ background: linear-gradient(90deg, #22c55e, #16a34a); }}
    .card.card-red::before   {{ background: linear-gradient(90deg, #ef4444, #dc2626); }}
    .card.card-yellow::before{{ background: linear-gradient(90deg, #f59e0b, #d97706); }}
    .card-label {{ font-size: 0.78rem; font-weight: 500; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    .card-value {{ font-size: 2.2rem; font-weight: 700; color: #fff; margin-top: 0.4rem; line-height: 1; }}
    .card-sub   {{ font-size: 0.8rem; color: var(--text-dim); margin-top: 0.5rem; }}

    /* Section title */
    .section-title {{
      font-size: 1.1rem; font-weight: 600; color: var(--text);
      margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.6rem;
    }}
    .section-title::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}

    /* Confusion matrix */
    .cm-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); border-radius: 12px; overflow: hidden; max-width: 480px; }}
    .cm-cell {{
      background: var(--surface2); padding: 1.5rem; text-align: center;
    }}
    .cm-cell.tp, .cm-cell.tn {{ background: rgba(34,197,94,0.08); }}
    .cm-cell.fp, .cm-cell.fn {{ background: rgba(239,68,68,0.08); }}
    .cm-label {{ font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; }}
    .cm-name  {{ font-size: 0.85rem; color: var(--text-muted); margin: 0.3rem 0; }}
    .cm-val   {{ font-size: 2.5rem; font-weight: 700; color: #fff; }}
    .cm-val.green {{ color: var(--green); }}
    .cm-val.red   {{ color: var(--red); }}

    /* Confusion matrix wrapper */
    .cm-wrapper {{ display: flex; gap: 3rem; align-items: flex-start; flex-wrap: wrap; margin-bottom: 2.5rem; }}
    .cm-legend {{ flex: 1; min-width: 260px; }}
    .cm-legend h4 {{ color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.8rem; }}
    .cm-legend ul {{ list-style: none; display: flex; flex-direction: column; gap: 0.45rem; }}
    .cm-legend li {{ font-size: 0.83rem; color: var(--text-dim); }}
    .cm-legend li strong {{ color: var(--text-muted); }}

    /* Bar charts */
    .bar-group {{ display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 2rem; }}
    .bar-row {{ display: flex; align-items: center; gap: 1rem; }}
    .bar-label {{ width: 160px; font-size: 0.82rem; color: var(--text-muted); flex-shrink: 0; text-align: right; }}
    .bar-track {{ flex: 1; height: 10px; background: var(--surface2); border-radius: 999px; overflow: hidden; }}
    .bar-fill  {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }}
    .bar-fill.green {{ background: linear-gradient(90deg, #22c55e, #16a34a); }}
    .bar-val   {{ width: 52px; font-size: 0.82rem; color: var(--text); font-weight: 600; }}

    /* Table */
    .table-wrap {{ overflow-x: auto; border-radius: 14px; border: 1px solid var(--border); margin-top: 1rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    thead th {{
      background: var(--surface); color: var(--text-muted);
      font-size: 0.73rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.07em; padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--border);
      position: sticky; top: 0;
    }}
    tbody tr {{ border-bottom: 1px solid rgba(46,50,88,0.5); transition: background 0.15s; }}
    tbody tr:hover {{ background: rgba(99,102,241,0.04); }}
    tbody tr:last-child {{ border-bottom: none; }}
    td {{ padding: 0.75rem 1rem; vertical-align: top; }}
    td.idx {{ color: var(--text-dim); font-size: 0.75rem; }}
    td.route-badge {{ min-width: 110px; font-size: 0.78rem; }}
    td.query-cell {{ min-width: 280px; max-width: 380px; color: var(--text); }}
    td.answer-cell {{ min-width: 240px; max-width: 320px; color: var(--text-muted); font-size: 0.8rem; }}
    td.ref-cell {{ min-width: 140px; max-width: 200px; color: var(--text-dim); font-size: 0.8rem; }}
    td.em-cell {{ text-align: center; font-size: 1rem; }}
    td.score {{ text-align: center; font-weight: 600; }}
    td.latency {{ text-align: right; color: var(--text-muted); font-size: 0.8rem; }}
    td.reasoning-cell {{ min-width: 200px; max-width: 280px; color: var(--text-dim); font-size: 0.78rem; font-style: italic; }}

    /* Score colors */
    td.score {{ }}
    .score-high {{ color: #4ade80; }}
    .score-mid  {{ color: #fbbf24; }}
    .score-low  {{ color: #f87171; }}

    /* Search */
    .search-bar {{
      width: 100%; padding: 0.65rem 1rem; border-radius: 8px;
      background: var(--surface); border: 1px solid var(--border);
      color: var(--text); font-size: 0.88rem; outline: none;
      margin-bottom: 1rem; transition: border-color 0.2s;
    }}
    .search-bar:focus {{ border-color: var(--accent); }}

    /* Filter tabs */
    .filter-tabs {{ display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }}
    .ftab {{
      padding: 0.35rem 0.9rem; border-radius: 6px; font-size: 0.8rem;
      cursor: pointer; border: 1px solid var(--border); background: var(--surface);
      color: var(--text-muted); transition: all 0.15s;
    }}
    .ftab:hover, .ftab.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}

    footer {{ text-align: center; padding: 2rem; color: var(--text-dim); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 3rem; }}
  </style>
</head>
<body>

<div class="header">
  <h1>🔀 <span>Hybrid RAG + Web</span> Evaluation Report</h1>
  <div class="meta">Generated on {ts} | SciFact (RAG) + Natural Questions (Web) | {total} queries</div>
  <div class="pills">
    <span class="pill pill-blue">Generator: gemini-3.1-flash-lite</span>
    <span class="pill pill-purple">Judge: gemini-3.1-flash-lite</span>
    <span class="pill pill-green">Retriever: BGE-M3 + Cross-Encoder</span>
  </div>
</div>

<div class="container">

  <!-- ── Summary Cards ── -->
  <div class="cards">
    <div class="card {routing_card_class}">
      <div class="card-label">Routing Accuracy</div>
      <div class="card-value">{routing_acc:.1f}<span style="font-size:1.2rem">%</span></div>
      <div class="card-sub">{total_correct}/{total} correctly routed</div>
    </div>
    <div class="card">
      <div class="card-label">Faithfulness</div>
      <div class="card-value">{gen_stats['avg_faith']:.2f}</div>
      <div class="card-sub">avg over all 100 queries</div>
    </div>
    <div class="card">
      <div class="card-label">Answer Relevancy</div>
      <div class="card-value">{gen_stats['avg_rel']:.2f}</div>
      <div class="card-sub">avg over all 100 queries</div>
    </div>
    <div class="card card-yellow">
      <div class="card-label">Avg Web Latency</div>
      <div class="card-value">{avg_web_lat:.1f}<span style="font-size:1.2rem">s</span></div>
      <div class="card-sub">search + generation time</div>
    </div>
  </div>

  <!-- ── Confusion Matrix ── -->
  <div class="section-title">📊 Routing Confusion Matrix</div>
  <div class="cm-wrapper">
    <div>
      <div style="font-size:0.75rem;color:var(--text-dim);margin-bottom:0.6rem;text-align:center;">Predicted Route →</div>
      <div style="display:grid;grid-template-columns:auto 1fr 1fr;gap:1px;background:var(--border);border-radius:12px;overflow:hidden;">
        <div style="background:var(--surface);padding:0.75rem;display:flex;align-items:center;justify-content:center;font-size:0.7rem;color:var(--text-dim);">Actual ↓</div>
        <div style="background:var(--surface);padding:0.75rem;text-align:center;font-size:0.75rem;color:var(--text-muted);font-weight:600;">RAG</div>
        <div style="background:var(--surface);padding:0.75rem;text-align:center;font-size:0.75rem;color:var(--text-muted);font-weight:600;">WEB</div>
        <div style="background:var(--surface);padding:0.75rem;display:flex;align-items:center;justify-content:center;font-size:0.75rem;color:var(--text-muted);font-weight:600;writing-mode:horizontal-tb;">RAG</div>
        <div class="cm-cell tp">
          <div class="cm-label">TP</div>
          <div class="cm-name">Correct RAG</div>
          <div class="cm-val green">{tp}</div>
        </div>
        <div class="cm-cell fn">
          <div class="cm-label">FN</div>
          <div class="cm-name">Unnecessary Web</div>
          <div class="cm-val red">{fn}</div>
        </div>
        <div style="background:var(--surface);padding:0.75rem;display:flex;align-items:center;justify-content:center;font-size:0.75rem;color:var(--text-muted);font-weight:600;">WEB</div>
        <div class="cm-cell fp">
          <div class="cm-label">FP</div>
          <div class="cm-name">Missed Web</div>
          <div class="cm-val red">{fp}</div>
        </div>
        <div class="cm-cell tn">
          <div class="cm-label">TN</div>
          <div class="cm-name">Correct Web</div>
          <div class="cm-val green">{tn}</div>
        </div>
      </div>
    </div>
    <div class="cm-legend">
      <h4>Legend</h4>
      <ul>
        <li><strong>TP (True Positive):</strong> SciFact query → correctly used RAG</li>
        <li><strong>TN (True Negative):</strong> NQ query → correctly used Web Search</li>
        <li><strong>FP (False Positive):</strong> NQ query → stuck in RAG, missed web fallback</li>
        <li><strong>FN (False Negative):</strong> SciFact query → unnecessarily went to web</li>
      </ul>
      <div style="margin-top:1.2rem;padding:1rem;background:var(--surface);border-radius:8px;border:1px solid var(--border);">
        <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:0.5rem;font-weight:600;">Key Rates</div>
        <div style="font-size:0.83rem;color:var(--text-dim);line-height:1.9;">
          Routing Accuracy: <b style="color:var(--text)">{routing_acc:.1f}%</b><br>
          RAG Precision: <b style="color:var(--text)">{rag_precision_str}</b> (of RAG routes, how many were correct)<br>
          Web Recall: <b style="color:var(--text)">{web_recall_str}</b> (of NQ queries, how many hit web)
        </div>
      </div>
    </div>
  </div>

  <!-- ── Metric Bars ── -->
  <div class="section-title">📈 Generation Quality</div>
  <div class="bar-group">
    <div class="bar-row">
      <div class="bar-label">Faithfulness (RAG)</div>
      <div class="bar-track"><div class="bar-fill green" style="width:{gen_stats['avg_faith_rag']*100:.1f}%"></div></div>
      <div class="bar-val">{gen_stats['avg_faith_rag']:.2f}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">Faithfulness (Web)</div>
      <div class="bar-track"><div class="bar-fill" style="width:{gen_stats['avg_faith_web']*100:.1f}%"></div></div>
      <div class="bar-val">{gen_stats['avg_faith_web']:.2f}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">Relevancy (RAG)</div>
      <div class="bar-track"><div class="bar-fill green" style="width:{gen_stats['avg_rel_rag']*100:.1f}%"></div></div>
      <div class="bar-val">{gen_stats['avg_rel_rag']:.2f}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">Relevancy (Web)</div>
      <div class="bar-track"><div class="bar-fill" style="width:{gen_stats['avg_rel_web']*100:.1f}%"></div></div>
      <div class="bar-val">{gen_stats['avg_rel_web']:.2f}</div>
    </div>
  </div>

  <!-- ── Per-Query Table ── -->
  <div class="section-title">🔍 Per-Query Results</div>

  <input id="searchInput" class="search-bar" placeholder="🔍 Search by query or answer..." oninput="filterTable()">
  <div class="filter-tabs">
    <div class="ftab active" onclick="setFilter('all', this)">All ({total})</div>
    <div class="ftab" onclick="setFilter('rag', this)">RAG Only</div>
    <div class="ftab" onclick="setFilter('web', this)">Web Only</div>
    <div class="ftab" onclick="setFilter('wrong', this)">Wrong Route ❌</div>
    <div class="ftab" onclick="setFilter('correct', this)">Correct Route ✅</div>
  </div>

  <div class="table-wrap">
    <table id="resultsTable">
      <thead>
        <tr>
          <th>#</th>
          <th>Route</th>
          <th>Query</th>
          <th>Generated Answer</th>
          <th>Reference</th>
          <th>Faith</th>
          <th>Rel</th>
          <th>Correctness</th>
          <th>Total</th>
          <th>Web Lat.</th>
          <th>Reasoning</th>
        </tr>
      </thead>
      <tbody id="tableBody">
        {rows_html}
      </tbody>
    </table>
  </div>

</div>

<footer>
  Hybrid RAG Evaluation Report · Generated {ts} · SciFact + Natural Questions · {total} queries
</footer>

<script>
  let currentFilter = 'all';

  function setFilter(f, el) {{
    currentFilter = f;
    document.querySelectorAll('.ftab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
    filterTable();
  }}

  function filterTable() {{
    const search = document.getElementById('searchInput').value.toLowerCase();
    const rows = document.querySelectorAll('#tableBody tr');
    rows.forEach(row => {{
      const text = row.textContent.toLowerCase();
      const route = row.cells[1].textContent.toLowerCase();
      const isCorrect = !route.includes('expected') || row.cells[1].querySelector('b').textContent === row.cells[1].querySelector('small').textContent.replace('expected: ', '');
      
      let show = text.includes(search);
      if (currentFilter === 'rag') show = show && route.includes('rag');
      if (currentFilter === 'web') show = show && (route.includes('web') || route.includes('both'));
      if (currentFilter === 'wrong') show = show && row.cells[1].style.color === 'rgb(239, 68, 68)';
      if (currentFilter === 'correct') show = show && row.cells[1].style.color === 'rgb(34, 197, 94)';
      row.style.display = show ? '' : 'none';
    }});
  }}

  // Color score cells dynamically
  document.querySelectorAll('td.score').forEach(cell => {{
    const v = parseFloat(cell.textContent);
    if (v >= 0.8) cell.classList.add('score-high');
    else if (v >= 0.5) cell.classList.add('score-mid');
    else cell.classList.add('score-low');
  }});
</script>
</body>
</html>"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n[REPORT] Saved -> {REPORT_PATH}")


# ── Main Evaluation Loop ─────────────────────────────────────────────────────
# ── Main Evaluation Loop ─────────────────────────────────────────────────────
async def main():
    # 1. Build/load dataset
    dataset = load_or_build_dataset()

    # 2. Build SciFact retriever
    retriever, chunks = build_scifact_retriever()

    # 3. Main LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    total = len(dataset)
    
    # ── Try loading checkpoint state ──
    start_idx = 1
    results = []
    routing = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    web_latencies: list[float] = []
    faithfulness_rag: list[float] = []
    faithfulness_web: list[float] = []
    relevancy_rag:    list[float] = []
    relevancy_web:    list[float] = []
    em_scores:              list[float] = []

    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            
            # Verify dataset signature hasn't changed (optional safeguard)
            if checkpoint.get("dataset_len") == total:
                results = checkpoint.get("results", [])
                routing = checkpoint.get("routing", routing)
                web_latencies = checkpoint.get("web_latencies", [])
                faithfulness_rag = checkpoint.get("faithfulness_rag", [])
                faithfulness_web = checkpoint.get("faithfulness_web", [])
                relevancy_rag = checkpoint.get("relevancy_rag", [])
                relevancy_web = checkpoint.get("relevancy_web", [])
                em_scores = checkpoint.get("em_scores", [])
                start_idx = len(results) + 1
                print(f"[CHECKPOINT] Resuming from query [{start_idx}/{total}] (loaded {len(results)} completed results).")
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}. Starting fresh.")
            results = []

    if start_idx == 1:
        print(f"\nStarting Hybrid Evaluation ({total} queries)...\n" + "=" * 80)
    else:
        print(f"\nResuming Hybrid Evaluation from [{start_idx}/{total}]...\n" + "=" * 80)

    for idx in range(start_idx, total + 1):
        item = dataset[idx - 1]
        query           = item["query"]
        expected_route  = item["expected_route"]
        reference_ans   = item.get("reference_answer", "")

        print(f"\n[{idx}/{total}] [{expected_route.upper()}] {query[:90]!r}")

        t_total_start = time.perf_counter()

        # ── Run hybrid agent ──
        try:
            agent_out = await run_hybrid_agent(query, retriever, chunks, llm)
        except Exception as err:
            print(f"  [ERROR] run_hybrid_agent failed for query '{query}': {err}")
            # Use fallback empty values to allow continuation
            agent_out = {
                "answer": f"Agent error: {err}",
                "actual_route": "rag",
                "rag_context": "",
                "web_context": "",
                "web_latency": 0.0,
            }

        total_latency = time.perf_counter() - t_total_start

        actual_route   = agent_out["actual_route"]
        generated_ans  = agent_out["answer"]
        rag_ctx        = agent_out["rag_context"]
        web_ctx        = agent_out["web_context"]
        web_lat        = agent_out["web_latency"]

        if web_lat > 0:
            web_latencies.append(web_lat)

        # ── Routing correctness ──
        rag_used = actual_route == "rag"
        web_used = actual_route in ("web", "both")

        if expected_route == "rag":
            if rag_used:
                routing["TP"] += 1
                routing_correct = True
            else:
                routing["FN"] += 1
                routing_correct = False
        else:  # expected_route == "web"
            if web_used:
                routing["TN"] += 1
                routing_correct = True
            else:
                routing["FP"] += 1
                routing_correct = False

        print(f"  Route: expected={expected_route} | actual={actual_route} | {'OK' if routing_correct else 'WRONG'}")

        # ── Judge: pick context ──
        if actual_route == "rag":
            judge_context = rag_ctx or "[no RAG context]"
        else:
            judge_context = web_ctx or "[no web context]"

        # ── LLM Judge quality scores ──
        faith = 0.5
        rel = 0.5
        reasoning = "Judge call skipped due to agent error"
        if not generated_ans.startswith("Agent error:"):
            try:
                judge_scores = await judge_answer(query, judge_context, generated_ans)
                faith = judge_scores["faithfulness"]
                rel   = judge_scores["answer_relevancy"]
                reasoning = judge_scores.get("reasoning", "")
            except Exception as err:
                print(f"  [ERROR] judge_answer call failed: {err}")
                reasoning = f"Judge error: {err}"

        if expected_route == "rag":
            faithfulness_rag.append(faith)
            relevancy_rag.append(rel)
        else:
            faithfulness_web.append(faith)
            relevancy_web.append(rel)

        # ── Exact match ──
        em_score      = 0.0
        if expected_route == "web" and reference_ans and not generated_ans.startswith("Agent error:"):
            try:
                em_score    = 1.0 if exact_match(generated_ans, reference_ans) else 0.0
                em_scores.append(em_score)
            except Exception as err:
                print(f"  [ERROR] exact_match check failed: {err}")

        if expected_route == "web":
            if reference_ans:
                em_str = f"EM={em_score:.0f} | "
            else:
                em_str = "EM=N/A | "
        else:
            em_str = ""
        print(
            f"  Faith={faith:.2f} | Rel={rel:.2f} | "
            + em_str
            + f"WebLat={web_lat:.2f}s | Total={total_latency:.2f}s"
        )

        results.append({
            "idx":              idx,
            "query":            query,
            "expected_route":   expected_route,
            "actual_route":     actual_route,
            "routing_correct":  routing_correct,
            "answer":           generated_ans,
            "reference_answer": ", ".join(reference_ans) if isinstance(reference_ans, list) else reference_ans,
            "faithfulness":     faith,
            "answer_relevancy": rel,
            "em_score":         em_score,
            "total_latency":    total_latency,
            "web_latency":      web_lat,
            "reasoning":        reasoning,
        })

        # ── Save current progress state checkpoint ──
        try:
            checkpoint_state = {
                "dataset_len": total,
                "results": results,
                "routing": routing,
                "web_latencies": web_latencies,
                "faithfulness_rag": faithfulness_rag,
                "faithfulness_web": faithfulness_web,
                "relevancy_rag": relevancy_rag,
                "relevancy_web": relevancy_web,
                "em_scores": em_scores,
            }
            with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                json.dump(checkpoint_state, f, indent=2, ensure_ascii=False)
        except Exception as checkpoint_err:
            print(f"  [WARNING] Failed to write checkpoint: {checkpoint_err}")

        # Rate-limit pause
        if idx < total:
            await asyncio.sleep(3)

    # ── Final stats ────────────────────────────────────────────────────────
    total_q = len(results)
    routing_acc = (routing["TP"] + routing["TN"]) / total_q * 100 if total_q else 0
    web_nq_queries = routing["TN"] + routing["FP"]
    web_trigger_rate = routing["TN"] / web_nq_queries if web_nq_queries else 0

    print("\n" + "=" * 80)
    print("FINAL HYBRID EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Routing Accuracy     : {routing_acc:.1f}%  (TP={routing['TP']} TN={routing['TN']} FP={routing['FP']} FN={routing['FN']})")
    print(f"  Web Trigger Rate     : {web_trigger_rate*100:.1f}%  ({routing['TN']}/{web_nq_queries} NQ queries reached web)")
    print(f"  Exact Match (NQ)     : {(sum(em_scores)/len(em_scores)*100) if em_scores else 0:.1f}%")
    print(f"  Faithfulness (RAG)   : {(sum(faithfulness_rag)/len(faithfulness_rag)) if faithfulness_rag else 0:.3f}")
    print(f"  Faithfulness (Web)   : {(sum(faithfulness_web)/len(faithfulness_web)) if faithfulness_web else 0:.3f}")
    print(f"  Relevancy (RAG)      : {(sum(relevancy_rag)/len(relevancy_rag)) if relevancy_rag else 0:.3f}")
    print(f"  Relevancy (Web)      : {(sum(relevancy_web)/len(relevancy_web)) if relevancy_web else 0:.3f}")
    print(f"  Avg Web Latency      : {(sum(web_latencies)/len(web_latencies)) if web_latencies else 0:.2f}s")
    print("=" * 80)

    routing_stats = {
        **routing,
        "web_trigger_rate": web_trigger_rate,
    }
    web_stats = {
        "avg_latency": (sum(web_latencies) / len(web_latencies)) if web_latencies else 0.0,
    }
    all_faith = faithfulness_rag + faithfulness_web
    all_rel   = relevancy_rag + relevancy_web
    gen_stats = {
        "avg_faith":          (sum(all_faith) / len(all_faith)) if all_faith else 0.0,
        "avg_rel":            (sum(all_rel) / len(all_rel)) if all_rel else 0.0,
        "avg_faith_rag":      (sum(faithfulness_rag) / len(faithfulness_rag)) if faithfulness_rag else 0.0,
        "avg_faith_web":      (sum(faithfulness_web) / len(faithfulness_web)) if faithfulness_web else 0.0,
        "avg_rel_rag":        (sum(relevancy_rag) / len(relevancy_rag)) if relevancy_rag else 0.0,
        "avg_rel_web":        (sum(relevancy_web) / len(relevancy_web)) if relevancy_web else 0.0,
        "em_rate":            (sum(em_scores) / len(em_scores)) if em_scores else 0.0,
    }

    save_html_report(results, routing_stats, web_stats, gen_stats)
    print(f"\nOpen report: file:///{REPORT_PATH.replace(os.sep, '/')}")

    # Remove checkpoint file upon successful full completion
    if os.path.exists(CHECKPOINT_PATH):
        try:
            os.remove(CHECKPOINT_PATH)
            print("[CLEANUP] Deleted evaluation checkpoint file.")
        except Exception as rm_err:
            print(f"[WARNING] Failed to clean up checkpoint file: {rm_err}")


if __name__ == "__main__":
    asyncio.run(main())
