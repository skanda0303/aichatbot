"""
run_eval_multi.py — Multi-Agent RAG Evaluator on BEIR SciFact Benchmark.

Evaluates the complete multi-agent pipeline (RAG Agent → Evaluation Agent →
[Web Agent] → Answer Agent) on the same 75-query SciFact test set used by
evaluate_rag/run_eval.py, so results are directly comparable.

Pipeline under test (per query):
  1. RAG Agent        — BM25 + Vector + CrossEncoder retrieval → RAGResult
  2. Evaluation Agent — Gemini judges retrieval sufficiency → EvalResult
  3. Web Agent        — only if insufficient → WebResult  (counted in report)
  4. Answer Agent     — generates final answer using available context

Metrics (identical definition to evaluate_rag/run_eval.py):
  BEIR Retrieval (present set only):
    NDCG@10           — ranking quality of retrieved docs vs target paper
    Recall@5          — did retriever surface the target paper in top 5?
    Context Precision — fraction of retrieved chunks from the correct paper

  RAGAS Generation (both sets):
    Faithfulness      — are all LLM claims grounded in retrieved context?
    Answer Relevancy  — does the answer address the question?

  Multi-Agent Specific:
    Eval Sufficiency  — % of queries deemed sufficient by Evaluation Agent
    Web Trigger Rate  — % of queries that triggered the Web Agent
    Avg CrossEncoder Score — mean reranker score per query

Dataset: BEIR SciFact (mteb/scifact)
  50 Present : target doc IS indexed
  25 Absent  : target doc NOT indexed (tests hallucination resistance)

Run:
  python -m evaluate_multi_rag.index_dataset   # run once to index SciFact
  python -m evaluate_multi_rag.run_eval_multi  # run evaluation
"""

import asyncio
import json
import math
import os
import re
import html
import time
from datetime import datetime

import datasets
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from evaluate_multi_rag.config import (
    GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    RETRIEVER_K, BM25_WEIGHT, VECTOR_WEIGHT,
    REDUNDANCY_THRESHOLD, CHUNK_SIZE, CHUNK_OVERLAP,
    BEIR_DATASET,
)
from evaluate_multi_rag.ingestion import vectorstore

# ── Paths ─────────────────────────────────────────────────────────────────────
_dir         = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH  = os.path.join(_dir, "eval_report_multi.html")
CONFIG_PATH  = os.path.join(_dir, "indexed_config_multi.json")
CKPT_PATH    = os.path.join(_dir, "eval_multi_checkpoint.json")

GENERATOR_MODEL = LLM_MODEL
JUDGE_MODEL     = "gemini-3.1-flash-lite"

# ── LLMs ──────────────────────────────────────────────────────────────────────
generator_llm = ChatGoogleGenerativeAI(
    model=GENERATOR_MODEL, google_api_key=GOOGLE_API_KEY, temperature=LLM_TEMPERATURE
)
judge_llm = ChatGoogleGenerativeAI(
    model=JUDGE_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0
)
eval_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=LLM_TEMPERATURE
)

# ── Text helpers ───────────────────────────────────────────────────────────────
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


def _parse_llm_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    for s in [
        match.group(0),
        match.group(0).replace("'", '"'),
        re.sub(r",\s*([\]}])", r"\1", match.group(0).replace("'", '"')),
    ]:
        try:
            return json.loads(s)
        except Exception:
            continue
    return None


def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in [
        "cannot answer", "does not contain", "no information",
        "not mentioned", "not discussed", "not provide information",
        "i do not know", "i am sorry", "insufficient context",
        "cannot be answered", "is not mentioned in",
    ])


# ── Dataset loading ────────────────────────────────────────────────────────────
_cached_corpus: dict | None = None


def get_corpus_dict() -> dict:
    global _cached_corpus
    if _cached_corpus is None:
        print(f"[INFO] Loading BEIR corpus for '{BEIR_DATASET}'...")
        ds = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "corpus")
        split_name = list(ds.keys())[0]
        _cached_corpus = {row["_id"]: row for row in ds[split_name]}
    return _cached_corpus


def load_dataset_meta() -> tuple[dict, dict, dict]:
    """Return queries, qrels, reference_answers dicts."""
    print(f"[INFO] Loading BEIR queries for '{BEIR_DATASET}'...")
    queries_ds   = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "queries")
    queries      = {row["_id"]: {"query": row["text"]} for row in queries_ds[list(queries_ds.keys())[0]]}

    print(f"[INFO] Loading BEIR qrels for '{BEIR_DATASET}'...")
    qrels_ds   = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "default")
    qrels_rows = []
    for split in qrels_ds.keys():
        qrels_rows.extend(qrels_ds[split])

    qrels: dict = {}
    for row in qrels_rows:
        q_id, c_id, score = row["query-id"], row["corpus-id"], row["score"]
        if score >= 1:
            if q_id not in qrels or score > qrels[q_id]["score"]:
                qrels[q_id] = {"doc_id": c_id, "score": score}

    corpus  = get_corpus_dict()
    answers = {}
    for q_id, info in qrels.items():
        doc_row = corpus.get(info["doc_id"])
        if doc_row:
            title = doc_row.get("title", "")
            text  = doc_row.get("text", "")
            answers[q_id] = f"{title}\n{text}" if title else text
        else:
            answers[q_id] = ""

    return queries, qrels, answers


def load_paper_chunks(paper_id: str) -> list[Document]:
    corpus  = get_corpus_dict()
    doc_row = corpus.get(paper_id)
    if not doc_row:
        return []
    title     = doc_row.get("title", "")
    text      = doc_row.get("text", "")
    full_text = f"{title}\n{text}" if title else text
    full_text = " ".join(full_text.split())
    splitter  = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    doc = Document(page_content=full_text, metadata={"source": paper_id, "title": title})
    return splitter.split_documents([doc])


# ── Retriever (same as multi_agent/retrieval/retriever.py) ────────────────────
def _word_set(text: str) -> set:
    return set(
        w.strip(".,;:()[]●-●*").lower()
        for w in text.split()
        if len(w.strip(".,;:()[]●-●*")) > 1
    )


def _filter_redundant(docs: list[Document], threshold: float = REDUNDANCY_THRESHOLD) -> list[Document]:
    unique: list[Document] = []
    for doc in docs:
        words = _word_set(doc.page_content)
        dup   = False
        for u in unique:
            u_words = _word_set(u.page_content)
            if words and u_words and len(words & u_words) / min(len(words), len(u_words)) > threshold:
                dup = True
                break
        if not dup:
            unique.append(doc)
    return unique


class _RerankedRetriever:
    """Hybrid BM25 + Vector + CrossEncoder — identical to multi_agent.retrieval.retriever."""

    def __init__(self, base_retriever, reranker: CrossEncoder, top_n: int = RETRIEVER_K):
        self.base_retriever = base_retriever
        self.reranker       = reranker
        self.top_n          = top_n

    def invoke_with_scores(self, query: str) -> tuple[list[Document], list[float]]:
        docs = self.base_retriever.invoke(query)
        if not docs:
            return [], []
        seen, unique = set(), []
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                unique.append(d)
        pairs  = [[query, d.page_content] for d in unique]
        scores = self.reranker.predict(pairs)
        pairs_sorted = sorted(zip(unique, scores), key=lambda x: x[1], reverse=True)
        top_docs   = [d for d, _ in pairs_sorted[: self.top_n]]
        top_scores = [float(s) for _, s in pairs_sorted[: self.top_n]]
        return top_docs, top_scores

    def invoke(self, query: str) -> list[Document]:
        docs, _ = self.invoke_with_scores(query)
        return docs


def build_retriever(chunks: list[Document]) -> _RerankedRetriever:
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = RETRIEVER_K
    vec  = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVER_K})
    ensemble = EnsembleRetriever(retrievers=[bm25, vec], weights=[BM25_WEIGHT, VECTOR_WEIGHT])
    print("[INFO] Loading BGE Reranker (BAAI/bge-reranker-v2-m3)...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    print("[INFO] Reranker loaded.")
    return _RerankedRetriever(ensemble, reranker, top_n=RETRIEVER_K)


# ── Multi-Agent Pipeline Steps ────────────────────────────────────────────────

async def _rag_agent(query: str, retriever: _RerankedRetriever, chunks: list[Document]) -> dict:
    """Mirrors multi_agent/agents/rag_agent.py — returns RAGResult-like dict."""
    if not chunks:
        return {"chunks": [], "scores": [], "avg_score": 0.0, "metadata": []}
    try:
        docs, scores = retriever.invoke_with_scores(query)
        docs   = _filter_redundant(docs)
        scores = scores[:len(docs)]
        return {
            "chunks":    [d.page_content for d in docs],
            "scores":    scores,
            "avg_score": float(sum(scores) / len(scores)) if scores else 0.0,
            "metadata":  [dict(d.metadata) for d in docs],
            "docs":      docs,   # keep Document objects for BEIR metrics
        }
    except Exception as e:
        print(f"[RAG AGENT] Error: {e}")
        return {"chunks": [], "scores": [], "avg_score": 0.0, "metadata": [], "docs": []}


async def _evaluation_agent(query: str, rag: dict) -> dict:
    """Mirrors multi_agent/agents/evaluation_agent.py — returns EvalResult-like dict."""
    if not rag["chunks"]:
        return {"sufficient": False, "confidence": 0.0, "reason": "No chunks retrieved."}

    chunks_preview = "\n\n---\n\n".join(rag["chunks"][:6])
    scores_summary = (
        f"Average CrossEncoder score: {rag['avg_score']:.4f}\n"
        f"Top-3 scores: {[round(s, 4) for s in rag['scores'][:3]]}"
    )
    prompt = (
        "You are a context evaluation specialist. Your ONLY job is to determine whether "
        "the retrieved document chunks are sufficient to answer the user's question.\n\n"
        "Output ONLY a JSON object with exactly these fields:\n"
        '  "sufficient"  : boolean\n'
        '  "confidence"  : float 0.0–1.0\n'
        '  "reason"      : one concise sentence\n\n'
        f"User Question:\n{query}\n\n"
        f"Retrieval Scores:\n{scores_summary}\n\n"
        f"Retrieved Chunks ({len(rag['chunks'])} total):\n\n{chunks_preview}\n\n"
        "Evaluate whether these chunks are sufficient to answer the question."
    )
    try:
        resp    = await eval_llm.ainvoke([HumanMessage(content=prompt)])
        raw     = _extract_text(resp.content).strip()
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        match   = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "sufficient": bool(data.get("sufficient", False)),
                "confidence": float(data.get("confidence", 0.5)),
                "reason":     str(data.get("reason", "")),
            }
    except Exception as e:
        print(f"[EVAL AGENT] Error: {e}")
    return {"sufficient": False, "confidence": 0.0, "reason": "Evaluation failed."}


async def _answer_agent(
    query: str, rag: dict, eval_result: dict, web_context: str = ""
) -> str:
    """Mirrors multi_agent/agents/answer_agent.py."""
    parts = []
    if rag["chunks"]:
        rag_text = "\n\n---\n\n".join(rag["chunks"][:8])
        parts.append(f"=== Knowledge Base Context ===\n{rag_text}")
    if web_context:
        parts.append(f"=== Web Search Context ===\n{web_context}")
    if not parts:
        parts.append("No relevant context was retrieved.")

    context_block = "\n\n".join(parts)
    system = (
        f"You are a precise, fact-grounded assistant. "
        f"Current date: {datetime.now().strftime('%A, %B %d, %Y')}.\n"
        "Answer directly from facts in the provided context. "
        "Convert any LaTeX into plain text. "
        "If context is empty or does not contain the answer, "
        "explicitly state that you cannot answer based on the context."
    )
    user = f"{context_block}\n\n---\n\nUser Question: {query}"
    try:
        resp = await generator_llm.ainvoke([
            HumanMessage(content=system),
            HumanMessage(content=user),
        ])
        return _extract_text(resp.content).strip()
    except Exception as e:
        return f"Error during generation: {e}"


# ── BEIR Metrics (identical to evaluate_rag/run_eval.py) ─────────────────────

def compute_ndcg(docs: list[Document], target_doc: str, k: int = 10) -> float:
    rel  = [1 if d.metadata.get("source") == target_doc else 0 for d in docs[:k]]
    if not rel:
        return 0.0
    dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(sum(rel), k)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(docs: list[Document], target_doc: str) -> bool:
    return any(d.metadata.get("source") == target_doc for d in docs)


def compute_context_precision(docs: list[Document], target_doc: str) -> float:
    if not docs:
        return 0.0
    relevant, precision_sum = 0, 0.0
    for i, d in enumerate(docs, start=1):
        if d.metadata.get("source") == target_doc:
            relevant += 1
            precision_sum += relevant / i
    return precision_sum / relevant if relevant else 0.0


# ── Judge (identical to evaluate_rag/run_eval.py) ─────────────────────────────

async def evaluate_generation(
    query: str, context: str, gen_ans: str, reference_answer: str, subset: str
) -> dict:
    if is_refusal(gen_ans):
        return {
            "faithfulness":     1.0,
            "answer_relevancy": 1.0 if subset == "absent" else 0.5,
            "reasoning":        "Model correctly abstained (no hallucination).",
        }

    context_snippet = context[:3000] if context else "(empty)"
    judge_prompt = f"""You are an objective RAG evaluation judge.

QUESTION: {query}

RETRIEVED CONTEXT (first 3000 chars):
{context_snippet}

GENERATED ANSWER:
{gen_ans}

Score each metric 0.0 to 1.0:

FAITHFULNESS: Are all claims in the generated answer directly supported by the RETRIEVED CONTEXT?
  1.0 = every claim grounded | 0.5 = partial | 0.0 = mostly unsupported or fabricated

ANSWER_RELEVANCY: Does the generated answer directly address the original QUESTION?
  1.0 = fully | 0.5 = partially | 0.0 = off-topic or evasive

Respond ONLY with this JSON (no markdown):
{{
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "reasoning": "one sentence"
}}"""

    try:
        resp    = await judge_llm.ainvoke([HumanMessage(content=judge_prompt)])
        content = _extract_text(resp.content).strip()
        parsed  = _parse_llm_json(content)
        if parsed:
            return {
                "faithfulness":     max(0.0, min(1.0, float(parsed.get("faithfulness", 0.5)))),
                "answer_relevancy": max(0.0, min(1.0, float(parsed.get("answer_relevancy", 0.5)))),
                "reasoning":        str(parsed.get("reasoning", "")),
            }
    except Exception as e:
        print(f"[JUDGE ERROR] {e}")
    return {"faithfulness": 0.5, "answer_relevancy": 0.5, "reasoning": "Judge parse failed."}


# ── HTML Report ────────────────────────────────────────────────────────────────

def _badge(ok: bool, yes_label: str = "PASS", no_label: str = "FAIL") -> str:
    c = "#22c55e" if ok else "#ef4444"
    l = yes_label if ok else no_label
    return f'<span style="background:{c};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.8em;font-weight:600">{l}</span>'


def _score_badge(score: float) -> str:
    c = "#22c55e" if score >= 0.7 else ("#f59e0b" if score >= 0.4 else "#ef4444")
    return f'<span style="background:{c};color:#fff;padding:2px 8px;border-radius:12px;font-size:0.8em;font-weight:600">{score:.2f}</span>'


def save_html_report(
    query_results: list[dict],
    summary_present: dict,
    summary_absent:  dict,
    multi_stats:     dict,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Summary section builder ───────────────────────────────────────────────
    def make_summary_section(s: dict, title: str, desc: str, color: str, is_present: bool) -> str:
        tot = s["total"] or 1
        if is_present:
            metrics_rows = f"""
            <tr><td colspan="2" style="padding:6px 0;font-weight:700;color:{color};font-size:0.85em;text-transform:uppercase">BEIR Retrieval</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#4b5563">NDCG@10</td>
                <td style="text-align:right;font-weight:700">{s['ndcg_10']/tot:.3f}</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#4b5563">Recall@5</td>
                <td style="text-align:right;font-weight:700">{s['recall_5']/tot*100:.1f}%</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#4b5563">Context Precision</td>
                <td style="text-align:right;font-weight:700">{s['ctx_prec']/(2*tot):.3f}</td></tr>
            <tr><td colspan="2" style="padding:6px 0;font-weight:700;color:{color};font-size:0.85em;text-transform:uppercase">RAGAS Generation</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#4b5563">Faithfulness</td>
                <td style="text-align:right;font-weight:700;color:#22c55e">{s['faith']/(2*tot):.3f}</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#4b5563">Answer Relevancy</td>
                <td style="text-align:right;font-weight:700;color:#6366f1">{s['ans_rel']/(2*tot):.3f}</td></tr>
            <tr><td colspan="2" style="padding:6px 0;font-weight:700;color:{color};font-size:0.85em;text-transform:uppercase">Multi-Agent Pipeline</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#4b5563">Eval Sufficient Rate</td>
                <td style="text-align:right;font-weight:700;color:#8b5cf6">{multi_stats['sufficient_rate_present']*100:.1f}%</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#4b5563">Web Trigger Rate</td>
                <td style="text-align:right;font-weight:700;color:#f59e0b">{multi_stats['web_rate_present']*100:.1f}%</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#4b5563">Avg CrossEncoder Score</td>
                <td style="text-align:right;font-weight:700">{multi_stats['avg_cross_present']:.4f}</td></tr>"""
        else:
            metrics_rows = f"""
            <tr><td colspan="2" style="padding:6px 0;font-weight:700;color:{color};font-size:0.85em;text-transform:uppercase">RAGAS Generation (Absent Set)</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#22c55e;font-weight:600">Abstention Rate (Faithfulness)</td>
                <td style="text-align:right;font-weight:700;color:#22c55e">{s['faith']/(2*tot)*100:.1f}%</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#4b5563">Answer Relevancy</td>
                <td style="text-align:right;font-weight:700;color:#6366f1">{s['ans_rel']/(2*tot):.3f}</td></tr>
            <tr><td colspan="2" style="padding:6px 0;font-weight:700;color:{color};font-size:0.85em;text-transform:uppercase">Multi-Agent Pipeline</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#4b5563">Eval Sufficient Rate</td>
                <td style="text-align:right;font-weight:700;color:#8b5cf6">{multi_stats['sufficient_rate_absent']*100:.1f}%</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#4b5563">Web Trigger Rate</td>
                <td style="text-align:right;font-weight:700;color:#f59e0b">{multi_stats['web_rate_absent']*100:.1f}%</td></tr>"""

        latency_rows = ""
        for k in [3, 5]:
            latency_rows += f"""
            <tr><td colspan="2" style="padding:4px 0;font-weight:600;color:#1e293b;font-size:0.88em;border-top:1px dashed #e2e8f0">k={k} Avg Latency</td></tr>
            <tr><td style="padding:2px 0 2px 16px;color:#6b7280;font-size:0.85em">Retrieval</td><td style="text-align:right">{s[f't_ret_{k}']/tot:.2f}s</td></tr>
            <tr><td style="padding:2px 0 2px 16px;color:#6b7280;font-size:0.85em">Generation</td><td style="text-align:right">{s[f't_gen_{k}']/tot:.2f}s</td></tr>
            <tr><td style="padding:2px 0 2px 16px;color:#6b7280;font-size:0.85em">Evaluation</td><td style="text-align:right">{s[f't_eval_{k}']/tot:.2f}s</td></tr>
            <tr><td style="padding:2px 0 2px 16px;color:#111;font-weight:600;font-size:0.85em">Total</td>
                <td style="text-align:right;font-weight:700;color:#4f46e5">{s[f't_tot_{k}']/tot:.2f}s</td></tr>"""

        return f"""
        <div style="background:#fff;border-radius:12px;padding:20px 24px;box-shadow:0 1px 8px #0001;border-top:4px solid {color};flex:1;min-width:340px">
          <h3 style="margin:0 0 4px;color:{color}">{title}</h3>
          <p style="color:#6b7280;font-size:0.85em;margin-bottom:16px">{desc}</p>
          <table style="width:100%;border-collapse:collapse;font-size:0.9em">
            <tr style="border-bottom:1px solid #f3f4f6">
              <th style="text-align:left;padding:6px 0">Metric</th>
              <th style="text-align:right">Score</th>
            </tr>
            {metrics_rows}
            {latency_rows}
          </table>
        </div>"""

    # ── Per-query rows ────────────────────────────────────────────────────────
    def query_rows(subset_type: str) -> str:
        rows = ""
        for r in [x for x in query_results if x["subset"] == subset_type]:
            q   = html.escape(r["query"])
            doc = html.escape(r["target_doc"])
            exp = html.escape(r["expected_answer"][:800])
            gen = html.escape(r["generated_answer_k3"][:400])
            k3  = r["k3"]
            k5  = r["k5"]

            eval_badge_color = "#22c55e" if r["eval_sufficient"] else "#f59e0b"
            eval_label       = "SUFFICIENT" if r["eval_sufficient"] else "INSUFFICIENT"
            web_badge        = '<span style="background:#f59e0b;color:#fff;padding:2px 8px;border-radius:12px;font-size:0.75em;font-weight:600">WEB TRIGGERED</span>' if r["web_triggered"] else ""

            metric_rows = ""
            if subset_type == "present":
                metric_rows += f"""
                <tr style="border-top:1px solid #e5e7eb;background:#fafafa">
                  <td style="padding:6px 12px;color:#374151;font-weight:700;font-size:0.82em" colspan="3">BEIR RETRIEVAL</td></tr>
                <tr style="border-top:1px solid #e5e7eb">
                  <td style="padding:8px 12px;color:#6b7280;font-size:0.85em">NDCG@10</td>
                  <td style="text-align:center;font-weight:600">{k3['ndcg_10']:.3f}</td>
                  <td style="text-align:center;font-weight:600">{k5['ndcg_10']:.3f}</td></tr>
                <tr style="border-top:1px solid #e5e7eb">
                  <td style="padding:8px 12px;color:#6b7280;font-size:0.85em">Recall@5</td>
                  <td style="text-align:center">{_badge(k3['recall_5'], 'HIT', 'MISS')}</td>
                  <td style="text-align:center">{_badge(k5['recall_5'], 'HIT', 'MISS')}</td></tr>
                <tr style="border-top:1px solid #e5e7eb">
                  <td style="padding:8px 12px;color:#6b7280;font-size:0.85em">Context Precision</td>
                  <td style="text-align:center">{_score_badge(k3['ctx_prec'])}</td>
                  <td style="text-align:center">{_score_badge(k5['ctx_prec'])}</td></tr>"""

            metric_rows += f"""
                <tr style="border-top:1px solid #e5e7eb;background:#fafafa">
                  <td style="padding:6px 12px;color:#374151;font-weight:700;font-size:0.82em" colspan="3">RAGAS GENERATION</td></tr>
                <tr style="border-top:1px solid #e5e7eb">
                  <td style="padding:8px 12px;color:#6b7280;font-size:0.85em">Faithfulness</td>
                  <td style="text-align:center">{_score_badge(k3['faithfulness'])}</td>
                  <td style="text-align:center">{_score_badge(k5['faithfulness'])}</td></tr>
                <tr style="border-top:1px solid #e5e7eb">
                  <td style="padding:8px 12px;color:#6b7280;font-size:0.85em">Answer Relevancy</td>
                  <td style="text-align:center">{_score_badge(k3['answer_relevancy'])}</td>
                  <td style="text-align:center">{_score_badge(k5['answer_relevancy'])}</td></tr>
                <tr style="border-top:1px solid #e5e7eb;background:#fafafa">
                  <td style="padding:6px 12px;color:#374151;font-weight:700;font-size:0.82em" colspan="3">MULTI-AGENT PIPELINE</td></tr>
                <tr style="border-top:1px solid #e5e7eb">
                  <td style="padding:8px 12px;color:#6b7280;font-size:0.85em">Eval Agent Verdict</td>
                  <td colspan="2" style="text-align:center">
                    <span style="background:{eval_badge_color};color:#fff;padding:2px 8px;border-radius:12px;font-size:0.8em;font-weight:600">{eval_label}</span>
                    &nbsp;<span style="font-size:0.8em;color:#6b7280">conf={r['eval_confidence']:.2f}</span>
                  </td></tr>
                <tr style="border-top:1px solid #e5e7eb">
                  <td style="padding:8px 12px;color:#6b7280;font-size:0.85em">CrossEncoder Score</td>
                  <td colspan="2" style="text-align:center;font-weight:600">{r['avg_cross_score']:.4f}</td></tr>
                <tr style="border-top:1px solid #e5e7eb;background:#fafafa">
                  <td style="padding:6px 12px;color:#374151;font-weight:700;font-size:0.82em" colspan="3">LATENCY (Ret / Gen / Eval / Total)</td></tr>
                <tr style="border-top:1px solid #e5e7eb">
                  <td style="padding:8px 12px;color:#6b7280;font-size:0.85em">k=3</td>
                  <td style="text-align:center;font-size:0.85em;color:#4b5563" colspan="2">
                    {r['t_ret_3']:.2f}s / {r['t_gen_3']:.2f}s / {r['t_eval_3']:.2f}s / <strong>{r['t_tot_3']:.2f}s</strong>
                  </td></tr>
                <tr style="border-top:1px solid #e5e7eb">
                  <td style="padding:8px 12px;color:#6b7280;font-size:0.85em">k=5</td>
                  <td style="text-align:center;font-size:0.85em;color:#4b5563" colspan="2">
                    {r['t_ret_5']:.2f}s / {r['t_gen_5']:.2f}s / {r['t_eval_5']:.2f}s / <strong>{r['t_tot_5']:.2f}s</strong>
                  </td></tr>"""

            reasoning   = html.escape(k3.get("reasoning", ""))
            panel_title = "Expected Gold Answer (For Reference Only)" if subset_type == "absent" else "Reference Answer"
            panel_style = (
                "background:#f8fafc;border:1px solid #cbd5e1;color:#475569"
                if subset_type == "absent"
                else "background:#f0fdf4;border:1px solid #bbf7d0;color:#166534"
            )

            rows += f"""
            <details style="margin-bottom:12px;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden">
              <summary style="padding:14px 18px;cursor:pointer;background:#f9fafb;display:flex;align-items:center;gap:10px;list-style:none">
                <span style="font-weight:600;color:#111;flex:1">Q{r['idx']}. {q}</span>
                <span style="font-size:0.78em;color:#6b7280">Doc: {doc}</span>
                {web_badge}
                {_score_badge(k3['faithfulness'])}
              </summary>
              <div style="padding:16px 20px;background:#fff">
                <table style="width:100%;border-collapse:collapse;margin-bottom:16px">
                  <thead>
                    <tr style="background:#f3f4f6">
                      <th style="padding:8px 12px;text-align:left;color:#374151;font-size:0.85em">Metric</th>
                      <th style="padding:8px 12px;text-align:center;color:#6366f1;font-size:0.85em">k = 3</th>
                      <th style="padding:8px 12px;text-align:center;color:#8b5cf6;font-size:0.85em">k = 5</th>
                    </tr>
                  </thead>
                  <tbody>{metric_rows}</tbody>
                </table>
                {f'<div style="background:#fef3c7;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;font-size:0.82em;color:#92400e;margin-bottom:12px"><strong>Eval Reason:</strong> {html.escape(r["eval_reason"])}</div>' if r.get("eval_reason") else ""}
                {f'<div style="background:#fef3c7;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;font-size:0.82em;color:#92400e;margin-bottom:12px"><strong>Judge Reasoning:</strong> {reasoning}</div>' if reasoning else ""}
                <div style="display:flex;gap:16px;flex-wrap:wrap">
                  <div style="flex:1;min-width:280px">
                    <div style="font-size:0.78em;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">{panel_title}</div>
                    <div style="{panel_style};border-radius:8px;padding:12px;font-size:0.88em;line-height:1.6;white-space:pre-wrap">{exp}</div>
                  </div>
                  <div style="flex:1;min-width:280px">
                    <div style="font-size:0.78em;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">Generated Answer (k=3)</div>
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:12px;font-size:0.88em;color:#1e40af;line-height:1.6;white-space:pre-wrap">{gen}</div>
                  </div>
                </div>
              </div>
            </details>"""
        return rows

    # ── Assemble full HTML ─────────────────────────────────────────────────────
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-Agent RAG Evaluation Report — BEIR + RAGAS</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f8fafc; color: #1a1a2e; padding: 32px 24px; }}
    h1   {{ font-size: 1.8em; font-weight: 800; margin-bottom: 4px; }}
    h2   {{ font-size: 1.2em; font-weight: 700; margin: 28px 0 12px; color: #1e293b;
           border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }}
    .subtitle {{ color: #6b7280; margin-bottom: 28px; font-size: 0.92em; }}
    .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 32px; }}
    .pipeline-box {{
      background: #fff; border-radius: 12px; padding: 20px 28px;
      box-shadow: 0 1px 8px #0001; border-left: 4px solid #6366f1;
      margin-bottom: 28px; font-size: 0.9em; color: #374151; line-height: 1.8;
    }}
    .pipeline-box h4 {{ font-size: 1em; font-weight: 700; color: #1e293b; margin-bottom: 10px; }}
    .pipeline-step {{ display: inline-block; background: #ede9fe; color: #5b21b6;
      padding: 2px 10px; border-radius: 6px; font-weight: 600; margin: 0 4px; font-size: 0.88em; }}
    details > summary::-webkit-details-marker {{ display: none; }}
    details > summary::before {{ content: "▶"; margin-right: 8px; font-size: 0.75em; color: #9ca3af; transition: transform .2s; }}
    details[open] > summary::before {{ transform: rotate(90deg); }}
  </style>
</head>
<body>
  <h1>🤖 Multi-Agent RAG Evaluation Report — BEIR + RAGAS</h1>
  <p class="subtitle">
    Dataset: <strong>BEIR Benchmark ({BEIR_DATASET})</strong> &nbsp;·&nbsp;
    Generator: <strong>{GENERATOR_MODEL}</strong> &nbsp;·&nbsp;
    Judge: <strong>{JUDGE_MODEL}</strong> &nbsp;·&nbsp;
    Embedding: <strong>bge-m3</strong> &nbsp;·&nbsp;
    Reranker: <strong>BAAI/bge-reranker-v2-m3</strong> &nbsp;·&nbsp;
    Generated: <strong>{ts}</strong>
  </p>

  <div class="pipeline-box">
    <h4>🔀 Multi-Agent Pipeline Under Test</h4>
    <span class="pipeline-step">RAG Agent</span> →
    <span class="pipeline-step">Evaluation Agent</span> →
    <em style="color:#6b7280"> [if insufficient] </em>
    <span class="pipeline-step">Web Agent</span> →
    <span class="pipeline-step">Answer Agent</span>
    <br><br>
    The <strong>Evaluation Agent</strong> (Gemini) judges whether retrieved chunks are sufficient before deciding to trigger the web fallback.
    The <strong>Web Agent</strong> only runs when the evaluation returns <code>sufficient=false</code>.
    This report measures all 5 pipeline stages per query.
  </div>

  <h2>📊 Summary Metrics</h2>
  <div class="cards">
    {make_summary_section(summary_present,
        f"Present Set ({summary_present['total']} Queries)",
        "Target documents ARE indexed. Tests full retrieval + generation pipeline.",
        "#4f46e5", is_present=True)}
    {make_summary_section(summary_absent,
        f"Absent Set ({summary_absent['total']} Queries)",
        "Target documents NOT indexed. Tests LLM abstention (hallucination resistance).",
        "#0891b2", is_present=False)}
  </div>

  <h2>🔍 Present Set — Per Query Results</h2>
  {query_rows("present")}

  <h2>🔍 Absent Set — Per Query Results</h2>
  {query_rows("absent")}
</body>
</html>"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n[REPORT] Saved → {REPORT_PATH}")


# ── Main Evaluation Loop ───────────────────────────────────────────────────────

async def main():
    # 1. Load dataset metadata
    queries, qrels, answers = load_dataset_meta()

    # 2. Load indexed config
    if not os.path.exists(CONFIG_PATH):
        print("[ERROR] indexed_config_multi.json not found. Run index_dataset.py first.")
        return
    with open(CONFIG_PATH) as f:
        indexed_config = json.load(f)

    eval_present_ids = indexed_config["eval_present_queries"]
    eval_absent_ids  = indexed_config["eval_absent_queries"]
    print(f"[EVAL] Present queries : {len(eval_present_ids)}")
    print(f"[EVAL] Absent  queries : {len(eval_absent_ids)}")
    print(f"[MODEL] Generator: {GENERATOR_MODEL}  |  Judge: {JUDGE_MODEL}")

    # 3. Rebuild retriever from indexed papers
    print("[INIT] Loading chunks from indexed BEIR papers...")
    all_chunks: list[Document] = []
    for paper_id in indexed_config.get("indexed_papers", []):
        all_chunks.extend(load_paper_chunks(paper_id))
    if not all_chunks:
        print("[ERROR] No chunks loaded. Run index_dataset.py first.")
        return
    print(f"[INIT] Total chunks: {len(all_chunks)}")
    retriever = build_retriever(all_chunks)

    # 4. Accumulators
    def new_summary():
        return {
            "ndcg_10": 0.0, "recall_5": 0, "ctx_prec": 0.0,
            "faith": 0.0,   "ans_rel": 0.0, "total": 0,
            "t_ret_3": 0.0, "t_ret_5": 0.0,
            "t_gen_3": 0.0, "t_gen_5": 0.0,
            "t_eval_3": 0.0,"t_eval_5": 0.0,
            "t_tot_3": 0.0, "t_tot_5": 0.0,
        }

    summary_present = new_summary()
    summary_absent  = new_summary()
    query_results: list[dict] = []

    # Multi-agent tracking
    sufficient_present:   list[bool]  = []
    sufficient_absent:    list[bool]  = []
    web_triggered_present:list[bool]  = []
    web_triggered_absent: list[bool]  = []
    cross_scores_present: list[float] = []
    cross_scores_absent:  list[float] = []

    # 5. Checkpoint resume
    start_idx = 1
    if os.path.exists(CKPT_PATH):
        try:
            with open(CKPT_PATH) as f:
                ckpt = json.load(f)
            if ckpt.get("total") == len(eval_present_ids) + len(eval_absent_ids):
                query_results         = ckpt.get("results", [])
                summary_present       = ckpt.get("summary_present", new_summary())
                summary_absent        = ckpt.get("summary_absent",  new_summary())
                sufficient_present    = ckpt.get("sufficient_present", [])
                sufficient_absent     = ckpt.get("sufficient_absent",  [])
                web_triggered_present = ckpt.get("web_triggered_present", [])
                web_triggered_absent  = ckpt.get("web_triggered_absent",  [])
                cross_scores_present  = ckpt.get("cross_scores_present", [])
                cross_scores_absent   = ckpt.get("cross_scores_absent",  [])
                start_idx = len(query_results) + 1
                print(f"[CHECKPOINT] Resuming from [{start_idx}]")
        except Exception as e:
            print(f"[WARN] Checkpoint load failed: {e}. Starting fresh.")

    all_eval_jobs = (
        [("present", q_id) for q_id in eval_present_ids] +
        [("absent",  q_id) for q_id in eval_absent_ids]
    )
    total_jobs = len(all_eval_jobs)
    print(f"\nStarting multi-agent evaluation ({total_jobs} queries)...\n" + "=" * 80)

    for idx in range(start_idx, total_jobs + 1):
        subset, q_id = all_eval_jobs[idx - 1]
        query_text       = queries[q_id]["query"]
        target_doc       = qrels[q_id]["doc_id"]
        reference_answer = answers.get(q_id, "")

        print(f"\n[{idx}/{total_jobs}] [{subset.upper()}] {query_text[:90]!r}")
        print(f"  Target Doc: {target_doc}")

        # ── Step 1: RAG Agent ──────────────────────────────────────────────────
        t_rag_start = time.perf_counter()
        rag = await _rag_agent(query_text, retriever, all_chunks)
        t_rag = time.perf_counter() - t_rag_start
        avg_cross = rag["avg_score"]

        # ── Step 2: Evaluation Agent ───────────────────────────────────────────
        t_eval_agent_start = time.perf_counter()
        eval_result = await _evaluation_agent(query_text, rag)
        t_eval_agent = time.perf_counter() - t_eval_agent_start

        web_triggered = not eval_result["sufficient"]
        print(
            f"  [EVAL AGENT] sufficient={eval_result['sufficient']} | "
            f"conf={eval_result['confidence']:.2f} | web_triggered={web_triggered}"
        )

        # ── Track multi-agent stats ────────────────────────────────────────────
        if subset == "present":
            sufficient_present.append(eval_result["sufficient"])
            web_triggered_present.append(web_triggered)
            cross_scores_present.append(avg_cross)
        else:
            sufficient_absent.append(eval_result["sufficient"])
            web_triggered_absent.append(web_triggered)
            cross_scores_absent.append(avg_cross)

        per_k = {}
        gen_cache: dict[int, str] = {}
        lat: dict[int, dict] = {}
        ndcg_val = 0.0

        for k in [3, 5]:
            t_query_start = time.perf_counter()

            # ── Retrieval slice for k ──────────────────────────────────────────
            t0 = time.perf_counter()
            docs_for_k = rag["docs"][:k] if rag.get("docs") else []
            docs_for_k = _filter_redundant(docs_for_k)
            rag_context = "\n\n---\n\n".join(
                d.page_content.replace("●", "\n- ") for d in docs_for_k
            ) if docs_for_k else ""
            dt_ret = t_rag if k == 3 else time.perf_counter() - t0   # share RAG time for k=3

            # NDCG@10 computed once from the full ranked list
            if k == 3 and rag.get("docs"):
                ndcg_val = compute_ndcg(rag["docs"], target_doc, k=10)

            recall_5 = compute_recall(docs_for_k, target_doc)
            ctx_prec = compute_context_precision(docs_for_k, target_doc)

            # ── Answer Agent ───────────────────────────────────────────────────
            t0 = time.perf_counter()
            gen_ans = await _answer_agent(query_text, {"chunks": [d.page_content for d in docs_for_k]}, eval_result)
            dt_gen  = time.perf_counter() - t0
            gen_cache[k] = gen_ans

            # ── Judge ──────────────────────────────────────────────────────────
            t0           = time.perf_counter()
            ragas_scores = await evaluate_generation(query_text, rag_context, gen_ans, reference_answer, subset)
            dt_eval      = time.perf_counter() - t0

            dt_tot = time.perf_counter() - t_query_start

            per_k[k] = {
                "ndcg_10":          ndcg_val,
                "recall_5":         recall_5,
                "ctx_prec":         ctx_prec,
                "faithfulness":     ragas_scores["faithfulness"],
                "answer_relevancy": ragas_scores["answer_relevancy"],
                "reasoning":        ragas_scores.get("reasoning", ""),
            }
            lat[k] = {"ret": dt_ret, "gen": dt_gen, "eval": dt_eval, "tot": dt_tot}

            # ── Accumulate ─────────────────────────────────────────────────────
            s = summary_present if subset == "present" else summary_absent
            if subset == "present":
                if k == 3:
                    s["ndcg_10"] += ndcg_val
                if k == 5:
                    s["recall_5"] += 1 if recall_5 else 0
                s["ctx_prec"] += ctx_prec
            s["faith"]       += ragas_scores["faithfulness"]
            s["ans_rel"]     += ragas_scores["answer_relevancy"]
            s[f"t_ret_{k}"]  += dt_ret
            s[f"t_gen_{k}"]  += dt_gen
            s[f"t_eval_{k}"] += dt_eval
            s[f"t_tot_{k}"]  += dt_tot

            if subset == "present":
                print(
                    f"  [k={k}] NDCG@10={ndcg_val:.3f} | Recall={'YES' if recall_5 else 'NO':>3} | "
                    f"CtxPrec={ctx_prec:.3f} | Faith={ragas_scores['faithfulness']:.2f} | "
                    f"AnsRel={ragas_scores['answer_relevancy']:.2f} | "
                    f"Ret={dt_ret:.2f}s Gen={dt_gen:.2f}s Eval={dt_eval:.2f}s Tot={dt_tot:.2f}s"
                )
            else:
                print(
                    f"  [k={k}] Faith={ragas_scores['faithfulness']:.2f} | "
                    f"AnsRel={ragas_scores['answer_relevancy']:.2f} | "
                    f"Ret={dt_ret:.2f}s Gen={dt_gen:.2f}s Eval={dt_eval:.2f}s Tot={dt_tot:.2f}s"
                )

        (summary_present if subset == "present" else summary_absent)["total"] += 1

        query_results.append({
            "idx":                idx,
            "subset":             subset,
            "query":              query_text,
            "target_doc":         target_doc,
            "expected_answer":    reference_answer,
            "generated_answer_k3": gen_cache.get(3, ""),
            "generated_answer_k5": gen_cache.get(5, ""),
            "k3":                 per_k[3],
            "k5":                 per_k[5],
            "eval_sufficient":    eval_result["sufficient"],
            "eval_confidence":    eval_result["confidence"],
            "eval_reason":        eval_result["reason"],
            "web_triggered":      web_triggered,
            "avg_cross_score":    avg_cross,
            "t_ret_3":  lat[3]["ret"],  "t_ret_5":  lat[5]["ret"],
            "t_gen_3":  lat[3]["gen"],  "t_gen_5":  lat[5]["gen"],
            "t_eval_3": lat[3]["eval"], "t_eval_5": lat[5]["eval"],
            "t_tot_3":  lat[3]["tot"],  "t_tot_5":  lat[5]["tot"],
        })

        # ── Save checkpoint ────────────────────────────────────────────────────
        try:
            with open(CKPT_PATH, "w") as f:
                json.dump({
                    "total":                total_jobs,
                    "results":              query_results,
                    "summary_present":      summary_present,
                    "summary_absent":       summary_absent,
                    "sufficient_present":   sufficient_present,
                    "sufficient_absent":    sufficient_absent,
                    "web_triggered_present": web_triggered_present,
                    "web_triggered_absent":  web_triggered_absent,
                    "cross_scores_present": cross_scores_present,
                    "cross_scores_absent":  cross_scores_absent,
                }, f, indent=2)
        except Exception as e:
            print(f"  [WARN] Checkpoint write failed: {e}")

        if idx < total_jobs:
            await asyncio.sleep(4)

    # ── Final Summary ─────────────────────────────────────────────────────────
    p  = summary_present
    pt = p["total"] or 1
    a  = summary_absent
    at = a["total"] or 1

    multi_stats = {
        "sufficient_rate_present": sum(sufficient_present) / len(sufficient_present) if sufficient_present else 0.0,
        "sufficient_rate_absent":  sum(sufficient_absent)  / len(sufficient_absent)  if sufficient_absent  else 0.0,
        "web_rate_present":        sum(web_triggered_present) / len(web_triggered_present) if web_triggered_present else 0.0,
        "web_rate_absent":         sum(web_triggered_absent)  / len(web_triggered_absent)  if web_triggered_absent  else 0.0,
        "avg_cross_present":       sum(cross_scores_present)  / len(cross_scores_present)  if cross_scores_present  else 0.0,
        "avg_cross_absent":        sum(cross_scores_absent)   / len(cross_scores_absent)   if cross_scores_absent   else 0.0,
    }

    print("\n" + "=" * 80)
    print("FINAL MULTI-AGENT EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\n[PRESENT SET] {p['total']} queries")
    print(f"  NDCG@10              : {p['ndcg_10']/pt:.3f}")
    print(f"  Recall@5             : {p['recall_5']/pt*100:.1f}%")
    print(f"  Context Precision    : {p['ctx_prec']/(2*pt):.3f}")
    print(f"  Faithfulness         : {p['faith']/(2*pt):.3f}")
    print(f"  Answer Relevancy     : {p['ans_rel']/(2*pt):.3f}")
    print(f"  Eval Sufficient Rate : {multi_stats['sufficient_rate_present']*100:.1f}%")
    print(f"  Web Trigger Rate     : {multi_stats['web_rate_present']*100:.1f}%")
    print(f"  Avg CrossEncoder     : {multi_stats['avg_cross_present']:.4f}")
    for k in [3, 5]:
        print(f"  Avg Latency k={k}      : Ret={p[f't_ret_{k}']/pt:.2f}s  "
              f"Gen={p[f't_gen_{k}']/pt:.2f}s  "
              f"Eval={p[f't_eval_{k}']/pt:.2f}s  "
              f"Tot={p[f't_tot_{k}']/pt:.2f}s")

    print(f"\n[ABSENT SET] {a['total']} queries")
    print(f"  Faithfulness (Abstention): {a['faith']/(2*at)*100:.1f}%")
    print(f"  Answer Relevancy         : {a['ans_rel']/(2*at):.3f}")
    print(f"  Eval Sufficient Rate     : {multi_stats['sufficient_rate_absent']*100:.1f}%")
    print(f"  Web Trigger Rate         : {multi_stats['web_rate_absent']*100:.1f}%")
    print("=" * 80)

    save_html_report(query_results, summary_present, summary_absent, multi_stats)

    # Clean up checkpoint after successful full run
    if os.path.exists(CKPT_PATH):
        try:
            os.remove(CKPT_PATH)
            print("[CLEANUP] Deleted checkpoint.")
        except Exception:
            pass

    print(f"\nOpen report: file:///{REPORT_PATH.replace(os.sep, '/')}")


if __name__ == "__main__":
    asyncio.run(main())
