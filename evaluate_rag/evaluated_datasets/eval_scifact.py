"""
run_eval.py — Comprehensive RAG Evaluator (BEIR + RAGAS Metrics).
"""

import asyncio
import os
import json
import time
from datetime import datetime
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from evaluate_rag.config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from evaluate_rag.retriever import build_retriever
from evaluate_rag.rag_pipeline import retrieve_chunks

# Shared imports
from evaluate_rag.evaluated_datasets.common import (
    compute_recall,
    compute_context_precision,
    compute_ndcg,
    run_agent_generation,
    evaluate_generation_judge,
)

# ── Dataset & output paths ─────────────────────────────────────────────────────
import datasets
from evaluate_rag.config import BEIR_DATASET

REPORT_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval_report_rag1.html"))

GENERATOR_MODEL = LLM_MODEL        # gemini-3.1-flash-lite
JUDGE_MODEL     = "gemini-3.1-flash-lite"

# ── LLMs ───────────────────────────────────────────────────────────────────────
generator_llm = ChatGoogleGenerativeAI(
    model=GENERATOR_MODEL, google_api_key=GOOGLE_API_KEY, temperature=LLM_TEMPERATURE
)
judge_llm = ChatGoogleGenerativeAI(
    model=JUDGE_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0
)


# ── Dataset loader ─────────────────────────────────────────────────────────────

_cached_corpus = None

def get_corpus_dict():
    global _cached_corpus
    if _cached_corpus is None:
        print(f"[INFO] Loading BEIR corpus for '{BEIR_DATASET}'...")
        ds = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "corpus")
        split_name = list(ds.keys())[0]  # usually 'corpus'
        corpus_ds = ds[split_name]
        _cached_corpus = {row["_id"]: row for row in corpus_ds}
    return _cached_corpus


def load_dataset():
    print(f"[INFO] Loading BEIR queries for '{BEIR_DATASET}'...")
    queries_ds = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "queries")
    split_name = list(queries_ds.keys())[0]
    queries_list = queries_ds[split_name]
    
    queries = {row["_id"]: {"query": row["text"]} for row in queries_list}
    
    print(f"[INFO] Loading BEIR qrels for '{BEIR_DATASET}'...")
    qrels_ds = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "default")
    qrels_rows = []
    for split in qrels_ds.keys():
        qrels_rows.extend(qrels_ds[split])
        
    qrels = {}
    for row in qrels_rows:
        q_id = row["query-id"]
        c_id = row["corpus-id"]
        score = row["score"]
        if score >= 1:
            if q_id not in qrels:
                qrels[q_id] = []
            if c_id not in qrels[q_id]:
                qrels[q_id].append(c_id)
                
    corpus_dict = get_corpus_dict()
    answers = {}
    for q_id, doc_ids in qrels.items():
        if doc_ids:
            doc_row = corpus_dict.get(doc_ids[0])
            if doc_row:
                title = doc_row.get("title", "")
                text = doc_row.get("text", "")
                answers[q_id] = f"{title}\n{text}" if title else text
            else:
                answers[q_id] = ""
        else:
            answers[q_id] = ""
            
    return queries, qrels, answers


def load_paper_chunks(paper_id: str) -> list[Document]:
    corpus = get_corpus_dict()
    doc_row = corpus.get(paper_id)
    if not doc_row:
        return []
    title = doc_row.get("title", "")
    text = doc_row.get("text", "")
    full_text = f"{title}\n{text}" if title else text
    full_text = " ".join(full_text.split())
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from evaluate_rag.config import CHUNK_SIZE, CHUNK_OVERLAP
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    doc = Document(page_content=full_text, metadata={"source": paper_id, "title": title})
    return splitter.split_documents([doc])


# ── HTML Report ─────────────────────────────────────────────────────────────────

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
    summary_absent: dict,
) -> None:
    import html as html_escape_mod
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def make_summary_section(s: dict, title: str, desc: str, color: str, is_present: bool = True) -> str:
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
                <td style="text-align:right;font-weight:700;color:#6366f1">{s['ans_rel']/(2*tot):.3f}</td></tr>"""
        else:
            metrics_rows = f"""
            <tr><td colspan="2" style="padding:6px 0;font-weight:700;color:{color};font-size:0.85em;text-transform:uppercase">RAGAS Generation (Absent Set)</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#22c55e;font-weight:600">Abstention Rate (Faithfulness)</td>
                <td style="text-align:right;font-weight:700;color:#22c55e">{s['faith']/(2*tot)*100:.1f}%</td></tr>
            <tr><td style="padding:4px 0 4px 12px;color:#4b5563">Answer Relevancy</td>
                <td style="text-align:right;font-weight:700;color:#6366f1">{s['ans_rel']/(2*tot):.3f}</td></tr>"""

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

    def query_rows(subset_type: str) -> str:
        rows = ""
        for r in [x for x in query_results if x["subset"] == subset_type]:
            q   = html_escape_mod.escape(r["query"])
            doc = html_escape_mod.escape(", ".join(r["target_docs"]) if isinstance(r["target_docs"], list) else r["target_docs"])
            exp = html_escape_mod.escape(r["expected_answer"])
            gen = html_escape_mod.escape(r["generated_answer_k3"])
            k3  = r["k3"]
            k5  = r["k5"]

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
                  <td style="text-align:center">{_badge(k3['recall_5'] > 0.5, 'HIT', 'MISS')}</td>
                  <td style="text-align:center">{_badge(k5['recall_5'] > 0.5, 'HIT', 'MISS')}</td></tr>
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

            reasoning = html_escape_mod.escape(k3.get("reasoning", ""))
            panel_title = "Expected Gold Answer (For Reference Only)" if subset_type == "absent" else "Reference Answer"
            panel_style = ("background:#f8fafc;border:1px solid #cbd5e1;color:#475569"
                           if subset_type == "absent"
                           else "background:#f0fdf4;border:1px solid #bbf7d0;color:#166534")

            rows += f"""
            <details style="margin-bottom:12px;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden">
              <summary style="padding:14px 18px;cursor:pointer;background:#f9fafb;display:flex;align-items:center;gap:10px;list-style:none">
                <span style="font-weight:600;color:#111;flex:1">Q{r['idx']}. {q}</span>
                <span style="font-size:0.78em;color:#6b7280">Doc: {doc}</span>
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
                {f'<div style="background:#fef3c7;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;font-size:0.82em;color:#92400e;margin-bottom:12px"><strong>Judge Reasoning:</strong> {reasoning}</div>' if reasoning else ''}
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

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Evaluation Report — BEIR + RAGAS</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f8fafc; color: #1a1a2e; padding: 32px 24px; }}
    h1   {{ font-size: 1.8em; font-weight: 800; margin-bottom: 4px; }}
    h2   {{ font-size: 1.2em; font-weight: 700; margin: 28px 0 12px; color: #1e293b;
           border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }}
    .subtitle {{ color: #6b7280; margin-bottom: 28px; font-size: 0.92em; }}
    .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 32px; }}
    details > summary::-webkit-details-marker {{ display: none; }}
    details > summary::before {{ content: "▶"; margin-right: 8px; font-size: 0.75em; color: #9ca3af; transition: transform .2s; }}
    details[open] > summary::before {{ transform: rotate(90deg); }}
  </style>
</head>
<body>
  <h1>🧪 RAG Evaluation Report — BEIR + RAGAS</h1>
  <p class="subtitle">
    Dataset: <strong>BEIR Benchmark ({BEIR_DATASET})</strong> &nbsp;·&nbsp;
    Generator: <strong>{GENERATOR_MODEL}</strong> &nbsp;·&nbsp;
    Judge: <strong>{JUDGE_MODEL}</strong> &nbsp;·&nbsp;
    Generated: <strong>{ts}</strong>
  </p>

  <h2>📊 Summary Metrics</h2>
  <div class="cards">
    {make_summary_section(summary_present, f"Present Set ({summary_present['total']} Queries)",
        "Target documents ARE indexed. Tests full retrieval + generation pipeline.", "#4f46e5", is_present=True)}
    {make_summary_section(summary_absent, f"Absent Set ({summary_absent['total']} Queries)",
        "Target documents NOT indexed. Tests LLM abstention (hallucination resistance).", "#0891b2", is_present=False)}
  </div>

  <h2>🔍 Present Set — Per Query Results</h2>
  {query_rows("present")}

  <h2>🔍 Absent Set — Per Query Results</h2>
  {query_rows("absent")}
</body>
</html>"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n[REPORT] Saved: {REPORT_PATH}")


# ── Main Evaluation ────────────────────────────────────────────────────────────

async def main():
    print("[INIT] Loading dataset...")
    queries, qrels, answers = load_dataset()

    config_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "indexed_config.json"))
    if not os.path.exists(config_path):
        print("[ERROR] indexed_config.json not found. Run index_dataset.py first.")
        return

    with open(config_path, "r") as f:
        indexed_config = json.load(f)

    eval_present_ids = indexed_config.get("eval_present_queries", [])
    eval_absent_ids  = indexed_config.get("eval_absent_queries", [])

    print(f"[EVAL] Selected {len(eval_present_ids)} Present queries.")
    print(f"[EVAL] Selected {len(eval_absent_ids)} Absent queries.")

    # Reconstruct retriever
    print("[INIT] Reconstructing retriever from indexed BEIR documents...")
    all_chunks = []
    for paper_id in indexed_config.get("indexed_papers", []):
        all_chunks.extend(load_paper_chunks(paper_id))

    if not all_chunks:
        print("[ERROR] No chunks loaded. Check dataset files.")
        return

    retriever = build_retriever(all_chunks)

    # ── Accumulators ──────────────────────────────────────────────────────────
    def new_summary():
        return {
            "ndcg_10": 0.0, "recall_5": 0, "ctx_prec": 0.0,
            "faith": 0.0, "ans_rel": 0.0,
            "total": 0,
            "t_ret_3": 0.0, "t_ret_5": 0.0,
            "t_gen_3": 0.0, "t_gen_5": 0.0,
            "t_eval_3": 0.0, "t_eval_5": 0.0,
            "t_tot_3": 0.0, "t_tot_5": 0.0,
        }

    summary_present = new_summary()
    summary_absent  = new_summary()
    query_results   = []

    all_eval_jobs = (
        [("present", q_id) for q_id in eval_present_ids] +
        [("absent",  q_id) for q_id in eval_absent_ids]
    )
    total_jobs = len(all_eval_jobs)
    print(f"\nStarting evaluation ({total_jobs} queries)...\n" + "=" * 80)

    for idx, (subset, q_id) in enumerate(all_eval_jobs, start=1):
        query_text       = queries[q_id]["query"]
        target_docs      = qrels.get(q_id, [])
        reference_answer = answers.get(q_id, "")

        print(f"\n[{idx}/{total_jobs}] [{subset.upper()}] Query : {query_text!r}")
        print(f"         Target Docs : {target_docs}")

        per_k = {}
        generated_ans_cache = {}
        latencies_query = {}
        ndcg_val = 0.0

        for k in [3, 5]:
            t_query_start = time.perf_counter()

            # Retrieval
            t0       = time.perf_counter()
            pipeline = retrieve_chunks(query=query_text, retriever=retriever, all_chunks=all_chunks, top_k=k)
            dt_ret   = time.perf_counter() - t0

            if k == 3:
                ndcg_val = compute_ndcg(pipeline["retrieved_unique"], target_docs, k=10)

            recall_5 = compute_recall(pipeline["retrieved_final"], target_docs)
            ctx_prec = compute_context_precision(pipeline["retrieved_final"], target_docs)

            # Generation
            t0      = time.perf_counter()
            gen_ans = await run_agent_generation(query_text, pipeline["rag_context"], generator_llm)
            dt_gen  = time.perf_counter() - t0
            generated_ans_cache[k] = gen_ans

            # RAGAS Evaluation (single LLM call)
            t0           = time.perf_counter()
            ragas_scores = await evaluate_generation_judge(
                query_text, pipeline["rag_context"], gen_ans, judge_llm, reference_answer, subset
            )
            dt_eval = time.perf_counter() - t0

            dt_tot = time.perf_counter() - t_query_start

            per_k[k] = {
                "ndcg_10":            ndcg_val,
                "recall_5":           recall_5,
                "ctx_prec":           ctx_prec,
                "faithfulness":       ragas_scores["faithfulness"],
                "answer_relevancy":   ragas_scores["answer_relevancy"],
                "reasoning":          ragas_scores.get("reasoning", ""),
            }
            latencies_query[k] = {"ret": dt_ret, "gen": dt_gen, "eval": dt_eval, "tot": dt_tot}

            # Accumulate
            s = summary_present if subset == "present" else summary_absent
            if subset == "present":
                if k == 3:
                    s["ndcg_10"]  += ndcg_val
                if k == 5:
                    s["recall_5"] += recall_5
                s["ctx_prec"] += ctx_prec
            s["faith"]        += ragas_scores["faithfulness"]
            s["ans_rel"]      += ragas_scores["answer_relevancy"]
            s[f"t_ret_{k}"]   += dt_ret
            s[f"t_gen_{k}"]   += dt_gen
            s[f"t_eval_{k}"]  += dt_eval
            s[f"t_tot_{k}"]   += dt_tot

            if subset == "present":
                print(
                    f"  [k={k}] NDCG@10={ndcg_val:.3f} | Recall@5={recall_5:.1f} | "
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
            "idx":                 idx,
            "subset":              subset,
            "query":               query_text,
            "target_docs":         target_docs,
            "expected_answer":     reference_answer,
            "generated_answer_k3": generated_ans_cache.get(3, ""),
            "generated_answer_k5": generated_ans_cache.get(5, ""),
            "k3":                  per_k[3],
            "k5":                  per_k[5],
            "t_ret_3":             latencies_query[3]["ret"],
            "t_ret_5":             latencies_query[5]["ret"],
            "t_gen_3":             latencies_query[3]["gen"],
            "t_gen_5":             latencies_query[5]["gen"],
            "t_eval_3":            latencies_query[3]["eval"],
            "t_eval_5":            latencies_query[5]["eval"],
            "t_tot_3":             latencies_query[3]["tot"],
            "t_tot_5":             latencies_query[5]["tot"],
        })

        if idx < total_jobs:
            await asyncio.sleep(4)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)

    p = summary_present
    pt = p["total"] or 1
    print(f"\n[PRESENT SET] {p['total']} queries")
    print(f"  NDCG@10           : {p['ndcg_10']/pt:.3f}")
    print(f"  Recall@5          : {p['recall_5']/pt*100:.1f}%")
    print(f"  Context Precision : {p['ctx_prec']/(2*pt):.3f}")
    print(f"  Faithfulness      : {p['faith']/(2*pt):.3f}")
    print(f"  Answer Relevancy  : {p['ans_rel']/(2*pt):.3f}")
    for k in [3, 5]:
        print(f"  Avg Latency k={k}   : Ret={p[f't_ret_{k}']/pt:.2f}s  "
              f"Gen={p[f't_gen_{k}']/pt:.2f}s  "
              f"Eval={p[f't_eval_{k}']/pt:.2f}s  "
              f"Tot={p[f't_tot_{k}']/pt:.2f}s")

    a = summary_absent
    at = a["total"] or 1
    print(f"\n[ABSENT SET] {a['total']} queries")
    print(f"  Faithfulness (Abstention Rate): {a['faith']/(2*at):.3f}  ({a['faith']/(2*at)*100:.1f}%)")
    print(f"  Answer Relevancy              : {a['ans_rel']/(2*at):.3f}")
    for k in [3, 5]:
        print(f"  Avg Latency k={k}              : Ret={a[f't_ret_{k}']/at:.2f}s  "
              f"Gen={a[f't_gen_{k}']/at:.2f}s  "
              f"Eval={a[f't_eval_{k}']/at:.2f}s  "
              f"Tot={a[f't_tot_{k}']/at:.2f}s")
    print("=" * 80)

    save_html_report(query_results, summary_present, summary_absent)
    print(f"\nOpen report: file:///{REPORT_PATH.replace(os.sep, '/')}")


if __name__ == "__main__":
    asyncio.run(main())
