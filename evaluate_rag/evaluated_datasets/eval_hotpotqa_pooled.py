"""
eval_hotpotqa_pooled.py — Evaluates the RAG pipeline on the HotpotQA dataset
using a pooled shared-index of ~500 documents.
"""

import os
import sys
import time
import json
import asyncio
import html
import dotenv
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import load_dataset
from sentence_transformers import CrossEncoder

# Ragas imports
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Shared imports
from evaluate_rag.retriever import RerankedRetriever
from evaluate_rag.rag_pipeline import retrieve_chunks
from evaluate_rag.evaluated_datasets.common import (
    compute_recall,
    compute_context_precision,
    compute_ndcg,
    run_agent_generation,
    evaluate_with_ragas,
)

# Load environment — must point to project root .env
_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_project_root = os.path.abspath(os.path.join(_dir, ".."))
dotenv.load_dotenv(dotenv_path=os.path.join(_project_root, ".env"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
print(f"[INIT] Using API key: {GOOGLE_API_KEY[:20]}...")

REPORT_PATH = os.path.join(_dir, "eval_report_hotpot_pooled.html")

# Initialize Models
GENERATOR_MODEL = "gemini-3.1-flash-lite"

generator_llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2)

# Local Embedding Model & GPU Reranker
embeddings = OllamaEmbeddings(model="bge-m3")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# Ragas Wrappers
ragas_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(model=GENERATOR_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0)
)
# Use your LOCAL embedding model for Ragas to save Google API calls!
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# Instantiate the modern Ragas metrics
faithfulness = Faithfulness(llm=ragas_llm)
answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1)


async def main():
    print("[INIT] Loading HotpotQA validation split...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    # 50 queries × 2 k-values × 4 API calls = 400 total (within 450 limit)
    eval_size = 50
    subset = dataset.select(range(eval_size))
    
    # ── Step 1: Build Shared Pool of Supporting Documents ─────────────────────
    print("[INIT] Extracting positive (gold) Wikipedia paragraphs for the shared index...")
    unique_paragraphs = {}  # title -> paragraph text
    
    for example in subset:
        gold_titles = list(set(example["supporting_facts"]["title"]))
        
        # Look through paragraphs in the context list for this query
        for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
            if title in gold_titles:
                paragraph = "".join(sentences)
                if title not in unique_paragraphs:
                    unique_paragraphs[title] = paragraph

    # Add ~10 gold paragraphs from other queries as distractors to each query's pool,
    # or just pool ALL 210 gold paragraphs together to simulate a shared database index
    docs_to_index = [
        Document(page_content=text, metadata={"source": title})
        for title, text in unique_paragraphs.items()
    ]
    print(f"[INIT] Total unique paragraphs in shared index: {len(docs_to_index)}")

    # ── Step 2: Index Corpus ───────────────────────────────────────────────────
    print("[INIT] Indexing corpus into shared Chroma database...")
    t_idx = time.perf_counter()
    vectorstore = Chroma.from_documents(
        documents=docs_to_index,
        embedding=embeddings,
        collection_name="temp_hotpot_pooled"
    )
    print(f"[SUCCESS] Indexed {len(docs_to_index)} docs in {time.perf_counter() - t_idx:.2f} seconds.")

    # ── Step 3: Configure Retrieval Stack ─────────────────────────────────────
    bm25_retriever = BM25Retriever.from_documents(docs_to_index)
    bm25_retriever.k = 10
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )
    retriever = RerankedRetriever(ensemble_retriever, reranker_model_name="BAAI/bge-reranker-v2-m3", top_n=10)

    summary = {
        "ndcg_10": 0.0, "recall_5": 0, "ctx_prec": 0.0,
        "faith": 0.0, "ans_rel": 0.0, "total": 0,
        "t_ret_3": 0.0, "t_ret_5": 0.0,
        "t_gen_3": 0.0, "t_gen_5": 0.0,
        "t_eval_3": 0.0, "t_eval_5": 0.0,
        "t_tot_3": 0.0, "t_tot_5": 0.0,
    }
    query_results = []

    print(f"\nStarting HotpotQA Shared-Index Evaluation ({eval_size} queries)...\n" + "="*80)

    for idx, example in enumerate(subset, start=1):
        query_text = example["question"]
        gold_answer = example["answer"]
        gold_titles = list(set(example["supporting_facts"]["title"]))
        
        print(f"\n[{idx}/{eval_size}] Query: {query_text!r}")
        print(f"         Supporting docs: {gold_titles}")

        per_k = {}
        generated_ans_cache = {}
        latencies_query = {}
        ndcg_val = 0.0

        for k in [3, 5]:
            t_query_start = time.perf_counter()

            # Retrieval
            t0 = time.perf_counter()
            pipeline = retrieve_chunks(query=query_text, retriever=retriever, all_chunks=docs_to_index, top_k=k)
            final_docs = pipeline["retrieved_final"]
            unique = pipeline["retrieved_unique"]
            rag_context = pipeline["rag_context"]
            dt_ret = pipeline["retrieval_time"]

            if k == 3:
                ndcg_val = compute_ndcg(unique, gold_titles, k=10)

            recall_5 = compute_recall(final_docs, gold_titles)
            ctx_prec = compute_context_precision(final_docs, gold_titles)

            # Generation
            t0 = time.perf_counter()
            gen_ans = await run_agent_generation(query_text, rag_context, generator_llm)
            dt_gen = time.perf_counter() - t0
            generated_ans_cache[k] = gen_ans

            # Evaluation using official Ragas (async thread executor)
            t0 = time.perf_counter()
            ragas_scores = await evaluate_with_ragas(
                query_text, rag_context, gen_ans, faithfulness, answer_relevancy
            )
            dt_eval = time.perf_counter() - t0

            dt_tot = time.perf_counter() - t_query_start

            per_k[k] = {
                "ndcg_10": ndcg_val,
                "recall_5": recall_5,
                "ctx_prec": ctx_prec,
                "faithfulness": ragas_scores["faithfulness"],
                "answer_relevancy": ragas_scores["answer_relevancy"],
            }
            latencies_query[k] = {"ret": dt_ret, "gen": dt_gen, "eval": dt_eval, "tot": dt_tot}

            # Accumulate
            if k == 3:
                summary["ndcg_10"] += ndcg_val
            if k == 5:
                summary["recall_5"] += recall_5
            summary["ctx_prec"] += ctx_prec
            summary["faith"] += ragas_scores["faithfulness"]
            summary["ans_rel"] += ragas_scores["answer_relevancy"]
            summary[f"t_ret_{k}"] += dt_ret
            summary[f"t_gen_{k}"] += dt_gen
            summary[f"t_eval_{k}"] += dt_eval
            summary[f"t_tot_{k}"] += dt_tot

            print(
                f"  [k={k}] NDCG@10={ndcg_val:.3f} | Recall@5={recall_5:.3f} | "
                f"CtxPrec={ctx_prec:.3f} | Faith={ragas_scores['faithfulness']:.2f} | "
                f"AnsRel={ragas_scores['answer_relevancy']:.2f} | "
                f"Ret={dt_ret:.2f}s Gen={dt_gen:.2f}s"
            )

        summary["total"] += 1
        query_results.append({
            "idx": idx,
            "query": query_text,
            "gold_titles": gold_titles,
            "gold_answer": gold_answer,
            "generated_answer_k3": generated_ans_cache.get(3, ""),
            "generated_answer_k5": generated_ans_cache.get(5, ""),
            "k3": per_k[3],
            "k5": per_k[5],
            "t_ret_3": latencies_query[3]["ret"],
            "t_ret_5": latencies_query[5]["ret"],
            "t_gen_3": latencies_query[3]["gen"],
            "t_gen_5": latencies_query[5]["gen"],
            "t_eval_3": latencies_query[3]["eval"],
            "t_eval_5": latencies_query[5]["eval"],
            "t_tot_3": latencies_query[3]["tot"],
            "t_tot_5": latencies_query[5]["tot"],
        })

        if idx < eval_size:
            await asyncio.sleep(4)

    # Clean up DB
    vectorstore.delete_collection()

    # Output stats
    pt = summary["total"] or 1
    n2 = 2 * pt  # two k values per query
    avg_ndcg    = summary["ndcg_10"] / pt
    avg_recall  = summary["recall_5"] / pt
    avg_ctxprec = summary["ctx_prec"] / n2
    avg_faith   = summary["faith"]    / n2
    avg_rel     = summary["ans_rel"]  / n2

    print("\n" + "="*80)
    print("FINAL HOTPOTQA POOLED-INDEX SUMMARY")
    print("="*80)
    print(f"  NDCG@10           : {avg_ndcg:.3f}")
    print(f"  Recall@5          : {avg_recall*100:.1f}%")
    print(f"  Context Precision : {avg_ctxprec:.3f}")
    print(f"  Faithfulness      : {avg_faith:.3f}")
    print(f"  Answer Relevancy  : {avg_rel:.3f}")
    print("="*80)

    save_html_report(query_results, avg_ndcg, avg_recall, avg_ctxprec, avg_faith, avg_rel)
    print(f"\n[REPORT] Open: file:///{REPORT_PATH.replace(os.sep, '/')}")

    sys.stderr = open(os.devnull, 'w')


def save_html_report(results, avg_ndcg, avg_recall, avg_ctxprec, avg_faith, avg_rel):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(results)

    # ── Build per-query rows ──
    rows_html = ""
    for r in results:
        q_esc  = html.escape(r["query"])
        ga_esc = html.escape(r["gold_answer"])
        gt_esc = html.escape(", ".join(r["gold_titles"]))

        def score_cls(v):
            return "score-high" if v >= 0.8 else ("score-mid" if v >= 0.5 else "score-low")

        # k=3 row
        k3 = r["k3"]
        ans3 = html.escape(r["generated_answer_k3"][:300] + ("…" if len(r["generated_answer_k3"]) > 300 else ""))
        rows_html += f"""
        <tr>
          <td class="idx" rowspan="2">{r['idx']}</td>
          <td class="query-cell" rowspan="2">{q_esc}<br><small style="color:var(--text-dim)">Gold: {gt_esc}</small></td>
          <td><span style="background:#2563eb22;color:#60a5fa;padding:2px 7px;border-radius:4px;font-size:0.75rem;">k=3</span></td>
          <td class="answer-cell">{ans3}</td>
          <td class="answer-cell">{ga_esc}</td>
          <td class="score {score_cls(k3['ndcg_10'])}">{k3['ndcg_10']:.3f}</td>
          <td class="score {score_cls(k3['recall_5'])}">{k3['recall_5']:.3f}</td>
          <td class="score {score_cls(k3['ctx_prec'])}">{k3['ctx_prec']:.3f}</td>
          <td class="score {score_cls(k3['faithfulness'])}">{k3['faithfulness']:.3f}</td>
          <td class="score {score_cls(k3['answer_relevancy'])}">{k3['answer_relevancy']:.3f}</td>
          <td class="latency">{r['t_ret_3']:.2f}s / {r['t_gen_3']:.2f}s / {r['t_eval_3']:.2f}s</td>
        </tr>"""

        # k=5 row
        k5 = r["k5"]
        ans5 = html.escape(r["generated_answer_k5"][:300] + ("…" if len(r["generated_answer_k5"]) > 300 else ""))
        rows_html += f"""
        <tr style="background:rgba(30,30,60,0.3)">
          <td><span style="background:#7c3aed22;color:#a78bfa;padding:2px 7px;border-radius:4px;font-size:0.75rem;">k=5</span></td>
          <td class="answer-cell">{ans5}</td>
          <td class="answer-cell">{ga_esc}</td>
          <td class="score {score_cls(k5['ndcg_10'])}">{k5['ndcg_10']:.3f}</td>
          <td class="score {score_cls(k5['recall_5'])}">{k5['recall_5']:.3f}</td>
          <td class="score {score_cls(k5['ctx_prec'])}">{k5['ctx_prec']:.3f}</td>
          <td class="score {score_cls(k5['faithfulness'])}">{k5['faithfulness']:.3f}</td>
          <td class="score {score_cls(k5['answer_relevancy'])}">{k5['answer_relevancy']:.3f}</td>
          <td class="latency">{r['t_ret_5']:.2f}s / {r['t_gen_5']:.2f}s / {r['t_eval_5']:.2f}s</td>
        </tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HotpotQA Pooled Eval Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0d0f1a; --surface: #12152b; --surface2: #1a1e38;
      --border: #2e3258; --text: #e2e8f0; --text-muted: #94a3b8;
      --text-dim: #64748b; --accent: #6366f1; --accent2: #8b5cf6;
      --green: #4ade80; --red: #f87171; --yellow: #fbbf24;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }}
    .header {{ background: linear-gradient(135deg, #1a1e38 0%, #0d0f1a 100%); border-bottom: 1px solid var(--border); padding: 2.5rem 3rem; }}
    .header h1 {{ font-size: 1.7rem; font-weight: 700; }}
    .header h1 span {{ background: linear-gradient(90deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .meta {{ color: var(--text-dim); font-size: 0.85rem; margin-top: 0.5rem; }}
    .pills {{ display: flex; gap: 0.6rem; margin-top: 1rem; flex-wrap: wrap; }}
    .pill {{ padding: 0.3rem 0.8rem; border-radius: 999px; font-size: 0.78rem; font-weight: 500; }}
    .pill-blue   {{ background: #1e40af33; color: #60a5fa; border: 1px solid #1e40af55; }}
    .pill-purple {{ background: #7c3aed33; color: #a78bfa; border: 1px solid #7c3aed55; }}
    .pill-green  {{ background: #15803d33; color: #4ade80; border: 1px solid #15803d55; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 2.5rem 3rem; }}
    /* Cards */
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2.5rem; }}
    .card {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
      padding: 1.4rem 1.6rem; position: relative; overflow: hidden;
    }}
    .card-label {{ font-size: 0.78rem; font-weight: 500; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    .card-value {{ font-size: 2.2rem; font-weight: 700; color: #fff; margin-top: 0.4rem; line-height: 1; }}
    .card-sub   {{ font-size: 0.8rem; color: var(--text-dim); margin-top: 0.5rem; }}
    /* Metric bars */
    .section-title {{ font-size: 1.1rem; font-weight: 600; color: var(--text); margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.6rem; }}
    .section-title::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}
    .bar-group {{ display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 2.5rem; }}
    .bar-row {{ display: flex; align-items: center; gap: 1rem; }}
    .bar-label {{ width: 180px; font-size: 0.82rem; color: var(--text-muted); flex-shrink: 0; text-align: right; }}
    .bar-track {{ flex: 1; height: 10px; background: var(--surface2); border-radius: 999px; overflow: hidden; }}
    .bar-fill  {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }}
    .bar-fill.green {{ background: linear-gradient(90deg, #22c55e, #16a34a); }}
    .bar-fill.yellow {{ background: linear-gradient(90deg, #f59e0b, #d97706); }}
    .bar-val   {{ width: 52px; font-size: 0.82rem; color: var(--text); font-weight: 600; }}
    /* Table */
    .table-wrap {{ overflow-x: auto; border-radius: 14px; border: 1px solid var(--border); margin-top: 1rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
    thead th {{ background: var(--surface); color: var(--text-muted); font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; padding: 0.85rem 0.8rem; border-bottom: 1px solid var(--border); position: sticky; top: 0; }}
    tbody tr {{ border-bottom: 1px solid rgba(46,50,88,0.5); transition: background 0.15s; }}
    tbody tr:hover {{ background: rgba(99,102,241,0.04); }}
    td {{ padding: 0.65rem 0.8rem; vertical-align: top; }}
    td.idx {{ color: var(--text-dim); font-size: 0.75rem; font-weight: 600; }}
    td.query-cell {{ min-width: 260px; max-width: 360px; }}
    td.answer-cell {{ min-width: 220px; max-width: 300px; color: var(--text-muted); font-size: 0.79rem; line-height: 1.5; }}
    td.score {{ text-align: center; font-weight: 600; }}
    td.latency {{ text-align: right; color: var(--text-dim); font-size: 0.77rem; white-space: nowrap; }}
    .score-high {{ color: #4ade80; }}
    .score-mid  {{ color: #fbbf24; }}
    .score-low  {{ color: #f87171; }}
    /* Search */
    .search-bar {{ width: 100%; padding: 0.65rem 1rem; border-radius: 8px; background: var(--surface); border: 1px solid var(--border); color: var(--text); font-size: 0.88rem; outline: none; margin-bottom: 1rem; transition: border-color 0.2s; }}
    .search-bar:focus {{ border-color: var(--accent); }}
    footer {{ text-align: center; padding: 2rem; color: var(--text-dim); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 3rem; }}
  </style>
</head>
<body>

<div class="header">
  <h1>📊 <span>HotpotQA Pooled-Index</span> Evaluation Report</h1>
  <div class="meta">Generated on {ts} | HotpotQA distractor split | {total} queries × 2 k-values</div>
  <div class="pills">
    <span class="pill pill-blue">Generator: {GENERATOR_MODEL}</span>
    <span class="pill pill-purple">Judge: Ragas (faithfulness + answer_relevancy)</span>
    <span class="pill pill-green">Retriever: BGE-M3 + BM25 Ensemble + Cross-Encoder</span>
  </div>
</div>

<div class="container">

  <!-- Summary Cards -->
  <div class="cards">
    <div class="card">
      <div class="card-label">NDCG@10</div>
      <div class="card-value">{avg_ndcg:.3f}</div>
      <div class="card-sub">Ranking quality (k=3 retrieval)</div>
    </div>
    <div class="card">
      <div class="card-label">Recall@5</div>
      <div class="card-value">{avg_recall*100:.1f}<span style="font-size:1.2rem">%</span></div>
      <div class="card-sub">Supporting docs in top-5</div>
    </div>
    <div class="card">
      <div class="card-label">Context Precision</div>
      <div class="card-value">{avg_ctxprec:.3f}</div>
      <div class="card-sub">Relevant chunks / total retrieved</div>
    </div>
    <div class="card">
      <div class="card-label">Faithfulness</div>
      <div class="card-value">{avg_faith:.3f}</div>
      <div class="card-sub">Ragas grounded-claim score</div>
    </div>
    <div class="card">
      <div class="card-label">Answer Relevancy</div>
      <div class="card-value">{avg_rel:.3f}</div>
      <div class="card-sub">Ragas semantic alignment score</div>
    </div>
  </div>

  <!-- Metric Bars -->
  <div class="section-title">📈 Metric Overview</div>
  <div class="bar-group">
    <div class="bar-row">
      <div class="bar-label">NDCG@10</div>
      <div class="bar-track"><div class="bar-fill green" style="width:{avg_ndcg*100:.1f}%"></div></div>
      <div class="bar-val">{avg_ndcg:.3f}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">Recall@5</div>
      <div class="bar-track"><div class="bar-fill green" style="width:{avg_recall*100:.1f}%"></div></div>
      <div class="bar-val">{avg_recall:.3f}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">Context Precision</div>
      <div class="bar-track"><div class="bar-fill yellow" style="width:{avg_ctxprec*100:.1f}%"></div></div>
      <div class="bar-val">{avg_ctxprec:.3f}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">Faithfulness (Ragas)</div>
      <div class="bar-track"><div class="bar-fill" style="width:{avg_faith*100:.1f}%"></div></div>
      <div class="bar-val">{avg_faith:.3f}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">Answer Relevancy (Ragas)</div>
      <div class="bar-track"><div class="bar-fill" style="width:{avg_rel*100:.1f}%"></div></div>
      <div class="bar-val">{avg_rel:.3f}</div>
    </div>
  </div>

  <!-- Per-Query Table -->
  <div class="section-title">🔍 Per-Query Results</div>
  <input id="searchInput" class="search-bar" placeholder="🔍 Search by query or answer..." oninput="filterTable()">

  <div class="table-wrap">
    <table id="resultsTable">
      <thead>
        <tr>
          <th>#</th>
          <th>Query &amp; Gold Docs</th>
          <th>k</th>
          <th>Generated Answer</th>
          <th>Gold Answer</th>
          <th>NDCG@10</th>
          <th>Recall@5</th>
          <th>CtxPrec</th>
          <th>Faith</th>
          <th>AnsRel</th>
          <th>Ret / Gen / Eval</th>
        </tr>
      </thead>
      <tbody id="tableBody">
        {rows_html}
      </tbody>
    </table>
  </div>

</div>

<footer>
  HotpotQA Pooled-Index Evaluation · {ts} · {total} queries · Ragas v{_ragas_version()}
</footer>

<script>
  function filterTable() {{
    const q = document.getElementById('searchInput').value.toLowerCase();
    document.querySelectorAll('#tableBody tr').forEach(row => {{
      row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
    }});
  }}
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
    print(f"[REPORT] Saved -> {REPORT_PATH}")


def _ragas_version():
    try:
        import ragas
        return ragas.__version__
    except Exception:
        return "?"


if __name__ == "__main__":
    asyncio.run(main())
