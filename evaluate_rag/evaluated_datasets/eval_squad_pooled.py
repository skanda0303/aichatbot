"""
run_eval_squad_pooled.py — Evaluates the RAG pipeline on the SQuAD v1.1 dataset.
"""

import os
import sys
import time
import html
import asyncio
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

REPORT_PATH = os.path.join(_dir, "eval_report_squad_pooled.html")

# Initialize Models
GENERATOR_MODEL = "gemini-3.1-flash-lite"

generator_llm = ChatGoogleGenerativeAI(
    model=GENERATOR_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2
)

# Local Embedding Model & GPU Reranker
embeddings = OllamaEmbeddings(model="bge-m3")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# Ragas Wrappers — use local embeddings to save API quota
ragas_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(model=GENERATOR_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0)
)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# Instantiate Ragas metrics
faithfulness_metric = Faithfulness(llm=ragas_llm)
answer_relevancy_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1)


# ── Main Evaluator ────────────────────────────────────────────────────────────

async def main():
    print("[INIT] Loading SQuAD v1.1 validation split...")
    dataset = load_dataset("rajpurkar/squad", split="validation")

    # 50 queries × 2 k-values × 4 API calls = 400 total (within 450 limit)
    eval_size = 50
    subset = dataset.select(range(eval_size))

    # ── Step 1: Build Shared Pool of Context Paragraphs ───────────────────────
    print("[INIT] Extracting unique Wikipedia paragraphs for the shared index...")
    unique_paragraphs = {}  # title -> paragraph
    qa_examples = []        # store processed examples for evaluation loop

    for example in subset:
        title   = example["title"]
        context = example["context"]
        question = example["question"]
        answers = example["answers"]["text"]
        gold_answer = answers[0] if answers else ""

        if title not in unique_paragraphs:
            unique_paragraphs[title] = context

        qa_examples.append({
            "question":    question,
            "gold_title":  title,
            "gold_answer": gold_answer,
            "gold_context": context,
        })

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
        collection_name="temp_squad_pooled"
    )
    print(f"[SUCCESS] Indexed {len(docs_to_index)} docs in {time.perf_counter() - t_idx:.2f} seconds.")

    # ── Step 3: Build Hybrid Retriever ────────────────────────────────────────
    bm25_retriever = BM25Retriever.from_documents(docs_to_index, k=10)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6])
    retriever = RerankedRetriever(ensemble, reranker_model_name="BAAI/bge-reranker-v2-m3", top_n=10)

    # ── Step 4: Evaluation Loop ───────────────────────────────────────────────
    summary = {
        "ndcg_10": 0.0, "recall_5": 0.0, "ctx_prec": 0.0,
        "faith": 0.0,   "ans_rel": 0.0,  "total": 0,
        "t_ret_3": 0.0, "t_ret_5": 0.0,
        "t_gen_3": 0.0, "t_gen_5": 0.0,
        "t_eval_3": 0.0, "t_eval_5": 0.0,
        "t_tot_3": 0.0,  "t_tot_5": 0.0,
    }
    query_results = []

    print(f"\nStarting SQuAD Shared-Index Evaluation ({eval_size} queries)...")
    print("=" * 80)

    for idx, example in enumerate(qa_examples, start=1):
        query_text  = example["question"]
        gold_title  = example["gold_title"]
        gold_answer = example["gold_answer"]

        print(f"\n[{idx}/{eval_size}] Query: {query_text!r}")
        print(f"         Gold title: '{gold_title}' | Gold answer: '{gold_answer[:60]}...' ")

        t_query_start = time.perf_counter()
        summary["total"] += 1

        per_k = {}
        generated_ans_cache = {}
        latencies_query = {}
        ndcg_val = 0.0

        for k in [3, 5]:
            t0 = time.perf_counter()
            pipeline = retrieve_chunks(query=query_text, retriever=retriever, all_chunks=docs_to_index, top_k=k)
            final_docs = pipeline["retrieved_final"]
            unique = pipeline["retrieved_unique"]
            rag_context = pipeline["rag_context"]
            dt_ret = pipeline["retrieval_time"]

            # NDCG evaluated on full candidate pool (computed once at k=3)
            if k == 3:
                ndcg_val = compute_ndcg(unique, gold_title, k=10)

            recall_5  = compute_recall(final_docs, gold_title)
            ctx_prec  = compute_context_precision(final_docs, gold_title)

            # Generation
            t0 = time.perf_counter()
            gen_ans = await run_agent_generation(query_text, rag_context, generator_llm)
            dt_gen = time.perf_counter() - t0
            generated_ans_cache[k] = gen_ans

            # Ragas evaluation
            t0 = time.perf_counter()
            ragas_scores = await evaluate_with_ragas(
                query_text, rag_context, gen_ans, faithfulness_metric, answer_relevancy_metric
            )
            dt_eval = time.perf_counter() - t0

            dt_tot = time.perf_counter() - t_query_start

            per_k[k] = {
                "ndcg_10":         ndcg_val,
                "recall_5":        recall_5,
                "ctx_prec":        ctx_prec,
                "faithfulness":    ragas_scores["faithfulness"],
                "answer_relevancy": ragas_scores["answer_relevancy"],
            }
            latencies_query[k] = {"ret": dt_ret, "gen": dt_gen, "eval": dt_eval, "tot": dt_tot}

            print(
                f"  [k={k}] NDCG@10={ndcg_val:.3f} | Recall@5={recall_5:.3f} | "
                f"CtxPrec={ctx_prec:.3f} | Faith={ragas_scores['faithfulness']:.2f} | "
                f"AnsRel={ragas_scores['answer_relevancy']:.2f} | "
                f"Ret={dt_ret:.2f}s Gen={dt_gen:.2f}s"
            )

            # Accumulate
            if k == 3:
                summary["ndcg_10"] += ndcg_val
            summary["recall_5"]  += recall_5
            summary["ctx_prec"]  += ctx_prec
            summary["faith"]     += ragas_scores["faithfulness"]
            summary["ans_rel"]   += ragas_scores["answer_relevancy"]
            summary[f"t_ret_{k}"]  += dt_ret
            summary[f"t_gen_{k}"]  += dt_gen
            summary[f"t_eval_{k}"] += dt_eval
            summary[f"t_tot_{k}"]  += dt_tot

        query_results.append({
            "idx":          idx,
            "query":        query_text,
            "gold_title":   gold_title,
            "gold_answer":  gold_answer,
            "gen_ans_3":    generated_ans_cache.get(3, ""),
            "gen_ans_5":    generated_ans_cache.get(5, ""),
            "per_k":        per_k,
            "latencies":    latencies_query,
            "t_ret_3":  latencies_query[3]["ret"],
            "t_gen_3":  latencies_query[3]["gen"],
            "t_ret_5":  latencies_query[5]["ret"],
            "t_gen_5":  latencies_query[5]["gen"],
            "t_eval_3": latencies_query[3]["eval"],
            "t_eval_5": latencies_query[5]["eval"],
            "t_tot_3":  latencies_query[3]["tot"],
            "t_tot_5":  latencies_query[5]["tot"],
        })

        if idx < eval_size:
            await asyncio.sleep(4)  # Respect free-tier rate limits

    # Clean up
    vectorstore.delete_collection()

    # Final stats
    pt = summary["total"] or 1
    n2 = 2 * pt
    avg_ndcg    = summary["ndcg_10"] / pt
    avg_recall  = summary["recall_5"] / n2
    avg_ctxprec = summary["ctx_prec"] / n2
    avg_faith   = summary["faith"]    / n2
    avg_rel     = summary["ans_rel"]  / n2

    print("\n" + "=" * 80)
    print("FINAL SQUAD POOLED-INDEX SUMMARY")
    print("=" * 80)
    print(f"  NDCG@10           : {avg_ndcg:.3f}")
    print(f"  Recall@5          : {avg_recall * 100:.1f}%")
    print(f"  Context Precision : {avg_ctxprec:.3f}")
    print(f"  Faithfulness      : {avg_faith:.3f}")
    print(f"  Answer Relevancy  : {avg_rel:.3f}")
    print("=" * 80)

    save_html_report(query_results, avg_ndcg, avg_recall, avg_ctxprec, avg_faith, avg_rel)
    print(f"\n[REPORT] Open: file:///{REPORT_PATH.replace(os.sep, '/')}")

    sys.stderr = open(os.devnull, 'w')


# ── HTML Report ───────────────────────────────────────────────────────────────

def save_html_report(results, avg_ndcg, avg_recall, avg_ctxprec, avg_faith, avg_rel):
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(results)

    def score_class(v):
        if v >= 0.75: return "score-high"
        if v >= 0.5:  return "score-mid"
        return "score-low"

    def bar(v):
        pct = int(v * 100)
        return f'<div class="bar-wrap"><div class="bar" style="width:{pct}%"></div><span>{v:.3f}</span></div>'

    def block(r):
        return f"""
        <div class="query-block">
          <div class="query-header">
            <span class="q-num">#{r['idx']}</span>
            <span class="q-text">{html.escape(r['query'])}</span>
          </div>
          <div class="gold-row">
            <span class="gold-label">Gold Title</span>
            <span class="gold-val">{html.escape(r['gold_title'])}</span>
            <span class="gold-label" style="margin-left:1.5rem">Gold Answer</span>
            <span class="gold-val">{html.escape(r['gold_answer'])}</span>
          </div>
          <table class="metric-table">
            <thead>
              <tr>
                <th>k</th><th>Generated Answer</th>
                <th>NDCG@10</th><th>Recall@5</th><th>CtxPrec</th>
                <th>Faith</th><th>AnsRel</th><th>Ret / Gen</th>
              </tr>
            </thead>
            <tbody>
              {"".join(f'''
              <tr class="k-row">
                <td class="k-badge">k={k}</td>
                <td class="answer-cell">{html.escape(str(r.get(f"gen_ans_{k}", "")))}</td>
                <td class="{score_class(r["per_k"][k]["ndcg_10"])}">{r["per_k"][k]["ndcg_10"]:.3f}</td>
                <td class="{score_class(r["per_k"][k]["recall_5"])}">{r["per_k"][k]["recall_5"]:.3f}</td>
                <td class="{score_class(r["per_k"][k]["ctx_prec"])}">{r["per_k"][k]["ctx_prec"]:.3f}</td>
                <td class="{score_class(r["per_k"][k]["faithfulness"])}">{r["per_k"][k]["faithfulness"]:.3f}</td>
                <td class="{score_class(r["per_k"][k]["answer_relevancy"])}">{r["per_k"][k]["answer_relevancy"]:.3f}</td>
                <td class="lat">{r["latencies"][k]["ret"]:.2f}s / {r["latencies"][k]["gen"]:.2f}s</td>
              </tr>''' for k in [3, 5] if k in r["per_k"])}
            </tbody>
          </table>
        </div>"""

    all_blocks = "".join(block(r) for r in results)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SQuAD Pooled Eval Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0d0f1a; --surface: #12152b; --surface2: #1a1e38;
      --border: #2e3258; --text: #e2e8f0; --text-muted: #94a3b8;
      --text-dim: #64748b; --accent: #06b6d4; --accent2: #3b82f6;
      --green: #4ade80; --red: #f87171; --yellow: #fbbf24;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }}
    .header {{ background: linear-gradient(135deg, #111827 0%, #0d0f1a 100%); border-bottom: 1px solid var(--border); padding: 2.5rem 3rem; }}
    .header h1 {{ font-size: 1.7rem; font-weight: 700; }}
    .header h1 span {{ background: linear-gradient(90deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .meta {{ color: var(--text-dim); font-size: 0.85rem; margin-top: 0.5rem; }}
    .pills {{ display: flex; gap: 0.6rem; margin-top: 1rem; flex-wrap: wrap; }}
    .pill {{ padding: 0.3rem 0.8rem; border-radius: 999px; font-size: 0.78rem; font-weight: 500; }}
    .pill-blue   {{ background: #1e40af33; color: #60a5fa; border: 1px solid #1e40af55; }}
    .pill-cyan   {{ background: #0e7490aa33; color: #22d3ee; border: 1px solid #0e749055; }}
    .pill-green  {{ background: #15803d33; color: #4ade80; border: 1px solid #15803d55; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 2.5rem 3rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2.5rem; }}
    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.4rem; position: relative; overflow: hidden; }}
    .card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }}
    .card-label {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim); margin-bottom: 0.5rem; }}
    .card-value {{ font-size: 2rem; font-weight: 700; background: linear-gradient(90deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .card-sub {{ font-size: 0.78rem; color: var(--text-muted); margin-top: 0.3rem; }}
    .chart-section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; }}
    .chart-title {{ font-size: 0.85rem; font-weight: 600; color: var(--text-muted); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .bar-wrap {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.7rem; }}
    .bar-wrap span {{ font-size: 0.8rem; font-weight: 600; min-width: 3rem; }}
    .bar {{ height: 10px; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width 0.4s; }}
    .bar-label {{ font-size: 0.8rem; color: var(--text-muted); min-width: 9rem; }}
    .bar-row {{ display: grid; grid-template-columns: 10rem 1fr; align-items: center; gap: 1rem; margin-bottom: 0.6rem; }}
    .section-title {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 1.2rem; color: var(--text); }}
    .query-block {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 1.2rem; overflow: hidden; transition: box-shadow 0.2s; }}
    .query-block:hover {{ box-shadow: 0 4px 20px rgba(6, 182, 212, 0.1); }}
    .query-header {{ display: flex; align-items: flex-start; gap: 0.75rem; padding: 1rem 1.2rem 0.7rem; }}
    .q-num {{ background: linear-gradient(135deg, var(--accent), var(--accent2)); color: white; font-size: 0.72rem; font-weight: 700; padding: 0.2rem 0.55rem; border-radius: 999px; flex-shrink: 0; margin-top: 2px; }}
    .q-text {{ font-weight: 500; font-size: 0.9rem; line-height: 1.4; }}
    .gold-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 0.5rem; padding: 0.4rem 1.2rem 0.7rem; font-size: 0.78rem; }}
    .gold-label {{ color: var(--text-dim); font-weight: 600; text-transform: uppercase; font-size: 0.7rem; }}
    .gold-val {{ color: var(--accent); font-weight: 500; }}
    .metric-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
    .metric-table thead {{ background: var(--surface2); }}
    .metric-table th {{ padding: 0.5rem 0.8rem; text-align: left; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-dim); }}
    .metric-table td {{ padding: 0.5rem 0.8rem; border-top: 1px solid var(--border); vertical-align: top; }}
    .k-badge {{ font-weight: 700; color: var(--accent); font-size: 0.75rem; white-space: nowrap; }}
    .answer-cell {{ color: var(--text-muted); max-width: 280px; font-size: 0.79rem; line-height: 1.4; }}
    .score-high {{ color: var(--green); font-weight: 700; }}
    .score-mid  {{ color: var(--yellow); font-weight: 700; }}
    .score-low  {{ color: var(--red); font-weight: 700; }}
    .lat {{ color: var(--text-dim); font-size: 0.76rem; white-space: nowrap; }}
    .search-bar {{ display: flex; gap: 0.8rem; margin-bottom: 1.5rem; }}
    .search-bar input {{ flex: 1; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.6rem 1rem; color: var(--text); font-size: 0.85rem; outline: none; font-family: inherit; }}
    .search-bar input:focus {{ border-color: var(--accent); }}
    footer {{ text-align: center; padding: 2rem; color: var(--text-dim); font-size: 0.78rem; border-top: 1px solid var(--border); margin-top: 2rem; }}
  </style>
</head>
<body>
<div class="header">
  <h1>SQuAD v1.1 <span>Pooled-Index Evaluation</span></h1>
  <p class="meta">Generated: {ts} &nbsp;|&nbsp; {total} queries</p>
  <div class="pills">
    <span class="pill pill-blue">Generator: {GENERATOR_MODEL}</span>
    <span class="pill pill-cyan">Dataset: SQuAD v1.1 (Wikipedia Q&amp;A)</span>
    <span class="pill pill-blue">Ragas Judge: {GENERATOR_MODEL}</span>
    <span class="pill pill-green">Retriever: BM25 + BGE-M3 + CrossEncoder Reranker</span>
    <span class="pill pill-green">Embeddings: bge-m3 (Local Ollama)</span>
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
      <div class="card-value">{avg_recall * 100:.1f}%</div>
      <div class="card-sub">Gold article found in top 5</div>
    </div>
    <div class="card">
      <div class="card-label">Context Precision</div>
      <div class="card-value">{avg_ctxprec:.3f}</div>
      <div class="card-sub">MAP-style relevance of retrieved docs</div>
    </div>
    <div class="card">
      <div class="card-label">Faithfulness</div>
      <div class="card-value">{avg_faith:.3f}</div>
      <div class="card-sub">Ragas — hallucination score</div>
    </div>
    <div class="card">
      <div class="card-label">Answer Relevancy</div>
      <div class="card-value">{avg_rel:.3f}</div>
      <div class="card-sub">Ragas — semantic relevance</div>
    </div>
  </div>

  <!-- Metric Bar Chart -->
  <div class="chart-section">
    <div class="chart-title">Metric Overview</div>
    <div class="bar-row"><span class="bar-label">NDCG@10</span>{bar(avg_ndcg)}</div>
    <div class="bar-row"><span class="bar-label">Recall@5</span>{bar(avg_recall)}</div>
    <div class="bar-row"><span class="bar-label">Context Precision</span>{bar(avg_ctxprec)}</div>
    <div class="bar-row"><span class="bar-label">Faithfulness</span>{bar(avg_faith)}</div>
    <div class="bar-row"><span class="bar-label">Answer Relevancy</span>{bar(avg_rel)}</div>
  </div>

  <!-- Per-Query Results -->
  <div class="section-title">Per-Query Results ({total} queries)</div>
  <div class="search-bar">
    <input type="text" id="search" placeholder="Search queries, answers or titles..." oninput="filterBlocks(this.value)">
  </div>
  <div id="query-list">
    {all_blocks}
  </div>
</div>

<footer>
  SQuAD v1.1 Evaluation &nbsp;|&nbsp; {ts} &nbsp;|&nbsp; Ragas v{_ragas_version()} &nbsp;|&nbsp; BGE-M3 Local Embeddings
</footer>

<script>
  function filterBlocks(q) {{
    q = q.toLowerCase();
    document.querySelectorAll('.query-block').forEach(b => {{
      b.style.display = b.innerText.toLowerCase().includes(q) ? '' : 'none';
    }});
  }}
  document.querySelectorAll('.metric-table td').forEach(cell => {{
    const v = parseFloat(cell.innerText);
    if (!isNaN(v) && v <= 1.0 && cell.innerText.length < 8) {{
      if (v >= 0.75) cell.classList.add('score-high');
      else if (v >= 0.5) cell.classList.add('score-mid');
      else cell.classList.add('score-low');
    }}
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
        return "unknown"


if __name__ == "__main__":
    asyncio.run(main())
