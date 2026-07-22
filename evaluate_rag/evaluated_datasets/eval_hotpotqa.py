"""
run_eval_hotpot.py — Evaluates the RAG pipeline on the HotpotQA dataset.
Tests multi-document reasoning by indexing the 10 context paragraphs per query
and retrieving supporting facts.
"""

import os
import time
import asyncio
import dotenv
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import load_dataset
from sentence_transformers import CrossEncoder

# Shared imports
from evaluate_rag.retriever import RerankedRetriever
from evaluate_rag.rag_pipeline import retrieve_chunks
from evaluate_rag.evaluated_datasets.common import (
    compute_recall,
    compute_context_precision,
    compute_ndcg,
    run_agent_generation,
    evaluate_generation_judge,
)

# Load environment — must point to project root .env
_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_project_root = os.path.abspath(os.path.join(_dir, ".."))
dotenv.load_dotenv(dotenv_path=os.path.join(_project_root, ".env"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
print(f"[INIT] Using API key: {GOOGLE_API_KEY[:20]}...")

# Initialize Models
GENERATOR_MODEL = "gemini-3.1-flash-lite"
JUDGE_MODEL = "gemini-3.1-flash-lite"

generator_llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2)
judge_llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0)

# Local Embedding Model (BGE-M3) and GPU Reranker
embeddings = OllamaEmbeddings(model="bge-m3")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")


async def main():
    print("[INIT] Loading HotpotQA validation split...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    # Select first 50 queries for evaluation
    eval_size = 50
    subset = dataset.select(range(eval_size))
    
    summary = {
        "ndcg_10": 0.0, "recall_5": 0, "ctx_prec": 0.0,
        "faith": 0.0, "ans_rel": 0.0, "total": 0,
        "t_ret_3": 0.0, "t_ret_5": 0.0,
        "t_gen_3": 0.0, "t_gen_5": 0.0,
        "t_eval_3": 0.0, "t_eval_5": 0.0,
        "t_tot_3": 0.0, "t_tot_5": 0.0,
    }
    query_results = []

    print(f"\nStarting HotpotQA Evaluation ({eval_size} queries)...\n" + "="*80)

    for idx, example in enumerate(subset, start=1):
        query_text = example["question"]
        gold_answer = example["answer"]
        gold_titles = list(set(example["supporting_facts"]["title"]))
        
        print(f"\n[{idx}/{eval_size}] Query: {query_text!r}")
        print(f"         Supporting docs: {gold_titles}")

        # Format context paragraphs
        ctx_data = example["context"]
        docs_to_index = []
        for title, sentences in zip(ctx_data["title"], ctx_data["sentences"]):
            paragraph = " ".join(sentences)
            docs_to_index.append(Document(page_content=paragraph, metadata={"source": title}))

        # Temporary in-memory Chroma database for this query's context
        vectorstore = Chroma.from_documents(
            documents=docs_to_index,
            embedding=embeddings,
            collection_name="temp_hotpot_eval"
        )

        # Build retrievers
        bm25_retriever = BM25Retriever.from_documents(docs_to_index)
        bm25_retriever.k = 10
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7]
        )
        retriever = RerankedRetriever(ensemble_retriever, reranker_model_name="BAAI/bge-reranker-v2-m3", top_n=10)

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

            # Evaluation
            t0 = time.perf_counter()
            ragas_scores = await evaluate_generation_judge(query_text, rag_context, gen_ans, judge_llm)
            dt_eval = time.perf_counter() - t0

            dt_tot = time.perf_counter() - t_query_start

            per_k[k] = {
                "ndcg_10": ndcg_val,
                "recall_5": recall_5,
                "ctx_prec": ctx_prec,
                "faithfulness": ragas_scores["faithfulness"],
                "answer_relevancy": ragas_scores["answer_relevancy"],
                "reasoning": ragas_scores.get("reasoning", "")
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

        # Clear vectorstore and collection to free memory
        vectorstore.delete_collection()
        
        # Rate-limiting sleep
        if idx < eval_size:
            await asyncio.sleep(4)

    # Output stats
    pt = summary["total"] or 1
    print("\n" + "="*80)
    print("FINAL HOTPOTQA SUMMARY")
    print("="*80)
    print(f"  NDCG@10           : {summary['ndcg_10']/pt:.3f}")
    print(f"  Recall@5          : {summary['recall_5']/pt*100:.1f}%")
    print(f"  Context Precision : {summary['ctx_prec']/(2*pt):.3f}")
    print(f"  Faithfulness      : {summary['faith']/(2*pt):.3f}")
    print(f"  Answer Relevancy  : {summary['ans_rel']/(2*pt):.3f}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
