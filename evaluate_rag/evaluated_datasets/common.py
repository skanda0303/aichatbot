"""
evaluate_rag/evaluated_datasets/common.py — Shared functions and metrics for dataset evaluations.
"""

import re
import json
import math
import time
import asyncio
import concurrent.futures
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from datasets import Dataset as HFDataset
from ragas import evaluate

# ── General Helpers ──────────────────────────────────────────────────────────

def is_refusal(text: str) -> bool:
    """Check if the generated response is a refusal/abstention."""
    t = text.lower()
    return any(p in t for p in [
        "cannot answer", "does not contain", "no information",
        "not mentioned", "not discussed", "not provide information",
        "i do not know", "i am sorry", "insufficient context",
        "cannot be answered", "is not mentioned in", "cannot answer from",
        "insufficient table data"
    ])


def _extract_text(content) -> str:
    """Safely extract plain text from LLM response content, ignoring thinking blocks."""
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
    """Extract and parse the first JSON object from a string."""
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


# ── BEIR / Retrieval Metrics ───────────────────────────────────────────────────

def compute_recall(docs: list[Document], gold_titles: list[str] | str) -> float:
    """Fractional/binary recall: 1.0 if any gold title matches retrieved docs' sources, else 0.0."""
    golds = [gold_titles] if isinstance(gold_titles, str) else gold_titles
    if not golds:
        return 0.0
    for doc in docs:
        if doc.metadata.get("source") in golds:
            return 1.0
    return 0.0


def compute_context_precision(docs: list[Document], gold_titles: list[str] | str) -> float:
    """MAP-style Context Precision using the gold titles as relevance signals."""
    golds = [gold_titles] if isinstance(gold_titles, str) else gold_titles
    if not docs or not golds:
        return 0.0
    relevant, precision_sum = 0, 0.0
    for i, doc in enumerate(docs, start=1):
        if doc.metadata.get("source") in golds:
            relevant += 1
            precision_sum += relevant / i
    return precision_sum / relevant if relevant else 0.0


def compute_ndcg(docs: list[Document], gold_titles: list[str] | str, k: int = 10) -> float:
    """Binary NDCG@k: relevant if doc's source matches any gold title."""
    golds = [gold_titles] if isinstance(gold_titles, str) else gold_titles
    if not golds:
        return 0.0
    dcg = 0.0
    for i, doc in enumerate(docs[:k], start=1):
        if doc.metadata.get("source") in golds:
            dcg += 1.0 / math.log2(i + 1)
    idcg = sum(1.0 / math.log2(j + 1) for j in range(1, min(len(golds), k) + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


# ── LLM Generation & Judges ───────────────────────────────────────────────────

async def run_agent_generation(query: str, context: str, generator_llm, system_prompt: str = None) -> str:
    """Generate answer using the generator LLM with the provided RAG context."""
    if system_prompt is None:
        system_prompt = (
            "You are a precise, fact-grounded assistant.\n"
            "Answer the question directly based on facts from the RAG context.\n"
            "If you cannot answer based on the context, state that you cannot answer."
        )
    user_input = f"RAG context:\n{context}\n\nQuestion: {query}"
    try:
        response = await generator_llm.ainvoke([
            HumanMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ])
        return _extract_text(response.content).strip()
    except Exception as e:
        print(f"\n[DEBUG GENERATION ERROR] {e}\n")
        return f"Error: {e}"


async def evaluate_with_ragas(
    query: str, context: str, gen_ans: str, faithfulness_metric, answer_relevancy_metric
) -> dict:
    """
    Evaluate using Ragas metrics. Runs in a separate threadpool to prevent
    async loop conflict issues.
    """
    def _run():
        ragas_data = HFDataset.from_dict({
            "question": [query],
            "answer":   [gen_ans],
            "contexts": [[context]],
        })
        result = evaluate(ragas_data, metrics=[faithfulness_metric, answer_relevancy_metric])
        scores = result.to_pandas().iloc[0]
        return {
            "faithfulness":     max(0.0, min(1.0, float(scores["faithfulness"]))),
            "answer_relevancy": max(0.0, min(1.0, float(scores["answer_relevancy"]))),
        }

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, _run)


async def evaluate_generation_judge(
    query: str, context: str, gen_ans: str, judge_llm, reference_answer: str = None, subset: str = None
) -> dict:
    """
    LLM-based judge mimicking RAGAS metrics (Faithfulness + Answer Relevancy).
    Includes optional fast-path for correct absent-set/refusal abstentions.
    """
    if is_refusal(gen_ans):
        return {
            "faithfulness":       1.0,
            "answer_relevancy":   1.0 if subset == "absent" else 0.5,
            "reasoning":          "Model correctly abstained.",
        }

    context_snippet = (context[:3000] if context else "(empty -- no context)")
    
    prompt = (
        "You are an objective RAG evaluation judge. Score each metric 0.0 to 1.0.\n\n"
        f"QUESTION: {query}\n\n"
        f"RETRIEVED CONTEXT (first 3000 chars):\n{context_snippet}\n\n"
        f"GENERATED ANSWER: {gen_ans}\n\n"
    )
    if reference_answer:
        prompt += f"REFERENCE ANSWER: {reference_answer}\n\n"
        
    prompt += (
        "FAITHFULNESS: Are ALL claims in the generated answer directly supported by the RETRIEVED CONTEXT?\n"
        "  1.0 = every claim grounded | 0.5 = partially | 0.0 = unsupported or fabricated\n\n"
        "ANSWER_RELEVANCY: Does the generated answer directly address the original QUESTION?\n"
        "  1.0 = fully addresses | 0.5 = partially | 0.0 = off-topic or evasive\n\n"
        "Respond ONLY with this JSON (no markdown):\n"
        '{"faithfulness": 0.0, "answer_relevancy": 0.0, "reasoning": "one sentence"}'
    )

    try:
        response = await judge_llm.ainvoke([HumanMessage(content=prompt)])
        content  = _extract_text(response.content).strip()
        parsed   = _parse_llm_json(content)
        if parsed:
            return {
                "faithfulness":       max(0.0, min(1.0, float(parsed.get("faithfulness", 0.5)))),
                "answer_relevancy":   max(0.0, min(1.0, float(parsed.get("answer_relevancy", 0.5)))),
                "reasoning":          str(parsed.get("reasoning", "")),
            }
    except Exception as e:
        print("[JUDGE ERROR]", e)
        
    return {"faithfulness": 0.5, "answer_relevancy": 0.5, "reasoning": "Judge parse error."}
