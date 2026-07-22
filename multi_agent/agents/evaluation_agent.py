"""
agents/evaluation_agent.py — Context Evaluation Agent.

Evaluates whether the retrieved RAG chunks are sufficient to answer the
user's query.  This agent replaces the old heuristic "does this need web
search?" prompt with a dedicated reasoning layer.

Rules:
  - No answer generation
  - No web search
  - Returns EvalResult: { sufficient, confidence, reason }
"""

import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from multi_agent.models.schemas import RAGResult, EvalResult
from multi_agent.config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from multi_agent.utils.helpers import format_chunks_for_prompt

_llm = None

_SYSTEM_PROMPT = """\
You are a context evaluation specialist. Your ONLY job is to determine whether \
the provided retrieved document chunks contain the direct answer or relevant facts for the user's question.

You must output a JSON object with exactly these three fields:
  "sufficient"  : boolean — true if the chunks contain the answer or direct facts for the question
  "confidence"  : float between 0.0 and 1.0
  "reason"      : one concise sentence explaining your decision

Evaluation criteria:
  - If the retrieved chunks contain a direct fact, table row, or explicit answer (e.g. "Challenge: Weather"), mark sufficient = true.
  - Do NOT mark context as insufficient simply because the answer is brief, concise, or tabular. If the document states the fact, it IS sufficient.
  - Mark sufficient = false ONLY if the chunks have zero relevant information or completely miss the subject of the user's question.

Do NOT generate an answer. Output ONLY the JSON object, nothing else.
"""


# Called in: multi_agent/agents/evaluation_agent.py (run)
def _parse_eval_response(raw: str) -> EvalResult:
    """Parse the LLM's JSON response into an EvalResult, with a safe fallback."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    # Extract the first JSON object
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return EvalResult(
                sufficient=bool(data.get("sufficient", False)),
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", "No reason provided.")),
            )
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: if we can't parse, assume insufficient (prefer web search on failure)
    print(f"[EVAL AGENT] Failed to parse LLM response: {raw!r}")
    return EvalResult(
        sufficient=False,
        confidence=0.0,
        reason="Could not parse evaluation response; defaulting to insufficient.",
    )


# Called in: multi_agent/agents/supervisor_agent.py (run_streaming, run)
def run(query: str, rag_result: RAGResult, user_gemini_key: str | None = None) -> EvalResult:
    """
    Evaluate retrieval quality and return a sufficiency verdict.

    Input:
      query      — original user question
      rag_result — output from rag_agent.run()

    Output:
      EvalResult — { sufficient, confidence, reason }
    """
    key = user_gemini_key or GOOGLE_API_KEY
    if not key:
        print("[EVAL AGENT] Gemini API Key is missing — returning insufficient.")
        return EvalResult(
            sufficient=False,
            confidence=0.0,
            reason="Gemini API Key is missing. Please enter it in the credentials input on the sidebar to proceed.",
        )

    if not rag_result.retrieved_chunks:
        print("[EVAL AGENT] No chunks to evaluate — insufficient.")
        return EvalResult(
            sufficient=False,
            confidence=0.0,
            reason="No documents were retrieved from the knowledge base.",
        )

    # Format evaluation prompt
    chunks_preview = format_chunks_for_prompt(rag_result.retrieved_chunks, max_chunks=6)
    scores_summary = (
        f"Average CrossEncoder score: {rag_result.avg_retrieval_score:.4f}\n"
        f"Top-3 scores: {[round(s, 4) for s in rag_result.cross_encoder_scores[:3]]}"
    )

    user_message = (
        f"User Question:\n{query}\n\n"
        f"Retrieval Scores:\n{scores_summary}\n\n"
        f"Retrieved Chunks ({len(rag_result.retrieved_chunks)} total):\n\n"
        f"{chunks_preview}\n\n"
        "Evaluate whether these chunks are sufficient to answer the question."
    )

    from datetime import datetime
    current_date_str = datetime.now().strftime('%A, %B %d, %Y')

    dynamic_prompt = (
        "You are a context evaluation specialist. Your ONLY job is to determine whether "
        "the provided retrieved document chunks contain the direct answer or relevant facts for the user's question.\n"
        f"Current date is {current_date_str}.\n\n"
        "You must output a JSON object with exactly these three fields:\n"
        "  \"sufficient\"  : boolean — true if the chunks contain the answer or direct facts for the question\n"
        "  \"confidence\"  : float between 0.0 and 1.0\n"
        "  \"reason\"      : one concise sentence explaining your decision\n\n"
        "Evaluation criteria:\n"
        "  - If the retrieved chunks contain a direct fact, table row, or explicit answer (e.g., 'Challenge: Weather'), mark sufficient = true.\n"
        "  - Do NOT mark context as insufficient simply because the answer is brief, concise, or tabular. If the document states the fact, it IS sufficient.\n"
        "  - Mark sufficient = false ONLY if the chunks have zero relevant information or completely miss the subject of the user's question.\n\n"
        "Do NOT generate an answer. Output ONLY the JSON object, nothing else."
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=key,
            temperature=LLM_TEMPERATURE,
        )
        response = llm.invoke(
            [
                SystemMessage(content=dynamic_prompt),
                HumanMessage(content=user_message),
            ]
        )
        raw = response.content
        if isinstance(raw, list):
            raw = " ".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in raw
            )

        result = _parse_eval_response(str(raw))
        print(
            f"[EVAL AGENT] sufficient={result.sufficient} | "
            f"confidence={result.confidence:.2f} | reason={result.reason}"
        )
        return result

    except Exception as e:
        print(f"[EVAL AGENT] Error during evaluation: {e}")
        return EvalResult(
            sufficient=False,
            confidence=0.0,
            reason=f"Evaluation failed with error: {e}",
        )
