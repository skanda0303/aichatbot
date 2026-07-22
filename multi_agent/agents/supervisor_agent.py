"""
agents/supervisor_agent.py — Supervisor / Orchestrator Agent.

Coordinates the multi-agent pipeline:
  1. Receive user query + conversation history
  2. Call RAG Agent  → RAGResult
  3. Call Evaluation Agent → EvalResult
  4. [Conditional] Call Composio Agent → ComposioResult (if query needs external tools)
  5. [Conditional] Call Web Agent → WebResult  (only if sufficient == False)
  6. Call Answer Agent → stream tokens to caller

The Supervisor does NOT decide RAG-vs-Web upfront — it always runs RAG first
and delegates the routing decision to the Evaluation Agent.
Composio tools are invoked when the query explicitly mentions external services.

Exposes:
  run_streaming()  — async generator yielding final answer tokens (for SSE)
  run()            — coroutine returning the full answer string (for testing)
"""

from __future__ import annotations
from collections.abc import AsyncGenerator

from langchain_core.documents import Document

from multi_agent.agents import rag_agent, evaluation_agent, web_agent, answer_agent, query_rewriter_agent
from multi_agent.evaluation.routing_logger import log_routing_decision
from multi_agent.models.schemas import RAGResult, EvalResult, WebResult, ComposioResult
from multi_agent.retrieval.retriever import RerankedRetriever


# def _needs_composio(query: str) -> bool:
#     """Detect if the query likely needs external Composio tools."""
#     query_lower = query.lower()
#     composio_keywords = [
#         "github", "git hub", "repository", "repo", "pull request", "pr ", "issue", "commit",
#         "google doc", "google docs", "gdoc", "docs.google.com",
#         "youtube", "youtu.be", "video", "transcript",
#         "hugging face", "huggingface", "hf.co", "model hub", "dataset hub",
#         "context7", "context 7", "mcp", "library docs", "package docs",
#         "tavily", "web search", "search the web",
#     ]
#     return any(keyword in query_lower for keyword in composio_keywords)


# Called in: multi_agent/api.py (chat)
async def run_streaming(
    query: str,
    history_messages: list,
    retriever: RerankedRetriever,
    chunks: list[Document],
    user_gemini_key: str | None = None,
    user_tavily_key: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Full pipeline as an async generator yielding answer tokens.

    Execution flow:
      Query → RAG Agent → Evaluation Agent → [Composio Agent] → [Web Agent] → Answer Agent → tokens
    """
    # ── Step 0: Query Rewriting (Web only) ────────────────────────────────────
    print(f"\n[SUPERVISOR] Original Query: '{query}'")
    rewrites = await query_rewriter_agent.run(query, history_messages, user_gemini_key)
    web_query = rewrites["web_query"]

    # ── Step 1: RAG Retrieval ─────────────────────────────────────────────────
    print(f"[SUPERVISOR] -> Invoking RAG Agent with original query: '{query}'...")
    rag_result: RAGResult = rag_agent.run(query, retriever, chunks)

    # ── Step 2: Context Evaluation ────────────────────────────────────────────
    print("[SUPERVISOR] -> Invoking Evaluation Agent...")
    eval_result: EvalResult = evaluation_agent.run(query, rag_result, user_gemini_key)

    # ── Step 3: Conditional Composio Tools ────────────────────────────────────
    composio_result: ComposioResult | None = None

    # ── Step 4: Conditional Web Search ────────────────────────────────────────
    web_result: WebResult | None = None
    if not eval_result.sufficient:
        print(
            f"[SUPERVISOR] RAG insufficient (confidence={eval_result.confidence:.2f}). "
            "-> Invoking Web Agent..."
        )
        web_result = web_agent.run(web_query, user_tavily_key)
        route = "rag+web"
    else:
        print(
            f"[SUPERVISOR] RAG sufficient (confidence={eval_result.confidence:.2f}). "
            "-> Skipping Web Agent."
        )
        route = "rag_only"

    # Log routing decision
    log_routing_decision(
        query=query,
        eval_result=eval_result,
        route=route,
        rag_chunks=len(rag_result.retrieved_chunks),
        web_pages=len(web_result.web_context) if web_result else 0,
        composio_tools=len(composio_result.tool_names) if composio_result else 0,
    )

    # ── Step 5: Answer Generation (streaming) ─────────────────────────────────
    print("[SUPERVISOR] -> Invoking Answer Agent (streaming)...")
    async for token in answer_agent.stream(
        query=query,
        history_messages=history_messages,
        rag_result=rag_result,
        eval_result=eval_result,
        web_result=web_result,
        composio_result=composio_result,
        user_gemini_key=user_gemini_key,
    ):
        yield token


# Unused in production
async def run(
    query: str,
    history_messages: list,
    retriever: RerankedRetriever,
    chunks: list[Document],
    user_gemini_key: str | None = None,
    user_tavily_key: str | None = None,
) -> str:
    """
    Blocking version — collects the full answer string.
    Useful for testing and evaluation scripts.
    """
    # ── Step 0: Query Rewriting (Web only) ────────────────────────────────────
    print(f"\n[SUPERVISOR] Original Query: '{query}'")
    rewrites = await query_rewriter_agent.run(query, history_messages, user_gemini_key)
    web_query = rewrites["web_query"]

    # ── Step 1: RAG Retrieval ─────────────────────────────────────────────────
    print(f"[SUPERVISOR] -> Invoking RAG Agent with original query: '{query}'...")
    rag_result: RAGResult = rag_agent.run(query, retriever, chunks)

    # ── Step 2: Context Evaluation ────────────────────────────────────────────
    print("[SUPERVISOR] -> Invoking Evaluation Agent...")
    eval_result: EvalResult = evaluation_agent.run(query, rag_result, user_gemini_key)

    # ── Step 3: Conditional Composio Tools ────────────────────────────────────
    composio_result: ComposioResult | None = None

    # ── Step 4: Conditional Web Search ────────────────────────────────────────
    web_result: WebResult | None = None
    if not eval_result.sufficient:
        print("[SUPERVISOR] RAG insufficient -> Invoking Web Agent...")
        web_result = web_agent.run(web_query, user_tavily_key)
        route = "rag+web"
    else:
        print("[SUPERVISOR] RAG sufficient -> Skipping Web Agent.")
        route = "rag_only"

    log_routing_decision(
        query=query,
        eval_result=eval_result,
        route=route,
        rag_chunks=len(rag_result.retrieved_chunks),
        web_pages=len(web_result.web_context) if web_result else 0,
        composio_tools=len(composio_result.tool_names) if composio_result else 0,
    )

    # ── Step 5: Answer Generation (blocking) ──────────────────────────────────
    print("[SUPERVISOR] -> Invoking Answer Agent (blocking)...")
    return await answer_agent.run(
        query=query,
        history_messages=history_messages,
        rag_result=rag_result,
        eval_result=eval_result,
        web_result=web_result,
        composio_result=composio_result,
        user_gemini_key=user_gemini_key,
    )
