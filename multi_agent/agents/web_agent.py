"""
agents/web_agent.py — Web Search Agent (Conditional Execution).

Only triggered when the Context Evaluation Agent returns sufficient=False.
Performs the same Tavily search + HTML fetch + clean pipeline as the original
ragbot/tools.py web_search tool.

Rules:
  - No answer generation
  - Returns WebResult: { web_context, source_urls, confidence }
"""

import time

from multi_agent.models.schemas import WebResult
from multi_agent.tools.web_tools import tavily_search, fetch_and_clean_results


# Called in: multi_agent/agents/supervisor_agent.py (run_streaming, run)
def run(query: str, user_tavily_key: str | None = None) -> WebResult:
    """
    Search the web for information relevant to the query.

    Pipeline:
      Tavily Search (same config as before)
        ↓
      Filter top URLs (video blacklist applied in web_tools)
        ↓
      Fetch HTML → clean text (same MAX_FETCH_CHARS as before)
        ↓
      Return structured WebResult

    Returns:
      WebResult — { web_context, source_urls, confidence }
    """
    print(f"[WEB AGENT] Searching web for: '{query}'")
    t_start = time.perf_counter()

    try:
        raw_results = tavily_search(query, user_tavily_key)
    except Exception as e:
        print(f"[WEB AGENT] Tavily search failed: {e}")
        return WebResult(web_context=[], source_urls=[], confidence=0.0)

    if not raw_results:
        print("[WEB AGENT] No results returned by Tavily.")
        return WebResult(web_context=[], source_urls=[], confidence=0.0)

    print(f"[WEB AGENT] {len(raw_results)} results. Fetching top 4 pages...")
    enriched = fetch_and_clean_results(raw_results, top_n=4)

    web_context: list[str] = []
    source_urls: list[str] = []
    scores: list[float]    = []

    for item in enriched:
        if item["snippet"]:
            web_context.append(item["snippet"])
            source_urls.append(item["url"])
            scores.append(item["score"])

    avg_confidence = float(sum(scores) / len(scores)) if scores else 0.0
    elapsed = time.perf_counter() - t_start

    print(
        f"[WEB AGENT] Done in {elapsed:.3f}s — "
        f"{len(web_context)} pages | avg score: {avg_confidence:.4f}"
    )

    return WebResult(
        web_context=web_context,
        source_urls=source_urls,
        confidence=avg_confidence,
    )
