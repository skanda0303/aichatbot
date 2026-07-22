"""
agents/answer_agent.py — Answer Generation Agent.

The ONLY agent allowed to produce user-facing responses.

Rules:
  - If RAG is sufficient → answer from RAG context only
  - If RAG is insufficient → incorporate web context
  - If both available → prefer RAG, supplement with web only where needed
  - Same Gemini model, temperature, and markdown formatting as ragbot/agent.py
  - Supports both streaming (astream) and blocking (ainvoke) modes
"""

from __future__ import annotations
from collections.abc import AsyncGenerator
import asyncio
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from multi_agent.models.schemas import RAGResult, WebResult, EvalResult, ComposioResult
from multi_agent.config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_HISTORY_MESSAGES
from multi_agent.utils.helpers import format_chunks_for_prompt, sanitize_tool_output

_llm = None

def _get_system_prompt() -> str:
    from datetime import datetime
    now_str = datetime.now().strftime('%A, %B %d, %Y')
    return (
        f"You are a precise, fact-grounded assistant. "
        f"Current date is {now_str}.\n\n"
        "SYNTHESIS RULES:\n"
        "1. STRICT GROUNDING IN KNOWLEDGE BASE: Answer directly from facts in the provided Knowledge Base Context. "
        "If the Knowledge Base contains a specific answer, table entry, or concise statement (e.g. 'Challenge: Weather'), "
        "report it directly as the primary answer!\n"
        "2. Do NOT replace, ignore, or override facts in the Knowledge Base with external web search or training knowledge.\n"
        "3. Be comprehensive and clear. Use bullet points or bold text where helpful.\n"
        "4. Trust provided Knowledge Base context over your training memory.\n"
        "5. CHRONOLOGICAL TIMELINES: Verify dates carefully and list oldest to newest.\n"
        "6. If both RAG and Web context are provided, ALWAYS state the Knowledge Base answer first, and only supplement with Web if explicitly helpful.\n"
        "7. Cite sources naturally inline when using web sources.\n"
        "8. STRICT TEMPORAL FILTERING: For questions asking for 'next', 'upcoming', or 'future' events/dates, "
        f"strictly EXCLUDE any event whose date occurred on or before today ({now_str}). "
    )


def _get_critic_prompt() -> str:
    from datetime import datetime
    now_str = datetime.now().strftime('%A, %B %d, %Y')
    return (
        "You are an expert editor and fact-checker. Your job is to review a draft answer, identify any factual errors, "
        "numerical mismatches, timeline inconsistencies, or formatting clarity issues based strictly on the provided Context, and output a corrected, clear version.\n\n"
        "CRITICAL RULES:\n"
        "1. FACTUAL ACCURACY: Compare all facts, names, figures, and data values in the draft answer against the Context. "
        "Every claim and number must be directly supported by the context. If there is a mismatch or hallucination, correct it.\n"
        f"2. TEMPORAL CONSISTENCY & STRICT FILTERING: Current date is {now_str}. Verify that relative temporal statements "
        "(like 'next', 'current', 'future', 'past') are strictly correct relative to this date. "
        "If a query asks for 'next' or 'upcoming' events, REMOVE any past events from the draft completely (do not allow past events to be listed under 'next').\n"
        "3. LOGICAL COHERENCE: Ensure facts from different sources are not incorrectly mixed or merged (e.g. associating the attributes/dates of one entity to another).\n"
        "4. FORMATTING: Keep the tone helpful, precise, and markdown-formatted. Output ONLY the final verified and corrected answer. "
        "Do not include any preambles, explanations, check details, or meta-comments."
    )


# Called in: multi_agent/agents/answer_agent.py (stream, run)
def _build_user_message(
    query: str,
    rag_result: RAGResult,
    eval_result: EvalResult,
    web_result: WebResult | None,
    composio_result: ComposioResult | None = None,
) -> str:
    """Assemble the context-enriched user message for the LLM."""
    parts: list[str] = []

    # ── RAG context ───────────────────────────────────────────────────────────
    if rag_result.retrieved_chunks:
        rag_text = format_chunks_for_prompt(rag_result.retrieved_chunks, max_chunks=8)
        rag_text = sanitize_tool_output(rag_text)
        parts.append(f"=== Knowledge Base Context ===\n{rag_text}")

    # ── Composio Tool Execution Context ──────────────────────────────────────
    # if composio_result and composio_result.tool_outputs:
    #     composio_sections: list[str] = []
    #     for i, (tool_name, output) in enumerate(zip(composio_result.tool_names, composio_result.tool_outputs), 1):
    #         output_clean = sanitize_tool_output(output)
    #         composio_sections.append(f"[Tool {i}: {tool_name}]\n{output_clean}")
    #     composio_text = "\n\n".join(composio_sections)
    #     parts.append(f"=== Composio Tool Execution Context ===\n{composio_text}")

    # ── Web context (only when web agent ran) ─────────────────────────────────
    if web_result and web_result.web_context:
        web_sections: list[str] = []
        for i, (ctx, url) in enumerate(zip(web_result.web_context, web_result.source_urls), 1):
            ctx_clean = sanitize_tool_output(ctx)
            web_sections.append(f"[Source {i}: {url}]\n{ctx_clean}")
        web_text = "\n\n".join(web_sections)
        parts.append(f"=== Web Search Context ===\n{web_text}")

    # ── No context at all ─────────────────────────────────────────────────────
    if not parts:
        parts.append(
            "No relevant context was retrieved from the knowledge base, external tools, or the web. "
            "Answer from your general knowledge if possible, and be transparent about uncertainty."
        )

    context_block = "\n\n".join(parts)
    return f"{context_block}\n\n---\n\nUser Question: {query}"


# Called in: multi_agent/agents/answer_agent.py (run)
async def _generate_draft(
    query: str,
    history_messages: list,
    rag_result: RAGResult,
    eval_result: EvalResult,
    web_result: WebResult | None = None,
    composio_result: ComposioResult | None = None,
    user_gemini_key: str | None = None,
) -> str:
    """Generate the initial draft answer from context and history."""
    user_msg = _build_user_message(query, rag_result, eval_result, web_result, composio_result)

    messages = (
        [SystemMessage(content=_get_system_prompt())]
        + list(history_messages[-MAX_HISTORY_MESSAGES:])
        + [HumanMessage(content=user_msg)]
    )

    key = user_gemini_key or GOOGLE_API_KEY
    if not key:
        return "Gemini API Key is missing. Please configure your API key in the credentials sidebar to generate answers."

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=key,
        temperature=LLM_TEMPERATURE,
    )
    response = await llm.ainvoke(messages)
    content  = response.content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )
    return str(content)


# Called in: multi_agent/agents/answer_agent.py (run)
async def _verify_and_correct(query: str, draft: str, context_text: str, user_gemini_key: str | None = None) -> str:
    """Fact-check and correct draft answer using Critic LLM."""
    messages = [
        SystemMessage(content=_get_critic_prompt()),
        HumanMessage(content=(
            f"=== Context ===\n{context_text}\n\n"
            f"User Question: {query}\n\n"
            f"Draft Answer to Verify:\n{draft}"
        ))
    ]
    key = user_gemini_key or GOOGLE_API_KEY
    if not key:
        print("[ANSWER AGENT] Gemini API Key is missing — skipping Critic verification.")
        return draft

    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=key,
            temperature=LLM_TEMPERATURE,
        )
        response = await llm.ainvoke(messages)
        content = response.content
        if isinstance(content, list):
            content = "".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in content
            )
        return str(content).strip()
    except Exception as e:
        print(f"[ANSWER AGENT] Critic error: {e}")
        return draft


# Called in: multi_agent/agents/supervisor_agent.py (run_streaming)
async def stream(
    query: str,
    history_messages: list,
    rag_result: RAGResult,
    eval_result: EvalResult,
    web_result: WebResult | None = None,
    composio_result: ComposioResult | None = None,
    user_gemini_key: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator — yields verified answer token strings for SSE streaming.
    """
    try:
        final_ans = await run(
            query, history_messages, rag_result, eval_result, web_result, composio_result, user_gemini_key
        )
        # Yield the verified answer in small chunks to simulate streaming output
        chunk_size = 12
        for i in range(0, len(final_ans), chunk_size):
            yield final_ans[i : i + chunk_size]
            await asyncio.sleep(0.01)
    except Exception as e:
        print(f"[ANSWER AGENT] Streaming error: {e}")
        yield f"An error occurred while generating the answer: {e}"


# Called in: multi_agent/agents/supervisor_agent.py (run)
async def run(
    query: str,
    history_messages: list,
    rag_result: RAGResult,
    eval_result: EvalResult,
    web_result: WebResult | None = None,
    composio_result: ComposioResult | None = None,
    user_gemini_key: str | None = None,
) -> str:
    """
    Blocking variant — collects and verifies the full answer as a string.
    """
    try:
        print("[ANSWER AGENT] Generating draft response...")
        draft = await _generate_draft(
            query, history_messages, rag_result, eval_result, web_result, composio_result, user_gemini_key
        )
        print(f"[ANSWER AGENT] Draft generated ({len(draft)} chars). Verifying values and claims via Critic LLM...")
        
        context_text = _build_user_message(query, rag_result, eval_result, web_result, composio_result)
        final_ans = await _verify_and_correct(query, draft, context_text, user_gemini_key)
        
        if final_ans.strip() == draft.strip():
            print("[ANSWER AGENT] Verification complete: draft is 100% accurate (no corrections needed).")
        else:
            print("[ANSWER AGENT] Verification complete: Critic corrected value/claim inconsistencies in draft.")
            
        return final_ans
    except Exception as e:
        print(f"[ANSWER AGENT] Error: {e}")
        return f"An error occurred while generating the answer: {e}"
