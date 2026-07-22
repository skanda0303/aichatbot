"""
agent.py — LLM setup, agent assembly, and query pipeline.

Initializes the Gemini LLM, assembles the LangChain agent with its
system prompt and tools, and exposes:
  - run_agent()           — returns the full answer as a string (used by tests)
  - stream_agent_tokens() — async generator yielding only final-answer tokens
                            (tool-call / reasoning events are filtered out)
"""

import time
from collections.abc import AsyncGenerator
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from ragbot.config import LLM_MODEL, GOOGLE_API_KEY, LLM_TEMPERATURE, MAX_HISTORY_MESSAGES
from ragbot.tools import web_search, fetch_webpage_content, get_datetime, register_rag_context
from ragbot.retriever import filter_redundant_docs

# LLM
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=LLM_TEMPERATURE,
)

# Agent — only web tools; RAG context is injected directly into the prompt
agent_tools = [web_search, fetch_webpage_content, get_datetime]

agent_executor = create_agent(
    model=llm,
    tools=agent_tools,
    system_prompt=(
        f"You are a precise, fact-grounded assistant. "
        f"Current date is {datetime.now().strftime('%A, %B %d, %Y')}.\n\n"
        "SYNTHESIS RULES:\n"
        "1. Answer directly from facts in tool results or pre-fetched RAG context "
        "— never tell the user to visit a site.\n"
        "2. Be comprehensive and detailed. Use bullet points or bold text where helpful.\n"
        "3. Trust tool outputs over your training memory.\n"
        "4. Do not repeat identical search queries. You are encouraged to do multi-step "
        "searching or call 'fetch_webpage_content' on multiple URLs for detailed info.\n"
        "5. If you need detailed info from a URL returned by web_search, "
        "call 'fetch_webpage_content' with that URL.\n"
        "6. CHRONOLOGICAL TIMELINES: Verify dates carefully and list oldest to newest.\n"
        "7. WEB FALLBACK: If 'Pre-fetched RAG context' doesn't contain the answer or is blank, "
        "you MUST call 'web_search'. Never just say the text doesn't contain the info.\n\n"
        # "VOICE EMOTION AND PROSODY:\n"
        # "To make the voice narration feel highly expressive, natural, and human-like, "
        # "you must inject descriptive style/emotion tags in square brackets where appropriate "
        # "(e.g., [excited], [sigh], [whisper], [laughing], [thoughtful], [sad], [proud], [serious], [gasp]). "
        # "Use them inline (e.g. '[excited] That is wonderful!' or 'I was quite surprised [gasp] by the results.'). "
        # "Use them when explaining something with natural context-driven emotion."
    ),
)


def initialise_agent(retriever, chunks: list) -> None:
    """Wire retriever and chunks into the tools module at startup."""
    register_rag_context(retriever, chunks)


def sanitize_tool_output(text: str) -> str:
    """Normalize whitespace and cap at 25k chars to prevent context overflow."""
    text = " ".join(text.split())
    if len(text) > 25000:
        text = text[:25000] + "... [truncated]"
    return text


def _build_inputs(query: str, history_messages: list, chunks: list, retriever) -> tuple[dict, str]:
    """
    Shared helper: pre-fetch RAG context, construct the agent input dict,
    and return (inputs_dict, user_input_string).
    """
    rag_context = ""
    has_rag_docs = False

    if chunks:
        try:
            t_ret = time.perf_counter()
            retrieved_docs = retriever.invoke(query)
            print(f"[AGENT] Retrieval: {time.perf_counter() - t_ret:.3f}s — {len(retrieved_docs)} docs")

            if retrieved_docs:
                seen, unique = set(), []
                for doc in retrieved_docs:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        unique.append(doc)

                filtered = filter_redundant_docs(unique)
                if filtered:
                    has_rag_docs = True
                    formatted_chunks = [
                        doc.page_content.replace("●", "\n- ") for doc in filtered[:3]
                    ]
                    rag_context = "\n\n---\n\n".join(formatted_chunks)
                    print(f"[AGENT] Injecting {len(filtered[:3])} RAG chunks.")
        except Exception as e:
            print(f"[AGENT] Retrieval error: {e}")

    user_input = query
    if has_rag_docs:
        sanitized_rag = sanitize_tool_output(rag_context)
        user_input = (
            f"Pre-fetched RAG context:\n{sanitized_rag}\n\n"
            f"Now answer the query: {query}"
        )

    inputs = {
        "messages": list(history_messages[-MAX_HISTORY_MESSAGES:])
        + [HumanMessage(content=user_input)]
    }
    return inputs, user_input


async def stream_agent_tokens(
    query: str, history_messages: list, chunks: list, retriever
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields only the final-answer token strings from the
    LangChain agent, filtering out tool-call and reasoning events.

    Consumers should accumulate all yielded tokens to reconstruct the full
    answer text for history persistence.
    """
    print(f"[AGENT] Streaming query: '{query}'")
    inputs, _ = _build_inputs(query, history_messages, chunks, retriever)

    try:
        async for event in agent_executor.astream_events(inputs, version="v2"):
            event_type = event.get("event", "")

            # Only forward tokens from the final chat model output.
            # Skip on_tool_start / on_tool_end / on_tool_call / on_chain_* etc.
            if event_type != "on_chat_model_stream":
                continue

            # Skip intermediate tool-calling model chunks (they have tool_calls data)
            chunk = event.get("data", {}).get("chunk")
            if chunk is None:
                continue

            # Filter out tool-call chunks (AIMessageChunk with tool_calls, no text)
            tool_calls = getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None)
            if tool_calls:
                continue

            content = chunk.content
            if isinstance(content, str) and content:
                yield content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str) and part:
                        yield part
                    elif isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        yield part["text"]

    except Exception as e:
        print(f"[AGENT ERROR] stream_agent_tokens: {e}")
        yield f"An error occurred: {e}"


async def run_agent(query: str, history_messages: list, chunks: list, retriever) -> str:
    """Process a user query: pre-fetch RAG context → run LLM agent → return answer."""
    print(f"[AGENT] Query: '{query}'")

    inputs, _ = _build_inputs(query, history_messages, chunks, retriever)

    try:
        response = await agent_executor.ainvoke(inputs)
        final_message = response["messages"][-1]
        content = final_message.content

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            return "".join(text_parts)

        return str(content)
    except Exception as e:
        print(f"[AGENT ERROR] {e}")
        return f"An error occurred while executing the agent: {e}"
