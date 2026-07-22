"""
agents/query_rewriter_agent.py — Query Rewriting and Expansion Agent.

Optimizes user queries by incorporating conversation history to resolve pronouns,
ellipses, and follow-ups, correcting typos, and expanding abbreviations for Web search.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from multi_agent.config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE

_llm = None

_REWRITER_PROMPT = (
    "You are an expert search query optimizer. Your task is to analyze the raw user query and the recent conversation history to "
    "resolve any context-dependent references, pronouns (e.g., 'it', 'they', 'its'), ellipses, or shorthand follow-ups. "
    "Correct any spelling typos, expand abbreviations (e.g., '1t+' -> '1 trillion parameters', "
    "'isro' -> 'Indian Space Research Organisation'), and generate an optimized web search query.\n\n"
    "Output MUST be in raw JSON format with exactly this key:\n"
    "{\n"
    "  \"web_query\": \"optimized query for Google/Tavily web search\"\n"
    "}\n"
    "Respond ONLY with the raw JSON object, without any markdown code fences or explanations."
)

def _format_history(history_messages: list) -> str:
    """Format recent history messages into a clean text transcript."""
    formatted = []
    for msg in history_messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

# Called in: multi_agent/agents/supervisor_agent.py (run_streaming, run)
async def run(query: str, history_messages: list, user_gemini_key: str | None = None) -> dict:
    """Analyze the query and history, and return a dictionary containing the contextual web_query."""
    from datetime import datetime
    current_date_str = datetime.now().strftime('%A, %B %d, %Y')
    current_year = datetime.now().year

    dynamic_prompt = (
        "You are an expert search query optimizer. "
        f"The current date is {current_date_str}.\n\n"
        "Your task is to analyze the raw user query and the recent conversation history to "
        "resolve any context-dependent references, pronouns (e.g., 'it', 'they', 'its'), ellipses, or shorthand follow-ups. "
        "Correct any spelling typos, expand abbreviations (e.g., '1t+' -> '1 trillion parameters', "
        "'isro' -> 'Indian Space Research Organisation'), and generate an optimized web search query.\n\n"
        "CRITICAL FOR TIME-SENSITIVE QUERIES:\n"
        "If the query asks about upcoming, current, or next occurrences of recurring events (e.g., 'next festival in India', 'latest AI models', 'who is currently the CEO'), "
        f"ensure that the generated search query explicitly targets the correct year/period (which is {current_year}) or future years, "
        "and never references outdated years like 2024 or 2025 unless explicitly asked by the user.\n\n"
        "Output MUST be in raw JSON format with exactly this key:\n"
        "{\n"
        "  \"web_query\": \"optimized query for Google/Tavily web search\"\n"
        "}\n"
        "Respond ONLY with the raw JSON object, without any markdown code fences or explanations."
    )

    history_str = _format_history(history_messages)
    user_content = f"=== Conversation History ===\n{history_str}\n\n=== Follow-up Query ===\n{query}"
    
    messages = [
        SystemMessage(content=dynamic_prompt),
        HumanMessage(content=user_content)
    ]
    key = user_gemini_key or GOOGLE_API_KEY
    if not key:
        print("[QUERY REWRITER] Gemini API Key is missing — returning raw query.")
        return {"web_query": query}

    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=key,
            temperature=LLM_TEMPERATURE,
        )
        response = await llm.ainvoke(messages)
        content = response.content
        if isinstance(content, list):
            content = "".join(part if isinstance(part, str) else part.get("text", "") for part in content)
        content_str = str(content).strip()
        
        # Strip any accidental markdown blocks
        if content_str.startswith("```"):
            lines = content_str.splitlines()
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                content_str = "\n".join(lines[1:-1]).strip()

        data = json.loads(content_str)
        if "web_query" in data:
            print(f"[QUERY REWRITER] Contextual Web query generated:\n  Web: '{data['web_query']}'")
            return data
    except Exception as e:
        print(f"[QUERY REWRITER] Failed to rewrite query: {e}. Falling back to raw query.")
        
    return {"web_query": query}
