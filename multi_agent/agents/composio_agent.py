"""
agents/composio_agent.py — Composio External Tools Agent.

Executes external tools via Composio (GitHub, Google Docs, Tavily, YouTube,
Context7 MCP, Hugging Face) based on user queries that require external actions.
"""

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from multi_agent.config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from multi_agent.models.schemas import ComposioResult
from multi_agent.tools.composio_tools import get_composio_tools


_SYSTEM_PROMPT = """You are an agent with access to external tools via Composio.
You can interact with:
- GitHub: repositories, issues, pull requests, files, code search
- Google Docs: create, read, update documents
- Tavily: web search
- YouTube: search videos, get transcripts, channel info
- Context7 (MCP): search code/documentation from Context7
- Hugging Face: models, datasets, spaces, inference

Given a user query, determine which tool(s) to use and execute them.
Return the tool outputs clearly so the Answer Agent can synthesize a response.

Be precise about which tool to use:
- GitHub: code, repos, issues, PRs, file contents
- Google Docs: document content, creation, editing
- Tavily: general web search
- YouTube: video search, transcripts, metadata
- Context7: code documentation, library docs
- Hugging Face: models, datasets, spaces, inference

Execute the appropriate tool(s) and return the results."""


def _build_agent(user_gemini_key: str | None = None):
    """Create and return the LangGraph agent with Composio tools."""
    tools = get_composio_tools()
    if not tools:
        return None

    api_key = user_gemini_key or GOOGLE_API_KEY
    if not api_key:
        return None

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=LLM_TEMPERATURE,
    )

    agent = create_react_agent(llm, tools, prompt=_SYSTEM_PROMPT)
    return agent


_agent_executor = None


def _get_agent(user_gemini_key: str | None = None):
    """Lazy-load the agent executor."""
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = _build_agent(user_gemini_key)
    return _agent_executor


def run(query: str, history_messages: list[Any] | None = None, user_gemini_key: str | None = None) -> ComposioResult:
    """
    Execute the Composio agent with the given query.

    Args:
        query: The user's query that may require external tool usage.
        history_messages: Optional conversation history for context.
        user_gemini_key: Optional custom Gemini API key.

    Returns:
        ComposioResult containing tool outputs, names, success status, and metadata.
    """
    try:
        agent = _get_agent(user_gemini_key)
        if not agent:
            return ComposioResult(
                tool_outputs=[],
                tool_names=[],
                success=False,
                error="Composio agent unavailable or no tools loaded.",
                metadata={},
            )

        messages = []
        if history_messages:
            messages.extend(history_messages[-6:])
        messages.append(HumanMessage(content=query))

        result = agent.invoke({"messages": messages})

        tool_outputs = []
        tool_names = []
        metadata = {}

        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_names.append(tool_call["name"])
            if msg.__class__.__name__ == "ToolMessage":
                tool_outputs.append(str(msg.content))
                if hasattr(msg, "metadata") and msg.metadata:
                    metadata.update(msg.metadata)

        output = ""
        for msg in reversed(result.get("messages", [])):
            if msg.__class__.__name__ == "AIMessage" and msg.content:
                output = str(msg.content)
                break

        return ComposioResult(
            tool_outputs=tool_outputs if tool_outputs else ([output] if output else []),
            tool_names=tool_names,
            success=True,
            error=None,
            metadata=metadata,
        )

    except Exception as e:
        print(f"[COMPOSIO AGENT] Error during execution: {e}")
        return ComposioResult(
            tool_outputs=[],
            tool_names=[],
            success=False,
            error=str(e),
            metadata={},
        )