# """
# agents/composio_agent.py — Composio External Tools Agent.
# 
# Executes external tools via Composio (GitHub, Google Docs, Tavily, YouTube,
# Context7 MCP, Hugging Face) based on user queries that require external actions.
# """
# 
# from typing import Any
# 
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langgraph.prebuilt import create_react_agent
# 
# from multi_agent.config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE
# from multi_agent.models.schemas import ComposioResult
# from multi_agent.tools.composio_tools import get_composio_tools
# 
# 
# _SYSTEM_PROMPT = """You are an agent with access to external tools via Composio.
# You can interact with:
# - GitHub: repositories, issues, pull requests, files, code search
# - Google Docs: create, read, update documents
# - Tavily: web search
# - YouTube: search videos, get transcripts, channel info
# - Context7 (MCP): search code/documentation from Context7
# - Hugging Face: models, datasets, spaces, inference
# 
# Given a user query, determine which tool(s) to use and execute them.
# Return the tool outputs clearly so the Answer Agent can synthesize a response.
# 
# Be precise about which tool to use:
# - GitHub: code, repos, issues, PRs, file contents
# - Google Docs: document content, creation, editing
# - Tavily: general web search
# - YouTube: video search, transcripts, metadata
# - Context7: code documentation, library docs
# - Hugging Face: models, datasets, spaces, inference
# 
# Execute the appropriate tool(s) and return the results."""
# 
# 
# def _build_agent():
#     """Create and return the LangGraph agent with Composio tools."""
#     tools = get_composio_tools()
# 
#     llm = ChatGoogleGenerativeAI(
#         model=LLM_MODEL,
#         google_api_key=GOOGLE_API_KEY,
#         temperature=LLM_TEMPERATURE,
#     )
# 
#     agent = create_react_agent(llm, tools, prompt=_SYSTEM_PROMPT)
#     return agent
# 
# 
# _agent_executor = None
# 
# 
# def _get_agent():
#     """Lazy-load the agent executor."""
#     global _agent_executor
#     if _agent_executor is None:
#         _agent_executor = _build_agent()
#     return _agent_executor
# 
# 
# def run(query: str, history_messages: list[Any] | None = None) -> ComposioResult:
#     """
#     Execute the Composio agent with the given query.
# 
#     Args:
#         query: The user's query that may require external tool usage.
#         history_messages: Optional conversation history for context.
# 
#     Returns:
#         ComposioResult containing tool outputs, names, success status, and metadata.
#     """
#     agent = _get_agent()
# 
#     # Build message history for context
#     messages = []
#     if history_messages:
#         messages.extend(history_messages[-6:])  # Last 6 messages for context
#     messages.append(HumanMessage(content=query))
# 
#     try:
#         result = agent.invoke({"messages": messages})
# 
#         # Extract tool outputs from the agent result
#         tool_outputs = []
#         tool_names = []
#         metadata = {}
# 
#         # LangGraph returns messages in result["messages"]
#         for msg in result.get("messages", []):
#             # Tool calls are in AIMessage with tool_calls
#             if hasattr(msg, "tool_calls") and msg.tool_calls:
#                 for tool_call in msg.tool_calls:
#                     tool_names.append(tool_call["name"])
#             # Tool responses are in ToolMessage
#             if msg.__class__.__name__ == "ToolMessage":
#                 tool_outputs.append(msg.content)
#                 # Extract metadata if available
#                 if hasattr(msg, "metadata") and msg.metadata:
#                     metadata.update(msg.metadata)
# 
#         output = ""
#         # Get the final AI message content
#         for msg in reversed(result.get("messages", [])):
#             if msg.__class__.__name__ == "AIMessage" and msg.content:
#                 output = msg.content
#                 break
# 
#         return ComposioResult(
#             tool_outputs=tool_outputs if tool_outputs else [output],
#             tool_names=tool_names,
#             success=True,
#             error=None,
#             metadata=metadata,
#         )
# 
#     except Exception as e:
#         return ComposioResult(
#             tool_outputs=[],
#             tool_names=[],
#             success=False,
#             error=str(e),
#             metadata={},
#         )

from multi_agent.models.schemas import ComposioResult

# Currently unused (external tool execution is disabled)
def run(query: str, history_messages: list = None) -> ComposioResult:
    return ComposioResult(
        tool_outputs=[],
        tool_names=[],
        success=False,
        error="Composio agent is disabled.",
        metadata={}
    )