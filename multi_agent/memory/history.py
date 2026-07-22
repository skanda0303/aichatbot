"""
memory/history.py — SQLChatMessageHistory wrapper with history trimming.
"""

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from multi_agent.config import MEMORY_DB, MAX_HISTORY_MESSAGES


# Called in: multi_agent/memory/history.py (get_recent_messages, get_all, append_exchange, clear_history)
def _history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(session_id=session_id, connection_string=MEMORY_DB)


# Called in: multi_agent/api.py (chat)
def get_recent_messages(session_id: str) -> list[BaseMessage]:
    return list(_history(session_id).messages[-MAX_HISTORY_MESSAGES:])


# Unused in production
def get_all(session_id: str) -> list[dict]:
    entries = []
    for msg in _history(session_id).messages:
        role = "user" if isinstance(msg, HumanMessage) else "bot"
        entries.append({"role": role, "content": msg.content})
    return entries


# Called in: multi_agent/api.py (chat)
def append_exchange(session_id: str, user_message: str, ai_message: str) -> None:
    history = _history(session_id)
    history.add_message(HumanMessage(content=user_message))
    history.add_message(AIMessage(content=ai_message))


# Called in: multi_agent/api.py (clear_chat)
def clear_history(session_id: str) -> None:
    _history(session_id).clear()
