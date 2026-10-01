from __future__ import annotations

from database.models import Message
from database.repositories.message_repository import MessageRepository

MAX_HISTORY_TURNS_FOR_CONTEXT = 4


class MessageService:
    def __init__(self) -> None:
        self.messages = MessageRepository()

    def list_for_chat(self, chat_id: str) -> list[Message]:
        return self.messages.list_for_chat(chat_id)

    def add_user_message(self, chat_id: str, user_id: str, content: str) -> Message:
        return self.messages.create(chat_id, user_id, role="user", content=content)

    def add_assistant_message(
        self, chat_id: str, user_id: str, content: str, sources: list[dict], debug_info: dict
    ) -> Message:
        return self.messages.create(
            chat_id, user_id, role="assistant", content=content, sources=sources, debug_info=debug_info
        )

    def render_recent_history(self, chat_id: str) -> str:
        """A short plain-text rendering of the last few turns, passed to the
        LLM as conversational context only — never as a substitute for
        retrieval, and never persisted as part of the graph's own state."""
        history = self.messages.list_for_chat(chat_id)
        recent = history[-(MAX_HISTORY_TURNS_FOR_CONTEXT * 2):]
        lines = []
        for m in recent:
            speaker = "User" if m.role == "user" else "Assistant"
            snippet = m.content if len(m.content) <= 500 else m.content[:500] + "…"
            lines.append(f"{speaker}: {snippet}")
        return "\n".join(lines)
