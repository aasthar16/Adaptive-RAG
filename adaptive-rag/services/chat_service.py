"""
Chat service: chat CRUD (with ownership enforced on every read/write, per
section 29) plus `ask()`, the single entrypoint that turns a user message
into a persisted assistant reply by driving the LangGraph workflow.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import get_settings
from database.models import Chat, Message
from database.repositories.chat_repository import ChatRepository
from database.repositories.chunk_repository import ChunkRepository
from database.repositories.document_repository import DocumentRepository
from generation.generator import suggest_chat_name
from graph.state import RAGState
from graph.workflow import run_query
from services.document_service import DocumentService, UploadOutcome
from services.message_service import MessageService

logger = logging.getLogger(__name__)


class ChatNotFoundError(LookupError):
    """Raised when a chat doesn't exist OR doesn't belong to the requesting
    user — the two cases are deliberately indistinguishable to callers."""


@dataclass
class AskResult:
    assistant_message: Message
    debug_info: dict
    final_state: RAGState


@dataclass
class NewChatResult:
    chat: Chat
    upload_outcomes: list[UploadOutcome] = field(default_factory=list)


class ChatService:
    def __init__(self) -> None:
        self.chats = ChatRepository()
        self.documents = DocumentRepository()
        self.chunks = ChunkRepository()
        self.document_service = DocumentService()
        self.message_service = MessageService()
        self.settings = get_settings()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def create_chat(
        self, *, user_id: str, uploads: list[tuple[str, bytes]], title: str | None = None
    ) -> NewChatResult:
        suggested = title
        if not suggested:
            filenames = [name for name, _ in uploads]
            suggested = suggest_chat_name(filenames) if filenames else "New Chat"

        chat = self.chats.create(user_id=user_id, title=suggested or "New Chat")

        outcomes: list[UploadOutcome] = []
        if uploads:
            outcomes = self.document_service.add_more_documents(
                user_id=user_id, chat_id=chat.id, uploads=uploads
            )
        return NewChatResult(chat=chat, upload_outcomes=outcomes)

    def list_chats(self, user_id: str) -> list[Chat]:
        return self.chats.list_for_user(user_id)

    def get_chat(self, chat_id: str, user_id: str) -> Chat:
        chat = self.chats.get_owned_by_user(chat_id, user_id)
        if chat is None:
            raise ChatNotFoundError(chat_id)
        return chat

    def rename_chat(self, chat_id: str, user_id: str, new_title: str) -> None:
        if not self.chats.rename(chat_id, user_id, new_title.strip()[:120] or "Untitled chat"):
            raise ChatNotFoundError(chat_id)

    def add_documents(
        self, chat_id: str, user_id: str, uploads: list[tuple[str, bytes]]
    ) -> list[UploadOutcome]:
        # Ownership check first — never let a chat_id from the frontend be
        # trusted without verifying it belongs to the requesting user.
        self.get_chat(chat_id, user_id)
        return self.document_service.add_more_documents(user_id=user_id, chat_id=chat_id, uploads=uploads)

    def delete_chat(self, chat_id: str, user_id: str) -> None:
        """Hard-delete a chat and everything that belongs to it (section 37):
        messages, documents (DB rows + files on disk), vector chunks, and
        the LangGraph checkpoint thread. Ownership is re-verified here even
        though callers should have already checked it — this method is
        destructive and irreversible, so it never trusts a bare chat_id.
        """
        chat = self.get_chat(chat_id, user_id)

        # Soft-delete immediately so the chat disappears from the sidebar
        # even if a later cleanup step below fails partway through.
        self.chats.soft_delete(chat.id, user_id)

        for doc in self.documents.list_for_chat(chat.id):
            try:
                Path(doc.storage_reference).unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove file for document %s", doc.id)

        self._delete_langgraph_thread(user_id=user_id, chat_id=chat.id)

        # documents / messages / document_chunks all cascade-delete via
        # ON DELETE CASCADE foreign keys defined in the migration, so a
        # single delete of the chat row is enough to leave no orphaned
        # vector records or messages behind.
        from database.connection import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chats WHERE id = %s AND user_id = %s", (chat.id, user_id))

    def _delete_langgraph_thread(self, *, user_id: str, chat_id: str) -> None:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            thread_id = f"user_{user_id}_chat_{chat_id}"
            with PostgresSaver.from_conn_string(self.settings.database_url_psycopg) as saver:
                saver.delete_thread(thread_id)
        except Exception:
            logger.warning("Could not delete LangGraph thread for chat %s", chat_id, exc_info=True)

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------
    def ask(self, *, chat_id: str, user_id: str, query: str) -> AskResult:
        # Ownership check before touching anything else in this chat.
        self.get_chat(chat_id, user_id)

        self.message_service.add_user_message(chat_id, user_id, query)
        history_text = self.message_service.render_recent_history(chat_id)

        final_state = run_query(
            user_id=user_id, chat_id=chat_id, query=query, chat_history_text=history_text
        )

        answer = final_state.get("answer") or "I wasn't able to generate an answer."
        sources = final_state.get("sources", [])
        debug_info = self._build_debug_info(final_state)

        assistant_message = self.message_service.add_assistant_message(
            chat_id, user_id, answer, sources, debug_info
        )
        self.chats.touch(chat_id)

        return AskResult(assistant_message=assistant_message, debug_info=debug_info, final_state=final_state)

    @staticmethod
    def _build_debug_info(state: RAGState) -> dict:
        """Everything the '🔍 Retrieval Analysis' UI panel (section 26)
        needs, plus the cost/latency counters from section 27."""
        import time

        started_at = state.get("started_at")
        elapsed = round(time.time() - started_at, 2) if started_at else None
        return {
            "query_type": state.get("query_type"),
            "complexity": state.get("complexity"),
            "initial_strategy": (state.get("retrieval_history") or [{}])[0].get("strategy")
            if state.get("retrieval_history")
            else None,
            "retrieval_history": state.get("retrieval_history", []),
            "evaluation_history": state.get("evaluation_history", []),
            "correction_history": state.get("correction_history", []),
            "verification_result": state.get("verification_result"),
            "verification_retries": state.get("verification_retries", 0),
            "iterations": state.get("iteration", 0),
            "llm_call_count": state.get("llm_call_count", 0),
            "retrieval_call_count": state.get("retrieval_call_count", 0),
            "total_tokens_estimate": state.get("total_tokens_estimate", 0),
            "elapsed_seconds": elapsed,
            "gave_up": state.get("give_up", False),
        }
