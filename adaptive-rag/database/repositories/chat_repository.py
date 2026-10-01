from __future__ import annotations

from typing import Optional

from database.connection import get_connection
from database.models import Chat


class ChatRepository:
    def create(self, user_id: str, title: str) -> Chat:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chats (user_id, title)
                    VALUES (%s, %s)
                    RETURNING id, user_id, title, created_at, updated_at, is_deleted
                    """,
                    (user_id, title),
                )
                row = cur.fetchone()
        return Chat(**row)

    def get_by_id(self, chat_id: str) -> Optional[Chat]:
        """Fetch a chat by id with NO ownership check.

        Callers MUST verify ownership themselves (see `belongs_to_user`) —
        this method exists only for internal use where the caller has
        already authorized the request.
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, title, created_at, updated_at, is_deleted
                    FROM chats WHERE id = %s AND NOT is_deleted
                    """,
                    (chat_id,),
                )
                row = cur.fetchone()
        return Chat(**row) if row else None

    def get_owned_by_user(self, chat_id: str, user_id: str) -> Optional[Chat]:
        """The one method the rest of the app should use to load a chat.

        Returns None (not an exception) if the chat does not exist OR
        belongs to a different user, so callers can't accidentally leak
        whether a chat id exists for someone else.
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, title, created_at, updated_at, is_deleted
                    FROM chats WHERE id = %s AND user_id = %s AND NOT is_deleted
                    """,
                    (chat_id, user_id),
                )
                row = cur.fetchone()
        return Chat(**row) if row else None

    def list_for_user(self, user_id: str) -> list[Chat]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, title, created_at, updated_at, is_deleted
                    FROM chats
                    WHERE user_id = %s AND NOT is_deleted
                    ORDER BY updated_at DESC
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()
        return [Chat(**row) for row in rows]

    def rename(self, chat_id: str, user_id: str, new_title: str) -> bool:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chats SET title = %s
                    WHERE id = %s AND user_id = %s AND NOT is_deleted
                    """,
                    (new_title, chat_id, user_id),
                )
                return cur.rowcount > 0

    def touch(self, chat_id: str) -> None:
        """Bump updated_at, used to keep the sidebar ordered by recency."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE chats SET updated_at = now() WHERE id = %s", (chat_id,))

    def soft_delete(self, chat_id: str, user_id: str) -> bool:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chats SET is_deleted = TRUE
                    WHERE id = %s AND user_id = %s
                    """,
                    (chat_id, user_id),
                )
                return cur.rowcount > 0
