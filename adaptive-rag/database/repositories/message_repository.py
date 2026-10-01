from __future__ import annotations

import json

from database.connection import get_connection
from database.models import Message


class MessageRepository:
    def create(
        self,
        chat_id: str,
        user_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
        debug_info: dict | None = None,
    ) -> Message:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO messages (chat_id, user_id, role, content, sources, debug_info)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    RETURNING id, chat_id, user_id, role, content, sources, debug_info, created_at
                    """,
                    (
                        chat_id,
                        user_id,
                        role,
                        content,
                        json.dumps(sources or []),
                        json.dumps(debug_info or {}),
                    ),
                )
                row = cur.fetchone()
        return Message(**row)

    def list_for_chat(self, chat_id: str, limit: int = 200) -> list[Message]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, chat_id, user_id, role, content, sources, debug_info, created_at
                    FROM messages WHERE chat_id = %s
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (chat_id, limit),
                )
                rows = cur.fetchall()
        return [Message(**row) for row in rows]
