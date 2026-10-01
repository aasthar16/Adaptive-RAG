from __future__ import annotations

from typing import Optional

from database.connection import get_connection
from database.models import Document


class DocumentRepository:
    def create(
        self,
        user_id: str,
        chat_id: str,
        filename: str,
        file_type: str,
        storage_reference: str,
    ) -> Document:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents
                        (user_id, chat_id, filename, file_type, storage_reference, status)
                    VALUES (%s, %s, %s, %s, %s, 'pending')
                    RETURNING id, user_id, chat_id, filename, file_type,
                              storage_reference, status, page_count, created_at
                    """,
                    (user_id, chat_id, filename, file_type, storage_reference),
                )
                row = cur.fetchone()
        return Document(**row)

    def set_storage_reference(self, document_id: str, storage_reference: str) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET storage_reference = %s WHERE id = %s",
                    (storage_reference, document_id),
                )

    def delete(self, document_id: str, user_id: str) -> bool:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE id = %s AND user_id = %s", (document_id, user_id)
                )
                return cur.rowcount > 0

    def set_status(
        self,
        document_id: str,
        status: str,
        page_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents
                    SET status = %s, page_count = COALESCE(%s, page_count),
                        error_message = %s
                    WHERE id = %s
                    """,
                    (status, page_count, error_message, document_id),
                )

    def list_for_chat(self, chat_id: str) -> list[Document]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, chat_id, filename, file_type,
                           storage_reference, status, page_count, created_at
                    FROM documents WHERE chat_id = %s
                    ORDER BY created_at ASC
                    """,
                    (chat_id,),
                )
                rows = cur.fetchall()
        return [Document(**row) for row in rows]

    def get_owned(self, document_id: str, user_id: str) -> Optional[Document]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, chat_id, filename, file_type,
                           storage_reference, status, page_count, created_at
                    FROM documents WHERE id = %s AND user_id = %s
                    """,
                    (document_id, user_id),
                )
                row = cur.fetchone()
        return Document(**row) if row else None
