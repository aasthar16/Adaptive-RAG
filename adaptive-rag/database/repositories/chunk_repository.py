"""
Direct SQL access to `document_chunks` for operations that the
langchain-postgres `PGVectorStore` abstraction doesn't cover:

- keyword / full-text (BM25-ish) search via PostgreSQL's `ts_rank`
- chat-scoped deletion (chat/document cleanup)
- ownership-aware counts for the UI

The dense-vector similarity search itself lives in
`retrieval/vector_retriever.py` and goes through `PGVectorStore`, but it is
built on the exact same table so results from both paths are directly
comparable and mergeable (hybrid retrieval).
"""
from __future__ import annotations

from database.connection import get_connection


class ChunkRepository:
    def keyword_search(
        self, chat_id: str, query: str, k: int = 6
    ) -> list[dict]:
        """PostgreSQL full-text search scored with ts_rank, scoped to one chat.

        This is intentionally implemented with Postgres' built-in FTS
        (GIN index on a generated tsvector column) rather than pulling in a
        separate BM25 engine — it needs no extra index-build step, stays
        consistent with the rest of the persistence layer, and is filtered
        by `chat_id` at the SQL level so isolation can never be bypassed.
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        dc.id,
                        dc.document_id,
                        dc.chunk_index,
                        dc.content,
                        dc.metadata,
                        ts_rank(dc.content_tsv, websearch_to_tsquery('english', %s)) AS score,
                        d.filename
                    FROM document_chunks dc
                    JOIN documents d ON d.id = dc.document_id
                    WHERE dc.chat_id = %s
                      AND dc.content_tsv @@ websearch_to_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, chat_id, query, k),
                )
                rows = cur.fetchall()
        return rows

    def count_for_chat(self, chat_id: str) -> int:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) AS n FROM document_chunks WHERE chat_id = %s", (chat_id,)
                )
                return cur.fetchone()["n"]

    def delete_for_document(self, document_id: str) -> int:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                return cur.rowcount

    def delete_for_chat(self, chat_id: str) -> int:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE chat_id = %s", (chat_id,))
                return cur.rowcount
