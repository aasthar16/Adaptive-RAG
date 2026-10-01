"""
Dense vector retrieval against PostgreSQL + pgvector via `langchain-postgres`.

The same `document_chunks` table (created explicitly in
`database/migrations/001_init_schema.sql`) is used both for writes during
ingestion (see `ingestion/pipeline.py`) and for the similarity search here,
so there is exactly one physical store for chunk data — no duplicated
vector storage.

Every query is filtered by `chat_id` (and `user_id` as defense in depth) at
the database layer via `PGVectorStore`'s `filter` argument, which compiles
down to a SQL `WHERE` clause — this is what actually enforces per-chat
knowledge-base isolation, not anything in the UI.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from langchain_postgres import PGEngine, PGVectorStore

from config.constants import VECTOR_TABLE_NAME, RetrievalStrategy
from config.settings import get_settings
from ingestion.embeddings import get_embeddings


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    content: str
    filename: str
    page_number: int | None
    score: float
    retrieval_method: str


@lru_cache(maxsize=1)
def _get_engine() -> PGEngine:
    settings = get_settings()
    return PGEngine.from_connection_string(settings.database_url_sqlalchemy)


@lru_cache(maxsize=1)
def get_vector_store() -> PGVectorStore:
    settings = get_settings()
    engine = _get_engine()
    return PGVectorStore.create_sync(
        engine=engine,
        embedding_service=get_embeddings(),
        table_name=VECTOR_TABLE_NAME,
        content_column="content",
        embedding_column="embedding",
        metadata_json_column="metadata",
        id_column="id",
        metadata_columns=["user_id", "chat_id", "document_id", "chunk_index"],
        k=settings.default_top_k,
    )


def vector_search(
    query: str, chat_id: str, user_id: str, k: int | None = None
) -> tuple[list[RetrievedChunk], float]:
    """Dense semantic search scoped to a single chat's knowledge base.

    Returns (chunks, latency_seconds).
    """
    settings = get_settings()
    store = get_vector_store()
    k = k or settings.default_top_k

    start = time.perf_counter()
    results = store.similarity_search_with_score(
        query,
        k=k,
        filter={"chat_id": chat_id, "user_id": user_id},
    )
    latency = time.perf_counter() - start

    chunks: list[RetrievedChunk] = []
    for doc, distance in results:
        # PGVectorStore's default distance_strategy is cosine distance in
        # [0, 2]; convert to an intuitive similarity score in [0, 1].
        similarity = max(0.0, 1.0 - (distance / 2.0))
        meta = doc.metadata or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=str(doc.id or meta.get("id") or ""),
                document_id=str(meta.get("document_id", "")),
                content=doc.page_content,
                filename=meta.get("filename", "unknown"),
                page_number=meta.get("page_number"),
                score=round(similarity, 4),
                retrieval_method=RetrievalStrategy.VECTOR.value,
            )
        )
    return chunks, latency
