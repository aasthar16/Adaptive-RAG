"""Lexical retrieval — PostgreSQL full-text search (BM25-style ranking via
`ts_rank`), scoped to a single chat's document set.

Chosen over an in-process `rank_bm25` index because the corpus lives in
Postgres already; running FTS there avoids pulling every chunk for a chat
into Python memory on every query and keeps the `chat_id` filter enforced
by the same database layer as everything else (see
`database/repositories/chunk_repository.py`).
"""
from __future__ import annotations

import time

from config.constants import RetrievalStrategy
from database.repositories.chunk_repository import ChunkRepository
from retrieval.vector_retriever import RetrievedChunk

_chunks = ChunkRepository()


def keyword_search(
    query: str, chat_id: str, user_id: str, k: int = 6
) -> tuple[list[RetrievedChunk], float]:
    start = time.perf_counter()
    rows = _chunks.keyword_search(chat_id=chat_id, query=query, k=k)
    latency = time.perf_counter() - start

    chunks: list[RetrievedChunk] = []
    for row in rows:
        meta = row.get("metadata") or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=str(row["id"]),
                document_id=str(row["document_id"]),
                content=row["content"],
                filename=row.get("filename") or meta.get("filename", "unknown"),
                page_number=meta.get("page_number"),
                # ts_rank is unbounded above zero; squash into ~[0,1] with a
                # simple saturating curve so it's comparable to cosine scores.
                score=round(min(1.0, float(row["score"]) * 4.0), 4),
                retrieval_method=RetrievalStrategy.KEYWORD.value,
            )
        )
    return chunks, latency
