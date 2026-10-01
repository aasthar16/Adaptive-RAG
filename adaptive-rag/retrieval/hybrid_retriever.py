"""Hybrid retrieval: fuse dense (vector) and lexical (keyword) results.

Uses Reciprocal Rank Fusion (RRF) rather than a naive score average —
cosine-similarity scores and ts_rank scores live on different, incomparable
scales, whereas RRF only needs each list's *rank order*, which makes fusion
robust regardless of how either underlying score is distributed.
"""
from __future__ import annotations

import time
from collections import defaultdict

from config.constants import RetrievalStrategy
from retrieval.keyword_retriever import keyword_search
from retrieval.vector_retriever import RetrievedChunk, vector_search

RRF_K = 60  # standard damping constant from the RRF literature


def hybrid_search(
    query: str, chat_id: str, user_id: str, k: int = 6
) -> tuple[list[RetrievedChunk], float]:
    start = time.perf_counter()
    vector_chunks, _ = vector_search(query, chat_id, user_id, k=max(k, 10))
    keyword_chunks, _ = keyword_search(query, chat_id, user_id, k=max(k, 10))
    latency = time.perf_counter() - start

    rrf_scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(vector_chunks):
        rrf_scores[chunk.chunk_id] += 1.0 / (RRF_K + rank + 1)
        by_id[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(keyword_chunks):
        rrf_scores[chunk.chunk_id] += 1.0 / (RRF_K + rank + 1)
        if chunk.chunk_id not in by_id:
            by_id[chunk.chunk_id] = chunk

    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]

    # Normalize fused RRF scores to [0, 1] for the top-k so they're
    # comparable to single-method scores in the evaluator/UI.
    max_score = max((rrf_scores[cid] for cid in ranked_ids), default=1.0) or 1.0

    fused: list[RetrievedChunk] = []
    for cid in ranked_ids:
        source = by_id[cid]
        fused.append(
            RetrievedChunk(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                content=source.content,
                filename=source.filename,
                page_number=source.page_number,
                score=round(rrf_scores[cid] / max_score, 4),
                retrieval_method=RetrievalStrategy.HYBRID.value,
            )
        )
    return fused, latency
