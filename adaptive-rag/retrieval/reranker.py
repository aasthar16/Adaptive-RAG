"""LLM-based listwise reranker.

Used by the Correction Controller when retrieval found broadly the right
documents but in a poor order (e.g. the single most relevant chunk was
returned 6th out of 8). Re-scores each candidate's relevance to the query
with one structured-output LLM call and returns the top N by that score —
cheaper than an extra retrieval round trip, and it doesn't need a change to
the query itself.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from evaluation.schemas import get_structured_llm
from generation.prompts import RERANK_SYSTEM_PROMPT
from retrieval.vector_retriever import RetrievedChunk


class ChunkRelevance(BaseModel):
    chunk_index: int = Field(description="0-based index of the chunk in the provided list.")
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="How relevant this chunk is to answering the query."
    )


class RerankResult(BaseModel):
    ranked: list[ChunkRelevance] = Field(
        description="All chunks, ordered most to least relevant, each with a relevance score."
    )


def rerank(query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    if not chunks:
        return []
    if len(chunks) <= top_n:
        # Nothing to trim; still worth reordering by fresh relevance below.
        pass

    llm = get_structured_llm(RerankResult, temperature=0.0)
    numbered = "\n\n".join(
        f"[{i}] (source: {c.filename}, page {c.page_number})\n{c.content[:800]}"
        for i, c in enumerate(chunks)
    )
    try:
        result: RerankResult = llm.invoke(
            [
                ("system", RERANK_SYSTEM_PROMPT),
                ("human", f"Query: {query}\n\nCandidate chunks:\n\n{numbered}"),
            ]
        )
    except Exception:
        # Reranking is a quality-of-life correction step, not load-bearing —
        # if the LLM call fails, fall back to the original ordering rather
        # than breaking the whole answer.
        return chunks[:top_n]

    scored = {r.chunk_index: r.relevance_score for r in result.ranked if 0 <= r.chunk_index < len(chunks)}
    ordered = sorted(range(len(chunks)), key=lambda i: scored.get(i, 0.0), reverse=True)

    reranked: list[RetrievedChunk] = []
    for i in ordered[:top_n]:
        original = chunks[i]
        reranked.append(
            RetrievedChunk(
                chunk_id=original.chunk_id,
                document_id=original.document_id,
                content=original.content,
                filename=original.filename,
                page_number=original.page_number,
                score=round(scored.get(i, original.score), 4),
                retrieval_method=f"{original.retrieval_method}+rerank",
            )
        )
    return reranked
