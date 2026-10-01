"""Free-text generation: final answers, query rewrites, and chat-name suggestions."""
from __future__ import annotations

from evaluation.schemas import QueryRewrite, get_llm, get_structured_llm
from generation.prompts import (
    ANSWER_GENERATOR_SYSTEM_PROMPT,
    CHAT_NAME_SUGGESTION_SYSTEM_PROMPT,
    QUERY_REWRITER_SYSTEM_PROMPT,
)
from retrieval.vector_retriever import RetrievedChunk


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        page = f", page {c.page_number}" if c.page_number else ""
        parts.append(f"[{i}] (source: {c.filename}{page})\n{c.content}")
    return "\n\n".join(parts)


def generate_answer(query: str, chunks: list[RetrievedChunk], chat_history: str = "") -> str:
    llm = get_llm(temperature=0.2, fast=False)
    context_block = build_context_block(chunks)
    history_block = f"\nPrior conversation (for context only):\n{chat_history}\n" if chat_history else ""
    human = (
        f"{history_block}\n"
        f"Retrieved context:\n\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above, citing sources like [1], [2]."
    )
    response = llm.invoke(
        [
            ("system", ANSWER_GENERATOR_SYSTEM_PROMPT),
            ("human", human),
        ]
    )
    return response.content if hasattr(response, "content") else str(response)


def rewrite_query(original_query: str, evaluation_reasoning: str) -> str:
    llm = get_structured_llm(QueryRewrite, temperature=0.3)
    human = (
        f"Original query: {original_query}\n\n"
        f"Why the previous retrieval fell short: {evaluation_reasoning}\n\n"
        "Rewrite the query."
    )
    try:
        result: QueryRewrite = llm.invoke(
            [
                ("system", QUERY_REWRITER_SYSTEM_PROMPT),
                ("human", human),
            ]
        )
        return result.rewritten_query.strip() or original_query
    except Exception:
        return original_query


def suggest_chat_name(filenames: list[str], snippets: list[str] | None = None) -> str:
    if not filenames:
        return "New Chat"
    llm = get_llm(temperature=0.3, fast=True)
    snippet_block = ""
    if snippets:
        joined = "\n---\n".join(s[:300] for s in snippets if s)
        snippet_block = f"\n\nExcerpts:\n{joined}"
    human = f"Filenames: {', '.join(filenames)}{snippet_block}"
    try:
        response = llm.invoke(
            [
                ("system", CHAT_NAME_SUGGESTION_SYSTEM_PROMPT),
                ("human", human),
            ]
        )
        title = (response.content if hasattr(response, "content") else str(response)).strip()
        title = title.strip('"').strip("'")
        return title[:80] if title else _fallback_title(filenames)
    except Exception:
        return _fallback_title(filenames)


def _fallback_title(filenames: list[str]) -> str:
    stems = [f.rsplit(".", 1)[0].replace("_", " ").replace("-", " ") for f in filenames[:2]]
    return " & ".join(stems)[:80] or "New Chat"
