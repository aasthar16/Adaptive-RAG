"""
Answer Verifier.

Checks that every major claim in a generated answer is actually supported
by the retrieved context, using an LLM judge with structured output. This
runs *after* generation and is the last line of defense against
hallucination before a response reaches the user.
"""
from __future__ import annotations

from evaluation.schemas import AnswerVerification, get_structured_llm
from generation.prompts import ANSWER_VERIFIER_SYSTEM_PROMPT
from retrieval.vector_retriever import RetrievedChunk


def verify_answer(query: str, answer: str, chunks: list[RetrievedChunk]) -> AnswerVerification:
    if not chunks:
        return AnswerVerification(
            faithfulness_score=0.0,
            supported=False,
            unsupported_claims=[],
            reasoning="No supporting context was available to verify this answer against.",
        )

    llm = get_structured_llm(AnswerVerification, temperature=0.0)
    context_block = "\n\n".join(
        f"[{i}] (source: {c.filename}, page {c.page_number})\n{c.content[:1000]}"
        for i, c in enumerate(chunks)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Generated answer:\n{answer}\n\n"
        f"Supporting context:\n\n{context_block}\n\n"
        "Check whether every major claim in the answer is supported by the context above."
    )
    try:
        return llm.invoke(
            [
                ("system", ANSWER_VERIFIER_SYSTEM_PROMPT),
                ("human", prompt),
            ]
        )
    except Exception as exc:
        return AnswerVerification(
            faithfulness_score=0.0,
            supported=False,
            unsupported_claims=[],
            reasoning=f"Answer verifier failed: {exc}",
        )
