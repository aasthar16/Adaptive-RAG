"""
Context Evaluator.

Scores a set of retrieved chunks against the query using an LLM judge with
structured output, then applies the *configured* thresholds (not the LLM's
own opinion) to classify the result — so the sufficient/partial/poor cut
points are guaranteed consistent and can be tuned in `config/settings.py`
(or via env vars) without touching prompts or retraining anything.
"""
from __future__ import annotations

from config.constants import EvaluationClassification
from config.settings import get_settings
from evaluation.schemas import RetrievalEvaluation, get_structured_llm
from generation.prompts import RETRIEVAL_EVALUATOR_SYSTEM_PROMPT
from retrieval.vector_retriever import RetrievedChunk


def classify(overall_score: float) -> EvaluationClassification:
    settings = get_settings()
    if overall_score >= settings.sufficient_threshold:
        return EvaluationClassification.SUFFICIENT
    if overall_score >= settings.partial_threshold:
        return EvaluationClassification.PARTIAL
    return EvaluationClassification.POOR


def evaluate_retrieval(query: str, chunks: list[RetrievedChunk]) -> RetrievalEvaluation:
    if not chunks:
        return RetrievalEvaluation(
            relevance_score=0.0,
            completeness_score=0.0,
            evidence_score=0.0,
            redundancy_score=1.0,
            overall_score=0.0,
            classification=EvaluationClassification.POOR,
            reasoning="No documents were retrieved.",
        )

    llm = get_structured_llm(RetrievalEvaluation, temperature=0.0)
    numbered = "\n\n".join(
        f"[{i}] (source: {c.filename}, page {c.page_number}, retrieval_score={c.score})\n{c.content[:1000]}"
        for i, c in enumerate(chunks)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Retrieved chunks ({len(chunks)} total):\n\n{numbered}\n\n"
        "Score this retrieval on relevance, completeness, evidence quality, and redundancy."
    )
    try:
        result: RetrievalEvaluation = llm.invoke(
            [
                ("system", RETRIEVAL_EVALUATOR_SYSTEM_PROMPT),
                ("human", prompt),
            ]
        )
    except Exception as exc:
        # If the judge call itself fails (e.g. transient API error), fail
        # closed: treat as poor evidence rather than silently generating an
        # unverified answer, and surface the failure in the reasoning field.
        return RetrievalEvaluation(
            relevance_score=0.0,
            completeness_score=0.0,
            evidence_score=0.0,
            redundancy_score=0.0,
            overall_score=0.0,
            classification=EvaluationClassification.POOR,
            reasoning=f"Retrieval evaluator failed: {exc}",
        )

    # Recompute overall_score deterministically from the sub-scores instead
    # of trusting whatever number the LLM produced for it, then re-classify
    # against the configured thresholds. Weights favor relevance and
    # completeness since those most directly gate hallucination risk.
    overall = round(
        0.35 * result.relevance_score
        + 0.35 * result.completeness_score
        + 0.20 * result.evidence_score
        + 0.10 * result.redundancy_score,
        4,
    )
    result.overall_score = overall
    result.classification = classify(overall)
    return result
