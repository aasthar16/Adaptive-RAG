"""
The Correction Controller.

This is the heart of AdaptiveRAG's "decision-making architecture": given a
structured retrieval evaluation, the query's characteristics, and what has
already been tried this turn, decide what corrective action to take next.

Deliberately implemented as an explicit, inspectable rule policy over the
evaluator's structured scores (rather than one more free-text LLM call)
so every routing decision is:
  1. explainable — the exact reason is a plain string, shown in the UI's
     "Retrieval Analysis" panel and attached to the LangSmith trace, and
  2. cheap — no extra LLM round trip just to decide what to do next, which
     matters for the project's cost/latency optimization goal.

The scores themselves (relevance/completeness/evidence/redundancy) *are*
produced by an LLM judge (`evaluation/retrieval_evaluator.py`) — this
module only encodes the policy for reacting to them.
"""
from __future__ import annotations

from dataclasses import dataclass

from config.constants import CorrectionAction, RetrievalStrategy
from config.settings import get_settings
from evaluation.schemas import RetrievalEvaluation


@dataclass
class ControllerDecision:
    action: CorrectionAction
    reason: str
    next_strategy: str | None = None  # retrieval strategy to use if retrieving again


def decide_correction_action(
    evaluation: RetrievalEvaluation,
    iteration: int,
    current_strategy: str,
    actions_already_tried: list[str],
    web_search_enabled: bool,
) -> ControllerDecision:
    settings = get_settings()

    if iteration >= settings.max_retrieval_iterations:
        return ControllerDecision(
            action=CorrectionAction.GIVE_UP,
            reason=(
                f"Reached the maximum of {settings.max_retrieval_iterations} retrieval "
                "iterations without sufficient evidence."
            ),
        )

    # Evidence looks broadly right but is arranged/scored poorly -> reorder,
    # don't re-fetch. Cheapest possible correction.
    if (
        evaluation.relevance_score >= 0.55
        and evaluation.evidence_score < 0.6
        and CorrectionAction.RERANK.value not in actions_already_tried
    ):
        return ControllerDecision(
            action=CorrectionAction.RERANK,
            reason=(
                f"Relevant material appears present (relevance={evaluation.relevance_score:.2f}) "
                f"but evidence quality/ordering is weak (evidence={evaluation.evidence_score:.2f}); "
                "reranking the existing candidates before re-retrieving."
            ),
        )

    # Retrieved chunks are relevant and non-redundant but don't cover the
    # whole question -> pull more, broaden to hybrid if not already.
    if evaluation.completeness_score < 0.6 and evaluation.relevance_score >= 0.5:
        next_strategy = (
            RetrievalStrategy.HYBRID.value
            if current_strategy != RetrievalStrategy.HYBRID.value
            else current_strategy
        )
        return ControllerDecision(
            action=CorrectionAction.ADDITIONAL_RETRIEVAL,
            reason=(
                f"Relevant but incomplete evidence (completeness={evaluation.completeness_score:.2f}); "
                f"retrieving additional context with '{next_strategy}' strategy."
            ),
            next_strategy=next_strategy,
        )

    # Low relevance suggests the query itself (not the retriever) is the
    # problem -> rewrite it, once.
    if (
        evaluation.relevance_score < 0.5
        and CorrectionAction.REWRITE_QUERY.value not in actions_already_tried
    ):
        return ControllerDecision(
            action=CorrectionAction.REWRITE_QUERY,
            reason=(
                f"Low relevance ({evaluation.relevance_score:.2f}) across retrieved chunks "
                "suggests the query is ambiguous or poorly formed; rewriting it before "
                "retrying retrieval."
            ),
            next_strategy=current_strategy,
        )

    # Local knowledge base genuinely doesn't seem to have this -> go external
    # if allowed and not already tried.
    if web_search_enabled and CorrectionAction.EXTERNAL_SEARCH.value not in actions_already_tried:
        return ControllerDecision(
            action=CorrectionAction.EXTERNAL_SEARCH,
            reason=(
                "Local document knowledge base still insufficient after prior corrections; "
                "escalating to external web search."
            ),
        )

    # Out of good options but iterations remain and we haven't tried a
    # broader retrieval yet — one last additional-retrieval attempt with
    # hybrid before giving up.
    if CorrectionAction.ADDITIONAL_RETRIEVAL.value not in actions_already_tried:
        return ControllerDecision(
            action=CorrectionAction.ADDITIONAL_RETRIEVAL,
            reason="Trying one broader hybrid retrieval pass before giving up.",
            next_strategy=RetrievalStrategy.HYBRID.value,
        )

    return ControllerDecision(
        action=CorrectionAction.GIVE_UP,
        reason="Exhausted available corrective actions without reaching sufficient evidence.",
    )
