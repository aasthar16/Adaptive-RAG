"""
Conditional edge functions for the AdaptiveRAG graph.

Each function reads only `RAGState` and returns the string name of the
next node, matching the LangGraph `add_conditional_edges` contract. Kept
separate from `nodes.py` so the *routing policy* (what happens next) is
easy to read in one place, distinct from the *node logic* (what each step
actually computes).
"""
from __future__ import annotations

from config.constants import CorrectionAction, EvaluationClassification
from config.settings import get_settings
from graph.state import RAGState

# Node name constants (kept in one place so typos become import errors,
# not silent mis-routing).
ANALYZE_QUERY = "analyze_query"
SELECT_STRATEGY = "select_retrieval_strategy"
RETRIEVE_DOCUMENTS = "retrieve_documents"
EVALUATE_RETRIEVAL = "evaluate_retrieval"
DECIDE_NEXT_ACTION = "decide_next_action"
RERANK_DOCUMENTS = "rerank_documents"
REWRITE_QUERY = "rewrite_query_node"
EXTERNAL_SEARCH = "external_search_node"
GENERATE_ANSWER = "generate_answer_node"
VERIFY_ANSWER = "verify_answer_node"
PREPARE_REGENERATION = "prepare_for_regeneration"


def route_after_evaluation(state: RAGState) -> str:
    """After scoring retrieval quality: go straight to generation if it's
    already sufficient (or we've been told to give up), otherwise hand off
    to the Correction Controller. This same function is reused for BOTH the
    first evaluation and every re-evaluation after a correction — sufficiency
    is sufficiency regardless of which loop produced it.
    """
    if state.get("give_up"):
        return GENERATE_ANSWER
    evaluation = state.get("evaluation_result") or {}
    if evaluation.get("classification") == EvaluationClassification.SUFFICIENT.value:
        return GENERATE_ANSWER
    return DECIDE_NEXT_ACTION


def route_after_decision(state: RAGState) -> str:
    """Dispatch to whichever corrective action the controller chose."""
    action = state.get("correction_action")
    if action == CorrectionAction.RERANK.value:
        return RERANK_DOCUMENTS
    if action == CorrectionAction.REWRITE_QUERY.value:
        return REWRITE_QUERY
    if action == CorrectionAction.ADDITIONAL_RETRIEVAL.value:
        return RETRIEVE_DOCUMENTS
    if action == CorrectionAction.EXTERNAL_SEARCH.value:
        return EXTERNAL_SEARCH
    # GIVE_UP (or anything unrecognized, defensively)
    return GENERATE_ANSWER


def route_after_verification(state: RAGState) -> str:
    """After the faithfulness check: end if the answer is well-supported,
    if we've already given up, or if we're out of verification retries;
    otherwise loop back for one more evidence-gathering + regeneration pass.
    """
    settings = get_settings()
    if state.get("give_up"):
        return "end"
    verification = state.get("verification_result") or {}
    if verification.get("supported", True):
        return "end"
    if state.get("verification_retries", 0) >= settings.max_verification_retries:
        return "end"
    return PREPARE_REGENERATION
