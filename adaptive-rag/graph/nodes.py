"""
LangGraph nodes for the AdaptiveRAG workflow.

Every node is a plain function `(RAGState) -> dict` that returns only the
keys it changed — LangGraph merges those into the persisted state (using
the `operator.add` reducers declared in `graph/state.py` for history/
counter fields, and plain overwrite for everything else). Keeping nodes as
pure functions over the state dict (rather than a class with hidden
instance state) is what makes each step independently unit-testable (see
`tests/test_graph.py`) and keeps LangSmith's per-node trace legible.
"""
from __future__ import annotations

import logging

from config.constants import (
    CHARS_PER_TOKEN_ESTIMATE,
    CorrectionAction,
    RetrievalStrategy,
)
from config.settings import get_settings
from evaluation.answer_evaluator import verify_answer as run_answer_verification
from evaluation.retrieval_evaluator import evaluate_retrieval as run_retrieval_evaluation
from evaluation.schemas import QueryAnalysis, get_structured_llm
from generation.generator import generate_answer as run_answer_generation
from generation.generator import rewrite_query as run_query_rewrite
from generation.prompts import INSUFFICIENT_EVIDENCE_MESSAGE, QUERY_ANALYZER_SYSTEM_PROMPT
from graph.state import RAGState
from retrieval.controller import decide_correction_action
from retrieval.hybrid_retriever import hybrid_search
from retrieval.keyword_retriever import keyword_search
from retrieval.reranker import rerank as run_rerank
from retrieval.vector_retriever import RetrievedChunk, vector_search
from retrieval.web_retriever import WebSearchUnavailableError, web_search

logger = logging.getLogger(__name__)


def _estimate_tokens(*texts: str) -> int:
    total_chars = sum(len(t) for t in texts if t)
    return max(1, total_chars // CHARS_PER_TOKEN_ESTIMATE)


def _to_dicts(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "content": c.content,
            "filename": c.filename,
            "page_number": c.page_number,
            "score": c.score,
            "retrieval_method": c.retrieval_method,
        }
        for c in chunks
    ]


def _from_dicts(dicts: list[dict]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=d["chunk_id"],
            document_id=d["document_id"],
            content=d["content"],
            filename=d["filename"],
            page_number=d.get("page_number"),
            score=d["score"],
            retrieval_method=d["retrieval_method"],
        )
        for d in dicts
    ]


def _merge_unique(existing: list[RetrievedChunk], new: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen = {c.chunk_id for c in existing}
    merged = list(existing)
    for c in new:
        if c.chunk_id not in seen:
            merged.append(c)
            seen.add(c.chunk_id)
    return merged


# =========================================================================
# 1. Query Analyzer
# =========================================================================
def analyze_query(state: RAGState) -> dict:
    llm = get_structured_llm(QueryAnalysis, temperature=0.0)
    try:
        analysis: QueryAnalysis = llm.invoke(
            [
                ("system", QUERY_ANALYZER_SYSTEM_PROMPT),
                ("human", state["current_query"]),
            ]
        )
        update = {
            "query_type": analysis.query_type.value,
            "complexity": analysis.complexity.value,
            "requires_multiple_sources": analysis.requires_multiple_sources,
            "recommended_strategy": analysis.recommended_strategy,
        }
    except Exception as exc:
        logger.warning("Query analysis failed, defaulting to hybrid: %s", exc)
        update = {
            "query_type": "factual",
            "complexity": "medium",
            "requires_multiple_sources": False,
            "recommended_strategy": RetrievalStrategy.HYBRID.value,
        }

    update["llm_call_count"] = 1
    update["total_tokens_estimate"] = _estimate_tokens(state["current_query"])
    update["node_trace"] = ["analyze_query"]
    return update


# =========================================================================
# 2. Retrieval Strategy Selector
# =========================================================================
def select_retrieval_strategy(state: RAGState) -> dict:
    strategy = state.get("recommended_strategy") or RetrievalStrategy.VECTOR.value
    if strategy not in {s.value for s in RetrievalStrategy if s != RetrievalStrategy.WEB}:
        strategy = RetrievalStrategy.HYBRID.value
    # Multi-source queries benefit from hybrid retrieval's broader recall
    # even if the analyzer suggested plain vector search.
    if state.get("requires_multiple_sources") and strategy == RetrievalStrategy.VECTOR.value:
        strategy = RetrievalStrategy.HYBRID.value
    return {"retrieval_strategy": strategy, "node_trace": ["select_retrieval_strategy"]}


# =========================================================================
# 3. Retrieve Documents (also handles "additional retrieval" corrections)
# =========================================================================
def retrieve_documents(state: RAGState) -> dict:
    settings = get_settings()
    strategy = state.get("retrieval_strategy") or RetrievalStrategy.VECTOR.value
    query = state["current_query"]
    is_additional = state.get("correction_action") == CorrectionAction.ADDITIONAL_RETRIEVAL.value
    iteration = state.get("iteration", 0) + 1

    try:
        if strategy == RetrievalStrategy.KEYWORD.value:
            chunks, latency = keyword_search(query, state["chat_id"], state["user_id"], k=settings.default_top_k)
        elif strategy == RetrievalStrategy.HYBRID.value:
            chunks, latency = hybrid_search(query, state["chat_id"], state["user_id"], k=settings.default_top_k)
        else:
            chunks, latency = vector_search(query, state["chat_id"], state["user_id"], k=settings.default_top_k)
    except Exception as exc:
        logger.error("Retrieval failed (strategy=%s): %s", strategy, exc)
        chunks, latency = [], 0.0

    existing = _from_dicts(state.get("retrieved_documents", [])) if is_additional else []
    merged = _merge_unique(existing, chunks) if is_additional else chunks

    event = {
        "iteration": iteration,
        "strategy": strategy,
        "query_used": query,
        "num_results": len(chunks),
        "latency_seconds": round(latency, 4),
    }

    return {
        "retrieved_documents": _to_dicts(merged),
        "retrieval_history": [event],
        "retrieval_call_count": 1,
        "iteration": 1,  # reducer-added: bumps the running iteration counter by 1
        "node_trace": ["retrieve_documents"],
    }


# =========================================================================
# 4. Context Evaluator
# =========================================================================
def evaluate_retrieval(state: RAGState) -> dict:
    chunks = _from_dicts(state.get("retrieved_documents", []))
    evaluation = run_retrieval_evaluation(state["current_query"], chunks)

    event = {
        "iteration": state.get("iteration", 0),
        "relevance_score": evaluation.relevance_score,
        "completeness_score": evaluation.completeness_score,
        "evidence_score": evaluation.evidence_score,
        "redundancy_score": evaluation.redundancy_score,
        "overall_score": evaluation.overall_score,
        "classification": evaluation.classification.value,
        "reasoning": evaluation.reasoning,
    }

    return {
        "evaluation_result": event,
        "evaluation_history": [event],
        "llm_call_count": 1,
        "node_trace": ["evaluate_retrieval"],
    }


# =========================================================================
# 5. Correction Controller ("decide_next_action")
# =========================================================================
def decide_next_action(state: RAGState) -> dict:
    from evaluation.schemas import RetrievalEvaluation
    from config.constants import EvaluationClassification as EC

    eval_event = state["evaluation_result"]
    evaluation = RetrievalEvaluation(
        relevance_score=eval_event["relevance_score"],
        completeness_score=eval_event["completeness_score"],
        evidence_score=eval_event["evidence_score"],
        redundancy_score=eval_event["redundancy_score"],
        overall_score=eval_event["overall_score"],
        classification=EC(eval_event["classification"]),
        reasoning=eval_event["reasoning"],
    )

    decision = decide_correction_action(
        evaluation=evaluation,
        iteration=state.get("iteration", 0),
        current_strategy=state.get("retrieval_strategy", RetrievalStrategy.VECTOR.value),
        actions_already_tried=state.get("actions_already_tried", []),
        web_search_enabled=state.get("web_search_enabled", False),
    )

    update: dict = {
        "correction_action": decision.action.value,
        "correction_history": [
            {
                "iteration": state.get("iteration", 0),
                "action": decision.action.value,
                "reason": decision.reason,
            }
        ],
        "actions_already_tried": [decision.action.value],
        "node_trace": ["decide_next_action"],
    }
    if decision.next_strategy:
        update["retrieval_strategy"] = decision.next_strategy
    if decision.action == CorrectionAction.GIVE_UP:
        update["give_up"] = True
    return update


# =========================================================================
# 6a. Reranker
# =========================================================================
def rerank_documents(state: RAGState) -> dict:
    settings = get_settings()
    chunks = _from_dicts(state.get("retrieved_documents", []))
    reranked = run_rerank(state["current_query"], chunks, top_n=settings.rerank_top_n)
    return {
        "retrieved_documents": _to_dicts(reranked),
        "llm_call_count": 1,
        "node_trace": ["rerank_documents"],
    }


# =========================================================================
# 6b. Query Rewriter
# =========================================================================
def rewrite_query_node(state: RAGState) -> dict:
    reasoning = (state.get("evaluation_result") or {}).get("reasoning", "")
    new_query = run_query_rewrite(state["original_query"], reasoning)
    return {
        "current_query": new_query,
        "llm_call_count": 1,
        "node_trace": ["rewrite_query"],
    }


# =========================================================================
# 6c. External Search
# =========================================================================
def external_search_node(state: RAGState) -> dict:
    existing = _from_dicts(state.get("retrieved_documents", []))
    try:
        web_chunks, latency = web_search(state["current_query"], k=5)
        merged = _merge_unique(existing, web_chunks)
        event = {
            "iteration": state.get("iteration", 0) + 1,
            "strategy": RetrievalStrategy.WEB.value,
            "query_used": state["current_query"],
            "num_results": len(web_chunks),
            "latency_seconds": round(latency, 4),
        }
        return {
            "retrieved_documents": _to_dicts(merged),
            "retrieval_history": [event],
            "retrieval_call_count": 1,
            "iteration": 1,
            "node_trace": ["external_search"],
        }
    except WebSearchUnavailableError as exc:
        logger.info("External search unavailable: %s", exc)
        # Degrade gracefully: leave retrieved_documents untouched and let
        # the controller's next decision (on re-evaluation) route to
        # give_up rather than crash the graph.
        return {"iteration": 1, "node_trace": ["external_search_skipped"]}


# =========================================================================
# 7. Answer Generation
# =========================================================================
def generate_answer_node(state: RAGState) -> dict:
    chunks = _from_dicts(state.get("retrieved_documents", []))
    if not chunks or state.get("give_up"):
        return {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
            "node_trace": ["generate_answer_give_up"],
        }

    answer = run_answer_generation(
        state["original_query"], chunks, chat_history=state.get("chat_history_text", "")
    )
    sources = [
        {
            "index": i + 1,
            "filename": c.filename,
            "page_number": c.page_number,
            "chunk_id": c.chunk_id,
        }
        for i, c in enumerate(chunks)
    ]
    return {
        "answer": answer,
        "sources": sources,
        "llm_call_count": 1,
        "total_tokens_estimate": _estimate_tokens(answer),
        "node_trace": ["generate_answer"],
    }


# =========================================================================
# 8. Answer Verification
# =========================================================================
def verify_answer_node(state: RAGState) -> dict:
    chunks = _from_dicts(state.get("retrieved_documents", []))
    answer = state.get("answer") or ""
    if state.get("give_up"):
        return {"node_trace": ["verify_answer_skipped"]}

    verification = run_answer_verification(state["original_query"], answer, chunks)
    return {
        "verification_result": {
            "faithfulness_score": verification.faithfulness_score,
            "supported": verification.supported,
            "unsupported_claims": [c.model_dump() for c in verification.unsupported_claims],
            "reasoning": verification.reasoning,
        },
        "llm_call_count": 1,
        "node_trace": ["verify_answer"],
    }


def prepare_for_regeneration(state: RAGState) -> dict:
    """Bump the verification retry counter before looping back for more
    evidence + a fresh generation attempt."""
    return {
        "verification_retries": state.get("verification_retries", 0) + 1,
        "correction_action": CorrectionAction.ADDITIONAL_RETRIEVAL.value,
        "node_trace": ["prepare_for_regeneration"],
    }
