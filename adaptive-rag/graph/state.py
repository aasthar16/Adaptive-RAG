"""
The LangGraph state for AdaptiveRAG.

This intentionally goes beyond the minimal state sketched in the original
spec (query/context/answer/iteration) in a few ways, because the project's
whole point is *explainable, cost-aware adaptive decision-making* — which
means the state itself needs to carry a legible trace of what happened, not
just the final answer:

1. `original_query` vs `current_query` are kept separate. The original is
   what gets shown/logged/verified conceptually as "the question", while
   `current_query` is whatever the Query Rewriter most recently produced —
   keeping them distinct means a rewrite never silently overwrites the
   user's actual question.

2. `retrieval_history`, `evaluation_history`, and `correction_history` are
   `Annotated[..., operator.add]` reducer lists rather than single "latest
   result" fields. LangGraph reducers mean every node can just return its
   own new event and LangGraph appends it — nodes never need to read-then-
   append the full history themselves. This is exactly what powers the
   "🔍 Retrieval Analysis" debug panel (section 26) and gives the
   Correction Controller (`retrieval/controller.py`) the
   `actions_already_tried` it needs to avoid repeating a failing action.

3. Cost/latency counters (`llm_call_count`, `retrieval_call_count`,
   `total_tokens_estimate`) are also reducer-accumulated ints, satisfying
   section 27's requirement to track this per-query without every node
   needing to know the running total.

4. `node_trace` is a reducer list of node names visited, purely for
   observability — a cheap, human-readable path trace independent of
   whatever LangSmith shows, useful when tracing is disabled.

The state is a `TypedDict` (not a Pydantic model) because that is what
LangGraph's `StateGraph` and its Postgres checkpointer serialize most
directly; validation of individual fields is handled by the Pydantic
schemas in `evaluation/schemas.py` at the point each field is produced.
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Optional, TypedDict


class RetrievedChunkDict(TypedDict):
    chunk_id: str
    document_id: str
    content: str
    filename: str
    page_number: Optional[int]
    score: float
    retrieval_method: str


class RetrievalEvent(TypedDict):
    """One entry in `retrieval_history` — a full record of a single
    retrieval call, kept even after later corrections replace
    `retrieved_documents`, so the debug panel can show the whole path."""

    iteration: int
    strategy: str
    query_used: str
    num_results: int
    latency_seconds: float


class EvaluationEvent(TypedDict):
    iteration: int
    relevance_score: float
    completeness_score: float
    evidence_score: float
    redundancy_score: float
    overall_score: float
    classification: str
    reasoning: str


class CorrectionEvent(TypedDict):
    iteration: int
    action: str
    reason: str


class SourceRef(TypedDict):
    index: int
    filename: str
    page_number: Optional[int]
    chunk_id: str


class RAGState(TypedDict, total=False):
    # --- identity / session -------------------------------------------
    user_id: str
    chat_id: str
    thread_id: str
    query_id: str

    # --- query -----------------------------------------------------------
    original_query: str
    current_query: str
    query_type: Optional[str]
    complexity: Optional[str]
    requires_multiple_sources: Optional[bool]
    recommended_strategy: Optional[str]
    chat_history_text: str  # short rendered summary of prior turns, for context

    # --- retrieval ---------------------------------------------------------
    retrieval_strategy: Optional[str]
    retrieved_documents: list[RetrievedChunkDict]
    retrieval_history: Annotated[list[RetrievalEvent], operator.add]

    # --- evaluation --------------------------------------------------------
    evaluation_result: Optional[EvaluationEvent]
    evaluation_history: Annotated[list[EvaluationEvent], operator.add]

    # --- correction ------------------------------------------------------
    correction_action: Optional[str]
    correction_history: Annotated[list[CorrectionEvent], operator.add]
    actions_already_tried: Annotated[list[str], operator.add]
    iteration: Annotated[int, operator.add]
    max_iterations: int

    # --- answer ------------------------------------------------------------
    answer: Optional[str]
    sources: list[SourceRef]
    verification_result: Optional[dict[str, Any]]
    verification_retries: int
    max_verification_retries: int

    # --- cost / performance tracking (section 27) ---------------------
    llm_call_count: Annotated[int, operator.add]
    retrieval_call_count: Annotated[int, operator.add]
    total_tokens_estimate: Annotated[int, operator.add]
    started_at: float
    node_trace: Annotated[list[str], operator.add]

    # --- control -------------------------------------------------------
    web_search_enabled: bool
    error: Optional[str]
    give_up: bool


def new_initial_state(
    *,
    user_id: str,
    chat_id: str,
    thread_id: str,
    query_id: str,
    query: str,
    chat_history_text: str,
    max_iterations: int,
    max_verification_retries: int,
    web_search_enabled: bool,
) -> RAGState:
    return RAGState(
        user_id=user_id,
        chat_id=chat_id,
        thread_id=thread_id,
        query_id=query_id,
        original_query=query,
        current_query=query,
        query_type=None,
        complexity=None,
        requires_multiple_sources=None,
        recommended_strategy=None,
        chat_history_text=chat_history_text,
        retrieval_strategy=None,
        retrieved_documents=[],
        retrieval_history=[],
        evaluation_result=None,
        evaluation_history=[],
        correction_action=None,
        correction_history=[],
        actions_already_tried=[],
        iteration=0,
        max_iterations=max_iterations,
        answer=None,
        sources=[],
        verification_result=None,
        verification_retries=0,
        max_verification_retries=max_verification_retries,
        llm_call_count=0,
        retrieval_call_count=0,
        total_tokens_estimate=0,
        started_at=time.time(),
        node_trace=[],
        web_search_enabled=web_search_enabled,
        error=None,
        give_up=False,
    )
