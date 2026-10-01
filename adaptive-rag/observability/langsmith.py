"""
LangSmith observability.

LangSmith tracing in LangChain/LangGraph is driven entirely by environment
variables, so `init_langsmith()` just makes sure those are set consistently
from our own `Settings` object (so there's one source of truth in `.env`),
and never unconditionally requires LangSmith to be reachable — if it's
misconfigured or the service is down, tracing silently no-ops and the rest
of the application keeps working (see section 43, "LangSmith failure
should NOT make the core application unusable").

`run_config()` builds the per-invocation `config` dict passed to
`graph.invoke(...)`, attaching exactly the metadata called for in section
25 of the spec: user_id, chat_id, query_id, retrieval_strategy,
retrieval_iteration, evaluation_score, correction_action — and nothing
that could be sensitive (no passwords, no raw auth tokens, ever).
"""
from __future__ import annotations

import logging
import os

from config.settings import get_settings

logger = logging.getLogger(__name__)

_initialized = False


def init_langsmith() -> None:
    global _initialized
    if _initialized:
        return
    settings = get_settings()

    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        # Older LangChain versions look for the LANGCHAIN_* names; set both
        # so tracing works regardless of exactly which version resolves.
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        logger.info("LangSmith tracing enabled for project '%s'.", settings.langsmith_project)
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled.")

    _initialized = True


def run_config(
    *,
    user_id: str,
    chat_id: str,
    query_id: str,
    thread_id: str,
    retrieval_strategy: str | None = None,
    retrieval_iteration: int = 0,
    evaluation_score: float | None = None,
    correction_action: str | None = None,
    recursion_limit: int = 25,
) -> dict:
    """Build the `config=` dict passed to `graph.invoke`/`graph.stream`.

    `configurable.thread_id` is what LangGraph's Postgres checkpointer uses
    to key persisted state (see graph/workflow.py); `metadata`/`tags` are
    what surface in the LangSmith UI for filtering and debugging runs.
    """
    metadata = {
        "user_id": user_id,
        "chat_id": chat_id,
        "query_id": query_id,
    }
    if retrieval_strategy is not None:
        metadata["retrieval_strategy"] = retrieval_strategy
    if evaluation_score is not None:
        metadata["evaluation_score"] = evaluation_score
    if correction_action is not None:
        metadata["correction_action"] = correction_action
    metadata["retrieval_iteration"] = retrieval_iteration

    return {
        "configurable": {"thread_id": thread_id},
        "metadata": metadata,
        "tags": ["adaptive-rag", f"chat:{chat_id}"],
        "recursion_limit": recursion_limit,
    }
