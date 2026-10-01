"""
Assembles the AdaptiveRAG LangGraph workflow (section 4 / section 42 of the
spec) and exposes a single `run_query(...)` entrypoint used by
`services/chat_service.py`.

Graph shape:

    analyze_query -> select_retrieval_strategy -> retrieve_documents
        -> evaluate_retrieval
             -> [sufficient]  -> generate_answer_node
             -> [insufficient] -> decide_next_action
                    -> rerank_documents -----------------> evaluate_retrieval
                    -> rewrite_query_node -> retrieve_documents -> evaluate_retrieval
                    -> retrieve_documents (additional) -----------> evaluate_retrieval
                    -> external_search_node -----------------------> evaluate_retrieval
                    -> [give_up] -> generate_answer_node
    generate_answer_node -> verify_answer_node
        -> [supported / give_up / out of retries] -> END
        -> [unsupported]  -> prepare_for_regeneration -> retrieve_documents -> evaluate_retrieval (loop)

Every one of those branches is an explicit LangGraph node/conditional edge
(see `graph/nodes.py` and `graph/edges.py`) rather than nested if/else, so
each hop is independently visible as a span in LangSmith.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from config.settings import get_settings
from graph import edges
from graph import nodes as N
from graph.state import RAGState, new_initial_state
from observability.langsmith import init_langsmith, run_config

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node(edges.ANALYZE_QUERY, N.analyze_query)
    graph.add_node(edges.SELECT_STRATEGY, N.select_retrieval_strategy)
    graph.add_node(edges.RETRIEVE_DOCUMENTS, N.retrieve_documents)
    graph.add_node(edges.EVALUATE_RETRIEVAL, N.evaluate_retrieval)
    graph.add_node(edges.DECIDE_NEXT_ACTION, N.decide_next_action)
    graph.add_node(edges.RERANK_DOCUMENTS, N.rerank_documents)
    graph.add_node(edges.REWRITE_QUERY, N.rewrite_query_node)
    graph.add_node(edges.EXTERNAL_SEARCH, N.external_search_node)
    graph.add_node(edges.GENERATE_ANSWER, N.generate_answer_node)
    graph.add_node(edges.VERIFY_ANSWER, N.verify_answer_node)
    graph.add_node(edges.PREPARE_REGENERATION, N.prepare_for_regeneration)

    graph.add_edge(START, edges.ANALYZE_QUERY)
    graph.add_edge(edges.ANALYZE_QUERY, edges.SELECT_STRATEGY)
    graph.add_edge(edges.SELECT_STRATEGY, edges.RETRIEVE_DOCUMENTS)
    graph.add_edge(edges.RETRIEVE_DOCUMENTS, edges.EVALUATE_RETRIEVAL)

    graph.add_conditional_edges(
        edges.EVALUATE_RETRIEVAL,
        edges.route_after_evaluation,
        {
            edges.GENERATE_ANSWER: edges.GENERATE_ANSWER,
            edges.DECIDE_NEXT_ACTION: edges.DECIDE_NEXT_ACTION,
        },
    )

    graph.add_conditional_edges(
        edges.DECIDE_NEXT_ACTION,
        edges.route_after_decision,
        {
            edges.RERANK_DOCUMENTS: edges.RERANK_DOCUMENTS,
            edges.REWRITE_QUERY: edges.REWRITE_QUERY,
            edges.RETRIEVE_DOCUMENTS: edges.RETRIEVE_DOCUMENTS,
            edges.EXTERNAL_SEARCH: edges.EXTERNAL_SEARCH,
            edges.GENERATE_ANSWER: edges.GENERATE_ANSWER,
        },
    )

    # Every corrective action loops back to re-evaluation except the query
    # rewrite, which must retrieve fresh evidence for the new query first.
    graph.add_edge(edges.RERANK_DOCUMENTS, edges.EVALUATE_RETRIEVAL)
    graph.add_edge(edges.REWRITE_QUERY, edges.RETRIEVE_DOCUMENTS)
    graph.add_edge(edges.EXTERNAL_SEARCH, edges.EVALUATE_RETRIEVAL)

    graph.add_edge(edges.GENERATE_ANSWER, edges.VERIFY_ANSWER)

    graph.add_conditional_edges(
        edges.VERIFY_ANSWER,
        edges.route_after_verification,
        {
            "end": END,
            edges.PREPARE_REGENERATION: edges.PREPARE_REGENERATION,
        },
    )
    graph.add_edge(edges.PREPARE_REGENERATION, edges.RETRIEVE_DOCUMENTS)

    return graph


@contextmanager
def _checkpointer() -> Iterator[PostgresSaver]:
    settings = get_settings()
    with PostgresSaver.from_conn_string(settings.database_url_psycopg) as saver:
        yield saver


@lru_cache(maxsize=1)
def _compiled_graph_no_checkpoint() -> CompiledStateGraph:
    """A checkpointer-less compiled graph, used only by unit tests that
    exercise the graph end-to-end without a database (see test_graph.py)."""
    return build_graph().compile()


def run_query(
    *,
    user_id: str,
    chat_id: str,
    query: str,
    chat_history_text: str = "",
) -> RAGState:
    """Run one full turn of the AdaptiveRAG workflow for a chat message.

    Persists LangGraph state under a stable per-chat `thread_id`
    (`user_{user_id}_chat_{chat_id}`, per section 8), so the conversational
    graph state survives Streamlit reruns/process restarts. A fresh
    `thread_id` per *query* rather than per *chat* would be simpler, but
    sharing one thread_id across every message in a chat is what lets
    LangGraph's own checkpoint history double as a durable, replayable
    audit log of that chat's whole adaptive-retrieval behavior, independent
    of what we store in the `messages` table for display.
    """
    import uuid

    init_langsmith()
    settings = get_settings()
    thread_id = f"user_{user_id}_chat_{chat_id}"
    query_id = str(uuid.uuid4())

    initial_state = new_initial_state(
        user_id=user_id,
        chat_id=chat_id,
        thread_id=thread_id,
        query_id=query_id,
        query=query,
        chat_history_text=chat_history_text,
        max_iterations=settings.max_retrieval_iterations,
        max_verification_retries=settings.max_verification_retries,
        web_search_enabled=settings.enable_web_search,
    )

    config = run_config(
        user_id=user_id,
        chat_id=chat_id,
        query_id=query_id,
        thread_id=thread_id,
    )

    with _checkpointer() as checkpointer:
        graph = build_graph().compile(checkpointer=checkpointer)
        final_state: RAGState = graph.invoke(initial_state, config=config)

    return final_state
