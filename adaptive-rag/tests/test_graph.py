"""
Exercises the actual compiled LangGraph workflow end-to-end (no checkpointer
— see `build_graph().compile()` with no `checkpointer=` argument, which
keeps everything in-memory) with every LLM call and retriever function
mocked at the boundary `graph/nodes.py` imports them at. This tests the
*routing/control-flow* of the adaptive workflow itself — sufficient vs.
partial vs. poor retrieval, the correction loop, and graceful give-up —
without needing a database, an OpenAI key, or network access.

Routing-function unit tests (no graph execution needed) are included too,
since they're the simplest possible check that each conditional edge's
policy is right in isolation.
"""
from __future__ import annotations

from config.constants import CorrectionAction, EvaluationClassification, QueryComplexity, QueryType
from evaluation.schemas import AnswerVerification, QueryAnalysis, RetrievalEvaluation
from generation.prompts import INSUFFICIENT_EVIDENCE_MESSAGE
from graph import edges
from graph.state import new_initial_state
from graph.workflow import build_graph
from retrieval.vector_retriever import RetrievedChunk

# ---------------------------------------------------------------------
# Pure routing-function tests
# ---------------------------------------------------------------------


def test_route_after_evaluation_sufficient_goes_to_generate():
    state = {"evaluation_result": {"classification": EvaluationClassification.SUFFICIENT.value}}
    assert edges.route_after_evaluation(state) == edges.GENERATE_ANSWER


def test_route_after_evaluation_partial_goes_to_controller():
    state = {"evaluation_result": {"classification": EvaluationClassification.PARTIAL.value}}
    assert edges.route_after_evaluation(state) == edges.DECIDE_NEXT_ACTION


def test_route_after_evaluation_give_up_short_circuits_to_generate():
    state = {"give_up": True, "evaluation_result": {"classification": EvaluationClassification.POOR.value}}
    assert edges.route_after_evaluation(state) == edges.GENERATE_ANSWER


def test_route_after_decision_dispatches_each_action():
    for action, expected in [
        (CorrectionAction.RERANK.value, edges.RERANK_DOCUMENTS),
        (CorrectionAction.REWRITE_QUERY.value, edges.REWRITE_QUERY),
        (CorrectionAction.ADDITIONAL_RETRIEVAL.value, edges.RETRIEVE_DOCUMENTS),
        (CorrectionAction.EXTERNAL_SEARCH.value, edges.EXTERNAL_SEARCH),
        (CorrectionAction.GIVE_UP.value, edges.GENERATE_ANSWER),
    ]:
        assert edges.route_after_decision({"correction_action": action}) == expected


def test_route_after_verification_ends_when_supported():
    state = {"verification_result": {"supported": True}, "verification_retries": 0}
    assert edges.route_after_verification(state) == "end"


def test_route_after_verification_loops_when_unsupported_and_retries_remain():
    state = {"verification_result": {"supported": False}, "verification_retries": 0}
    assert edges.route_after_verification(state) == edges.PREPARE_REGENERATION


def test_route_after_verification_ends_when_retries_exhausted():
    state = {"verification_result": {"supported": False}, "verification_retries": 99}
    assert edges.route_after_verification(state) == "end"


# ---------------------------------------------------------------------
# Full graph execution tests (mocked LLM + retrieval boundaries)
# ---------------------------------------------------------------------


def _fake_chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id="doc-1",
        content="Gradient descent iteratively updates model parameters to minimize loss.",
        filename="ml.pdf",
        page_number=3,
        score=0.9,
        retrieval_method="vector",
    )


def _patch_common(monkeypatch, *, query_analysis: QueryAnalysis | None = None):
    from graph import nodes

    class _FakeStructuredLLM:
        def __init__(self, result):
            self._result = result

        def invoke(self, _messages):
            return self._result

    analysis = query_analysis or QueryAnalysis(
        query_type=QueryType.FACTUAL,
        complexity=QueryComplexity.LOW,
        requires_multiple_sources=False,
        recommended_strategy="vector",
        reasoning="test",
    )
    monkeypatch.setattr(nodes, "get_structured_llm", lambda *a, **k: _FakeStructuredLLM(analysis))
    return nodes


def _initial_state(query: str = "How does gradient descent work?", max_iterations: int = 3):
    return new_initial_state(
        user_id="user-1",
        chat_id="chat-1",
        thread_id="thread-1",
        query_id="query-1",
        query=query,
        chat_history_text="",
        max_iterations=max_iterations,
        max_verification_retries=1,
        web_search_enabled=False,
    )


def test_graph_generates_directly_when_retrieval_is_sufficient(monkeypatch):
    nodes = _patch_common(monkeypatch)

    chunks = [_fake_chunk("c1"), _fake_chunk("c2")]
    monkeypatch.setattr(nodes, "vector_search", lambda *a, **k: (chunks, 0.01))

    sufficient_eval = RetrievalEvaluation(
        relevance_score=0.9,
        completeness_score=0.9,
        evidence_score=0.9,
        redundancy_score=0.9,
        overall_score=0.9,
        classification=EvaluationClassification.SUFFICIENT,
        reasoning="Strong, complete evidence.",
    )
    monkeypatch.setattr(nodes, "run_retrieval_evaluation", lambda query, chunks: sufficient_eval)
    monkeypatch.setattr(nodes, "run_answer_generation", lambda query, chunks, chat_history="": "Gradient descent minimizes loss via iterative updates. [1]")

    verification_calls = []

    def _fake_verify(query, answer, chunks):
        verification_calls.append(1)
        return AnswerVerification(
            faithfulness_score=0.95, supported=True, unsupported_claims=[], reasoning="Fully supported."
        )

    monkeypatch.setattr(nodes, "run_answer_verification", _fake_verify)

    graph = build_graph().compile()
    final_state = graph.invoke(_initial_state())

    assert final_state["answer"] == "Gradient descent minimizes loss via iterative updates. [1]"
    assert final_state["give_up"] is False
    assert final_state["iteration"] == 1  # only one retrieval round was needed
    assert len(verification_calls) == 1
    assert "generate_answer" in final_state["node_trace"]
    assert "decide_next_action" not in final_state["node_trace"]


def test_graph_corrects_via_additional_retrieval_then_succeeds(monkeypatch):
    nodes = _patch_common(monkeypatch)

    sparse_chunks = [_fake_chunk("c1")]
    rich_chunks = [_fake_chunk("c1"), _fake_chunk("c2"), _fake_chunk("c3")]

    monkeypatch.setattr(nodes, "vector_search", lambda *a, **k: (sparse_chunks, 0.01))
    monkeypatch.setattr(nodes, "hybrid_search", lambda *a, **k: (rich_chunks, 0.01))

    partial_eval = RetrievalEvaluation(
        relevance_score=0.7, completeness_score=0.3, evidence_score=0.8, redundancy_score=0.9,
        overall_score=0.5, classification=EvaluationClassification.PARTIAL, reasoning="Incomplete.",
    )
    sufficient_eval = RetrievalEvaluation(
        relevance_score=0.85, completeness_score=0.85, evidence_score=0.85, redundancy_score=0.9,
        overall_score=0.85, classification=EvaluationClassification.SUFFICIENT, reasoning="Now complete.",
    )
    eval_results = iter([partial_eval, sufficient_eval])
    monkeypatch.setattr(nodes, "run_retrieval_evaluation", lambda query, chunks: next(eval_results))
    monkeypatch.setattr(nodes, "run_answer_generation", lambda query, chunks, chat_history="": "Final answer. [1][2]")
    monkeypatch.setattr(
        nodes, "run_answer_verification",
        lambda query, answer, chunks: AnswerVerification(faithfulness_score=0.9, supported=True, unsupported_claims=[], reasoning="ok"),
    )

    graph = build_graph().compile()
    final_state = graph.invoke(_initial_state())

    assert final_state["answer"] == "Final answer. [1][2]"
    assert final_state["iteration"] == 2  # first retrieval + one additional-retrieval correction
    assert final_state["correction_action"] == CorrectionAction.ADDITIONAL_RETRIEVAL.value
    assert len(final_state["retrieval_history"]) == 2
    assert final_state["retrieval_history"][1]["strategy"] == "hybrid"
    assert final_state["give_up"] is False


def test_graph_gives_up_gracefully_after_max_iterations_without_hallucinating(monkeypatch):
    nodes = _patch_common(monkeypatch)

    chunks = [_fake_chunk("c1")]
    monkeypatch.setattr(nodes, "vector_search", lambda *a, **k: (chunks, 0.01))
    monkeypatch.setattr(nodes, "hybrid_search", lambda *a, **k: (chunks, 0.01))

    poor_eval = RetrievalEvaluation(
        relevance_score=0.1, completeness_score=0.1, evidence_score=0.1, redundancy_score=0.5,
        overall_score=0.1, classification=EvaluationClassification.POOR, reasoning="No relevant evidence found.",
    )
    monkeypatch.setattr(nodes, "run_retrieval_evaluation", lambda query, chunks: poor_eval)

    generation_calls = []
    verification_calls = []
    monkeypatch.setattr(
        nodes, "run_answer_generation",
        lambda *a, **k: generation_calls.append(1) or "should never be used",
    )
    monkeypatch.setattr(
        nodes, "run_answer_verification",
        lambda *a, **k: verification_calls.append(1) or AnswerVerification(
            faithfulness_score=0.0, supported=False, unsupported_claims=[], reasoning="n/a"
        ),
    )

    graph = build_graph().compile()
    final_state = graph.invoke(_initial_state(max_iterations=3))

    assert final_state["give_up"] is True
    assert final_state["answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert final_state["sources"] == []
    # No hallucinated generation and no wasted verification call on a
    # give-up answer (section 22: "Do NOT hallucinate").
    assert generation_calls == []
    assert verification_calls == []
    assert final_state["correction_action"] == CorrectionAction.GIVE_UP.value


def test_graph_rewrites_query_when_relevance_is_low(monkeypatch):
    nodes = _patch_common(monkeypatch)

    monkeypatch.setattr(nodes, "vector_search", lambda *a, **k: ([_fake_chunk("c1")], 0.01))

    low_relevance_eval = RetrievalEvaluation(
        relevance_score=0.2, completeness_score=0.2, evidence_score=0.5, redundancy_score=0.9,
        overall_score=0.25, classification=EvaluationClassification.POOR, reasoning="Query seems ambiguous.",
    )
    good_eval = RetrievalEvaluation(
        relevance_score=0.9, completeness_score=0.9, evidence_score=0.9, redundancy_score=0.9,
        overall_score=0.9, classification=EvaluationClassification.SUFFICIENT, reasoning="Good.",
    )
    eval_results = iter([low_relevance_eval, good_eval])
    monkeypatch.setattr(nodes, "run_retrieval_evaluation", lambda query, chunks: next(eval_results))

    rewrite_calls = []

    def _fake_rewrite(original_query, reasoning):
        rewrite_calls.append(original_query)
        return "a clearer rephrasing of: " + original_query

    monkeypatch.setattr(nodes, "run_query_rewrite", _fake_rewrite)
    monkeypatch.setattr(nodes, "run_answer_generation", lambda query, chunks, chat_history="": "Answer after rewrite.")
    monkeypatch.setattr(
        nodes, "run_answer_verification",
        lambda query, answer, chunks: AnswerVerification(faithfulness_score=0.9, supported=True, unsupported_claims=[], reasoning="ok"),
    )

    graph = build_graph().compile()
    final_state = graph.invoke(_initial_state(query="what about it"))

    assert len(rewrite_calls) == 1
    assert final_state["current_query"] == "a clearer rephrasing of: what about it"
    # The rewrite must never overwrite the user's actual original question.
    assert final_state["original_query"] == "what about it"
    assert final_state["answer"] == "Answer after rewrite."
