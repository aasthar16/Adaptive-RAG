from __future__ import annotations

from config.constants import CorrectionAction, EvaluationClassification, RetrievalStrategy
from evaluation.schemas import RetrievalEvaluation
from retrieval.controller import decide_correction_action
from retrieval.vector_retriever import RetrievedChunk


def _evaluation(**overrides) -> RetrievalEvaluation:
    defaults = dict(
        relevance_score=0.5,
        completeness_score=0.5,
        evidence_score=0.5,
        redundancy_score=0.5,
        overall_score=0.5,
        classification=EvaluationClassification.PARTIAL,
        reasoning="test",
    )
    defaults.update(overrides)
    return RetrievalEvaluation(**defaults)


def test_controller_reranks_when_relevant_but_poorly_ordered():
    decision = decide_correction_action(
        evaluation=_evaluation(relevance_score=0.8, evidence_score=0.3),
        iteration=1,
        current_strategy=RetrievalStrategy.VECTOR.value,
        actions_already_tried=[],
        web_search_enabled=False,
    )
    assert decision.action == CorrectionAction.RERANK


def test_controller_does_not_rerank_twice():
    decision = decide_correction_action(
        evaluation=_evaluation(relevance_score=0.8, evidence_score=0.3),
        iteration=1,
        current_strategy=RetrievalStrategy.VECTOR.value,
        actions_already_tried=[CorrectionAction.RERANK.value],
        web_search_enabled=False,
    )
    assert decision.action != CorrectionAction.RERANK


def test_controller_requests_additional_retrieval_when_incomplete():
    decision = decide_correction_action(
        evaluation=_evaluation(relevance_score=0.7, completeness_score=0.3, evidence_score=0.8),
        iteration=1,
        current_strategy=RetrievalStrategy.VECTOR.value,
        actions_already_tried=[],
        web_search_enabled=False,
    )
    assert decision.action == CorrectionAction.ADDITIONAL_RETRIEVAL
    assert decision.next_strategy == RetrievalStrategy.HYBRID.value


def test_controller_rewrites_query_when_relevance_is_low():
    decision = decide_correction_action(
        evaluation=_evaluation(relevance_score=0.2, completeness_score=0.2, evidence_score=0.8),
        iteration=1,
        current_strategy=RetrievalStrategy.VECTOR.value,
        actions_already_tried=[],
        web_search_enabled=False,
    )
    assert decision.action == CorrectionAction.REWRITE_QUERY


def test_controller_escalates_to_web_search_when_enabled_and_exhausted():
    decision = decide_correction_action(
        evaluation=_evaluation(relevance_score=0.2, completeness_score=0.2, evidence_score=0.8),
        iteration=1,
        current_strategy=RetrievalStrategy.VECTOR.value,
        actions_already_tried=[CorrectionAction.REWRITE_QUERY.value],
        web_search_enabled=True,
    )
    assert decision.action == CorrectionAction.EXTERNAL_SEARCH


def test_controller_never_suggests_web_search_when_disabled():
    decision = decide_correction_action(
        evaluation=_evaluation(relevance_score=0.1, completeness_score=0.1, evidence_score=0.8),
        iteration=1,
        current_strategy=RetrievalStrategy.VECTOR.value,
        actions_already_tried=[CorrectionAction.REWRITE_QUERY.value],
        web_search_enabled=False,
    )
    assert decision.action != CorrectionAction.EXTERNAL_SEARCH


def test_controller_gives_up_at_max_iterations():
    decision = decide_correction_action(
        evaluation=_evaluation(relevance_score=0.1),
        iteration=3,
        current_strategy=RetrievalStrategy.VECTOR.value,
        actions_already_tried=[],
        web_search_enabled=True,
    )
    assert decision.action == CorrectionAction.GIVE_UP


def test_controller_eventually_gives_up_when_all_actions_exhausted():
    decision = decide_correction_action(
        evaluation=_evaluation(relevance_score=0.55, completeness_score=0.9, evidence_score=0.9),
        iteration=1,
        current_strategy=RetrievalStrategy.HYBRID.value,
        actions_already_tried=[
            CorrectionAction.RERANK.value,
            CorrectionAction.REWRITE_QUERY.value,
            CorrectionAction.EXTERNAL_SEARCH.value,
            CorrectionAction.ADDITIONAL_RETRIEVAL.value,
        ],
        web_search_enabled=True,
    )
    assert decision.action == CorrectionAction.GIVE_UP


def _chunk(chunk_id: str, method: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=f"content for {chunk_id}",
        filename="f.pdf",
        page_number=1,
        score=0.5,
        retrieval_method=method,
    )


def test_hybrid_rrf_prioritizes_chunks_found_by_both_methods(monkeypatch):
    from retrieval import hybrid_retriever

    vector_results = [_chunk("a", "vector"), _chunk("b", "vector"), _chunk("c", "vector")]
    keyword_results = [_chunk("b", "keyword"), _chunk("d", "keyword")]

    monkeypatch.setattr(hybrid_retriever, "vector_search", lambda *a, **k: (vector_results, 0.01))
    monkeypatch.setattr(hybrid_retriever, "keyword_search", lambda *a, **k: (keyword_results, 0.01))

    fused, _latency = hybrid_retriever.hybrid_search("query", "chat-1", "user-1", k=4)

    fused_ids = [c.chunk_id for c in fused]
    # "b" appears in both lists (ranked high in each) so RRF should place
    # it ahead of anything found by only one method.
    assert fused_ids[0] == "b"
    assert set(fused_ids) == {"a", "b", "c", "d"}
