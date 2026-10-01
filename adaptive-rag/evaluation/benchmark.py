"""
Benchmarking module (section 39): compares a naive baseline RAG pipeline
against the full AdaptiveRAG workflow on the same set of questions, so the
value of the adaptive/corrective machinery is measurable rather than
assumed.

    Baseline RAG:    query -> vector_search(k) -> generate_answer
    AdaptiveRAG:     the full LangGraph workflow in graph/workflow.py

For each question we compute, for both pipelines:
    - retrieval relevance/completeness (via the SAME LLM judge used inside
      AdaptiveRAG's own Context Evaluator, so the comparison is apples-to-
      apples and not "AdaptiveRAG grading its own homework differently")
    - answer faithfulness (via the same Answer Verifier)
    - latency
    - number of retrieval calls and LLM calls (the cost side of the
      cost/reliability tradeoff this whole project is about)

This is intentionally a library module, not a CLI by itself — see
`scripts/run_benchmark.py` for a runnable example against a real chat.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from config.settings import get_settings
from evaluation.answer_evaluator import verify_answer
from evaluation.retrieval_evaluator import evaluate_retrieval
from generation.generator import generate_answer
from graph.workflow import run_query
from retrieval.vector_retriever import vector_search


@dataclass
class PipelineRunResult:
    pipeline: str
    query: str
    answer: str
    relevance_score: float
    completeness_score: float
    faithfulness_score: float
    supported: bool
    latency_seconds: float
    retrieval_calls: int
    llm_calls: int


def run_baseline_rag(query: str, chat_id: str, user_id: str) -> PipelineRunResult:
    settings = get_settings()
    start = time.perf_counter()

    chunks, _ = vector_search(query, chat_id, user_id, k=settings.default_top_k)
    answer = generate_answer(query, chunks)

    latency = time.perf_counter() - start

    retrieval_eval = evaluate_retrieval(query, chunks)
    verification = verify_answer(query, answer, chunks)

    return PipelineRunResult(
        pipeline="baseline_rag",
        query=query,
        answer=answer,
        relevance_score=retrieval_eval.relevance_score,
        completeness_score=retrieval_eval.completeness_score,
        faithfulness_score=verification.faithfulness_score,
        supported=verification.supported,
        latency_seconds=round(latency, 3),
        retrieval_calls=1,
        llm_calls=1,  # generation only; the eval/verification calls above are benchmark overhead, not pipeline cost
    )


def run_adaptive_rag(query: str, chat_id: str, user_id: str) -> PipelineRunResult:
    start = time.perf_counter()
    final_state = run_query(user_id=user_id, chat_id=chat_id, query=query)
    latency = time.perf_counter() - start

    eval_history = final_state.get("evaluation_history", [])
    last_eval = eval_history[-1] if eval_history else {}
    verification = final_state.get("verification_result") or {}

    return PipelineRunResult(
        pipeline="adaptive_rag",
        query=query,
        answer=final_state.get("answer") or "",
        relevance_score=last_eval.get("relevance_score", 0.0),
        completeness_score=last_eval.get("completeness_score", 0.0),
        faithfulness_score=verification.get("faithfulness_score", 0.0),
        supported=verification.get("supported", False),
        latency_seconds=round(latency, 3),
        retrieval_calls=final_state.get("retrieval_call_count", 0),
        llm_calls=final_state.get("llm_call_count", 0),
    )


def compare(query: str, chat_id: str, user_id: str) -> dict:
    baseline = run_baseline_rag(query, chat_id, user_id)
    adaptive = run_adaptive_rag(query, chat_id, user_id)
    return {"baseline": asdict(baseline), "adaptive": asdict(adaptive)}


def run_benchmark_suite(questions: list[str], chat_id: str, user_id: str) -> dict:
    """Run every question through both pipelines and return per-question
    results plus averaged summary metrics for each pipeline."""
    rows = [compare(q, chat_id, user_id) for q in questions]

    def _avg(pipeline: str, field: str) -> float:
        values = [r[pipeline][field] for r in rows]
        return round(sum(values) / len(values), 4) if values else 0.0

    summary = {}
    for pipeline in ("baseline", "adaptive"):
        summary[pipeline] = {
            "avg_relevance_score": _avg(pipeline, "relevance_score"),
            "avg_completeness_score": _avg(pipeline, "completeness_score"),
            "avg_faithfulness_score": _avg(pipeline, "faithfulness_score"),
            "pct_supported": round(
                100 * sum(1 for r in rows if r[pipeline]["supported"]) / len(rows), 1
            )
            if rows
            else 0.0,
            "avg_latency_seconds": _avg(pipeline, "latency_seconds"),
            "avg_retrieval_calls": _avg(pipeline, "retrieval_calls"),
            "avg_llm_calls": _avg(pipeline, "llm_calls"),
        }

    return {"results": rows, "summary": summary}
