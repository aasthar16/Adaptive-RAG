"""
Shared structured-output schemas and an LLM factory used throughout the
graph. Every node that needs the LLM to make a *decision* (as opposed to
generate free-text prose) asks for one of these Pydantic models via
`with_structured_output`, rather than parsing arbitrary text — this is
what section 16 of the spec calls for and it's what makes the Correction
Controller's routing explainable and testable.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from config.constants import EvaluationClassification, QueryComplexity, QueryType
from config.settings import get_settings

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=8)
def _base_llm(model_name: str, temperature: float) -> BaseChatModel:
    settings = get_settings()
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.openai_api_key or None,
        )
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. "
        "Add a branch in evaluation/schemas.py to support it."
    )


def get_llm(temperature: float = 0.2, fast: bool = False) -> BaseChatModel:
    """Plain chat model for free-text generation (answers, rewrites)."""
    settings = get_settings()
    model_name = settings.fast_model if fast else settings.chat_model
    return _base_llm(model_name, temperature)


def get_structured_llm(schema: Type[T], temperature: float = 0.0, fast: bool = True) -> BaseChatModel:
    """Chat model bound to always return an instance of `schema`.

    Structured-output nodes default to the faster/cheaper model since the
    task is classification/scoring, not open-ended generation — this is
    part of the project's cost-optimization story (section 27).
    """
    llm = get_llm(temperature=temperature, fast=fast)
    return llm.with_structured_output(schema)


# ---------------------------------------------------------------------
# Query analysis (Query Analyzer node)
# ---------------------------------------------------------------------
class QueryAnalysis(BaseModel):
    query_type: QueryType = Field(description="The category that best describes the query.")
    complexity: QueryComplexity = Field(description="Estimated difficulty of answering this query.")
    requires_multiple_sources: bool = Field(
        description="Whether answering well likely requires synthesizing multiple documents/chunks."
    )
    recommended_strategy: str = Field(
        description="One of: vector, keyword, hybrid. Best initial retrieval strategy for this query."
    )
    reasoning: str = Field(description="One or two sentences explaining the classification.")


# ---------------------------------------------------------------------
# Retrieval evaluation (Context Evaluator node)
# ---------------------------------------------------------------------
class RetrievalEvaluation(BaseModel):
    relevance_score: float = Field(ge=0.0, le=1.0, description="Are the retrieved chunks topically relevant to the query?")
    completeness_score: float = Field(ge=0.0, le=1.0, description="Do the chunks contain enough information to fully answer the query?")
    evidence_score: float = Field(ge=0.0, le=1.0, description="Is the evidence specific, concrete, and well-ordered (best evidence first)?")
    redundancy_score: float = Field(ge=0.0, le=1.0, description="1.0 = no wasted redundancy between chunks, 0.0 = highly repetitive chunks.")
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall weighted judgment of retrieval quality.")
    classification: EvaluationClassification = Field(description="sufficient | partial | poor, per configured thresholds.")
    reasoning: str = Field(description="Brief explanation of the scores.")


# ---------------------------------------------------------------------
# Answer verification (Answer Verifier node)
# ---------------------------------------------------------------------
class UnsupportedClaim(BaseModel):
    claim: str
    reason: str


class AnswerVerification(BaseModel):
    faithfulness_score: float = Field(ge=0.0, le=1.0, description="Fraction of major claims that are supported by the provided context.")
    supported: bool = Field(description="True if the answer is adequately supported by the retrieved context overall.")
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)
    reasoning: str = Field(description="Brief explanation of the verdict.")


# ---------------------------------------------------------------------
# Query rewrite (used by the Correction Controller's rewrite action)
# ---------------------------------------------------------------------
class QueryRewrite(BaseModel):
    rewritten_query: str = Field(description="A clearer, more retrievable rephrasing of the original query.")
    reasoning: str = Field(description="Why this rewrite should retrieve better evidence.")
