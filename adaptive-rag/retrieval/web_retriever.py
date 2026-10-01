"""Optional external web search, used only when local document evidence is
insufficient AND `ENABLE_WEB_SEARCH=true`. Built as a self-contained,
swappable module — disabling it (the default) removes any outbound network
dependency from the correction loop entirely.
"""
from __future__ import annotations

import time

from config.constants import RetrievalStrategy
from config.settings import get_settings
from retrieval.vector_retriever import RetrievedChunk


class WebSearchUnavailableError(RuntimeError):
    pass


def web_search(query: str, k: int = 5) -> tuple[list[RetrievedChunk], float]:
    settings = get_settings()
    if not settings.enable_web_search:
        raise WebSearchUnavailableError("Web search is disabled (ENABLE_WEB_SEARCH=false).")
    if not settings.web_search_api_key:
        raise WebSearchUnavailableError("WEB_SEARCH_API_KEY is not configured.")

    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.web_search_api_key)

    start = time.perf_counter()
    try:
        response = client.search(query=query, max_results=k, search_depth="basic")
    except Exception as exc:  # network/API failure must degrade gracefully
        raise WebSearchUnavailableError(f"Web search request failed: {exc}") from exc
    latency = time.perf_counter() - start

    chunks: list[RetrievedChunk] = []
    for i, result in enumerate(response.get("results", [])):
        chunks.append(
            RetrievedChunk(
                chunk_id=f"web-{i}",
                document_id="web",
                content=result.get("content", ""),
                filename=result.get("url", "web result"),
                page_number=None,
                score=round(float(result.get("score", 0.5)), 4),
                retrieval_method=RetrievalStrategy.WEB.value,
            )
        )
    return chunks, latency
