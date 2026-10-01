"""Embedding model factory.

Kept separate from the retriever/ingestion code so the embedding provider
can be swapped (e.g. local/HuggingFace) by changing this one function,
without touching ingestion or retrieval logic.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from config.settings import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    settings = get_settings()
    if settings.llm_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key or None,
            dimensions=settings.embedding_dimensions,
        )
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}' for embeddings. "
        "Add a branch in ingestion/embeddings.py to support it."
    )
