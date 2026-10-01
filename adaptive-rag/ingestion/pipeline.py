"""
The document ingestion pipeline (section 6 of the spec):

    Uploaded file -> Loader -> Clean -> Chunk -> Metadata enrichment
                  -> Embedding -> PostgreSQL + pgvector

This is the one place ingestion actually happens; it is deliberately
*decoupled* from retrieval (`retrieval/*.py`), matching the spec's core
distinction (section 47): ingestion runs once, per document, when a chat
is created or a document is added; retrieval runs fresh on every query
against whatever has already been ingested.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document as LCDocument

from database.repositories.document_repository import DocumentRepository
from ingestion.chunking import chunk_pages
from ingestion.loaders import DocumentLoadError, UnsupportedFileTypeError, load_document
from retrieval.vector_retriever import get_vector_store

logger = logging.getLogger(__name__)

_documents = DocumentRepository()


class IngestionError(RuntimeError):
    pass


def ingest_document(
    *,
    document_id: str,
    user_id: str,
    chat_id: str,
    file_path: Path,
    filename: str,
) -> int:
    """Run the full pipeline for one already-uploaded, already-saved file.

    Returns the number of chunks stored. Raises `IngestionError` on
    failure, after marking the document row as failed with a human-
    readable message — callers (the document service / UI) don't need
    their own try/except around loader-specific exception types.
    """
    _documents.set_status(document_id, status="processing")
    try:
        pages = load_document(file_path)
    except (UnsupportedFileTypeError, DocumentLoadError) as exc:
        logger.warning("Ingestion failed for %s: %s", filename, exc)
        _documents.set_status(document_id, status="failed", error_message=str(exc))
        raise IngestionError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected loader crash
        logger.exception("Unexpected ingestion failure for %s", filename)
        _documents.set_status(document_id, status="failed", error_message=f"Unexpected error: {exc}")
        raise IngestionError(f"Unexpected error while processing '{filename}': {exc}") from exc

    chunks = chunk_pages(pages)
    if not chunks:
        msg = f"'{filename}' produced no usable text chunks after cleaning."
        _documents.set_status(document_id, status="failed", error_message=msg)
        raise IngestionError(msg)

    lc_documents = []
    for chunk in chunks:
        metadata = {
            "user_id": user_id,
            "chat_id": chat_id,
            "document_id": document_id,
            "chunk_index": chunk.chunk_index,
            "filename": filename,
            "page_number": chunk.page_number,
            "source": filename,
        }
        lc_documents.append(LCDocument(page_content=chunk.content, metadata=metadata))

    try:
        store = get_vector_store()
        store.add_documents(lc_documents)
    except Exception as exc:
        msg = f"Embedding/storage failed for '{filename}': {exc}"
        logger.exception(msg)
        _documents.set_status(document_id, status="failed", error_message=msg)
        raise IngestionError(msg) from exc

    _documents.set_status(document_id, status="indexed", page_count=len(pages))
    logger.info("Ingested '%s' -> %d chunks (document_id=%s)", filename, len(chunks), document_id)
    return len(chunks)
