"""Text cleaning + chunking used by the ingestion pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.loaders import RawPage

_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# Strips common PDF extraction artifacts: form-feed / null bytes / stray
# control characters that occasionally survive text extraction.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str) -> str:
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


@dataclass
class Chunk:
    content: str
    chunk_index: int
    page_number: int | None


def chunk_pages(
    pages: list[RawPage],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Clean each page, then split with a recursive character splitter that
    tries to break on paragraph/sentence boundaries before falling back to
    hard character cuts. Each resulting chunk keeps a reference to the page
    it came from so citations can show real page numbers.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    idx = 0
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue
        for piece in splitter.split_text(cleaned):
            piece = piece.strip()
            if len(piece) < 20:
                # Skip near-empty fragments (e.g. lone page headers) —
                # they add noise to retrieval without adding evidence.
                continue
            chunks.append(Chunk(content=piece, chunk_index=idx, page_number=page.page_number))
            idx += 1
    return chunks
