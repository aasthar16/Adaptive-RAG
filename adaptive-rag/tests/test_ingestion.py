from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.chunking import chunk_pages, clean_text
from ingestion.loaders import (
    DocumentLoadError,
    UnsupportedFileTypeError,
    load_document,
    load_docx,
    load_pdf,
    load_txt,
)
from tests.fixtures import build_docx, build_minimal_pdf


def test_load_txt(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("Neural networks learn representations through backpropagation.")
    pages = load_txt(path)
    assert len(pages) == 1
    assert "backpropagation" in pages[0].text


def test_load_empty_txt_raises(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n  ")
    with pytest.raises(DocumentLoadError):
        load_txt(path)


def test_load_pdf_extracts_text(tmp_path: Path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(build_minimal_pdf("Attention is all you need"))
    pages = load_pdf(path)
    assert len(pages) == 1
    assert "Attention" in pages[0].text


def test_load_corrupted_pdf_raises_document_load_error(tmp_path: Path):
    path = tmp_path / "corrupted.pdf"
    path.write_bytes(b"this is definitely not a real pdf file")
    with pytest.raises(DocumentLoadError):
        load_pdf(path)


def test_load_docx_extracts_paragraphs(tmp_path: Path):
    path = tmp_path / "notes.docx"
    path.write_bytes(build_docx(["Operating systems manage processes and memory.", "Second paragraph."]))
    pages = load_docx(path)
    assert len(pages) == 1
    assert "Operating systems" in pages[0].text
    assert "Second paragraph" in pages[0].text


def test_load_document_rejects_unsupported_extension(tmp_path: Path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04fakezipcontent")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(path)


def test_clean_text_collapses_whitespace_and_strips_control_chars():
    dirty = "Hello    world\x0c\n\n\n\nSecond   line\x00"
    cleaned = clean_text(dirty)
    assert "\x0c" not in cleaned
    assert "\x00" not in cleaned
    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned


def test_chunking_preserves_page_numbers_and_produces_metadata_ready_chunks():
    from ingestion.loaders import RawPage

    pages = [
        RawPage(page_number=1, text="A" * 1500),
        RawPage(page_number=2, text="B" * 500),
    ]
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)

    assert len(chunks) >= 3
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2}
    # chunk_index must be unique and sequential — this is what downstream
    # storage relies on for metadata enrichment (section 6).
    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(set(indices))


def test_chunking_skips_near_empty_fragments():
    from ingestion.loaders import RawPage

    pages = [RawPage(page_number=1, text="Real content sentence here.\n\nHi\n\nAnother real sentence.")]
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=0)
    assert all(len(c.content) >= 20 for c in chunks)
