"""Document parsing and structure-aware chunking (paragraph/clause boundaries, not fixed windows)."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 60


class UnsupportedDocument(ValueError):
    pass


@dataclass
class ParsedChunk:
    page_number: int
    chunk_index: int
    text: str


def extract_pages(filename: str, data: bytes) -> list[str]:
    """Return one text string per page. PDFs are detected by magic bytes, not just extension."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise UnsupportedDocument("File exceeds the 10 MB limit")
    if data[:5] == b"%PDF-":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        if not any(p.strip() for p in pages):
            raise UnsupportedDocument("PDF contains no extractable text (scanned PDFs need an OCR step)")
        return pages
    if filename.lower().endswith((".txt", ".md")):
        try:
            return [data.decode("utf-8")]
        except UnicodeDecodeError:
            raise UnsupportedDocument("Text files must be UTF-8") from None
    raise UnsupportedDocument("Only PDF, .txt and .md files are supported")


def _split_long(paragraph: str) -> list[str]:
    if len(paragraph) <= MAX_CHUNK_CHARS:
        return [paragraph]
    out, cur = [], ""
    for sentence in re.split(r"(?<=[.;])\s+", paragraph):
        if cur and len(cur) + len(sentence) > MAX_CHUNK_CHARS:
            out.append(cur.strip())
            cur = ""
        cur += sentence + " "
    if cur.strip():
        out.append(cur.strip())
    return out


def chunk_pages(pages: list[str]) -> list[ParsedChunk]:
    chunks: list[ParsedChunk] = []
    for page_no, page_text in enumerate(pages, start=1):
        buf = ""
        for para in re.split(r"\n\s*\n", page_text):
            para = " ".join(para.split())
            if not para:
                continue
            buf = f"{buf} {para}".strip() if buf and len(buf) < MIN_CHUNK_CHARS else para
            if len(buf) >= MIN_CHUNK_CHARS:
                for piece in _split_long(buf):
                    chunks.append(ParsedChunk(page_no, len(chunks), piece))
                buf = ""
        if buf:
            chunks.append(ParsedChunk(page_no, len(chunks), buf))
    return chunks
