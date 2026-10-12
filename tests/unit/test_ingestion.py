import io

import pytest
from reportlab.pdfgen import canvas

from app.services.ingestion import UnsupportedDocument, chunk_pages, extract_pages


def test_chunking_splits_on_paragraphs_and_tracks_pages():
    text = (
        "First clause about indemnification liability and its cap amount here.\n\n"
        "Second clause about governing law and jurisdiction of the agreement."
    )
    chunks = chunk_pages([text, "Page two clause covering termination notice requirements for either party."])
    assert [c.page_number for c in chunks] == [1, 1, 2]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_tiny_paragraphs_are_merged():
    chunks = chunk_pages(["Heading\n\nA much longer paragraph that carries the actual contractual obligation text."])
    assert len(chunks) == 1 and "Heading" in chunks[0].text


def test_long_paragraph_is_split():
    long = " ".join(f"Sentence number {i} describes a contractual obligation in detail." for i in range(60))
    assert len(chunk_pages([long])) > 1


def test_rejects_unknown_extension():
    with pytest.raises(UnsupportedDocument):
        extract_pages("malware.exe", b"MZ\x90\x00")


def test_rejects_oversized_file():
    with pytest.raises(UnsupportedDocument):
        extract_pages("big.txt", b"a" * (10 * 1024 * 1024 + 1))


def test_extracts_text_from_real_pdf():
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 750, "The Company may terminate this Agreement upon 30 days written notice.")
    c.showPage()
    c.drawString(72, 750, "Governed by the laws of Delaware.")
    c.save()
    pages = extract_pages("contract.pdf", buf.getvalue())
    assert len(pages) == 2 and "terminate" in pages[0]


def test_pdf_detected_by_magic_bytes_not_extension():
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 750, "Confidentiality obligations survive for five years.")
    c.save()
    assert "Confidentiality" in extract_pages("renamed.txt", buf.getvalue())[0]
