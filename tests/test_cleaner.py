"""Unit tests for text cleaning and structural metadata preservation (Phase 2)."""

import pytest
import pymupdf

from core.ingestion.cleaner import TextCleaner
from core.ingestion.pdf_extractor import PDFExtractor
from core.ingestion.models import CleanedPage


@pytest.fixture
def synthetic_pdf_bytes() -> bytes:
    """Generate an in-memory 2-page synthetic AUTOSAR HLD PDF with headers, footers, and hyphens."""
    doc = pymupdf.open()

    # Page 1
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 40), "AUTOSAR Classic Platform 4.4.0", fontsize=9)
    p1.insert_text((50, 100), "1. Architecture Overview", fontsize=16)
    p1.insert_text((50, 140), "Software Components define boundaries for automotive control units.", fontsize=11)
    p1.insert_text((50, 810), "Page 1 of 2", fontsize=9)

    # Page 2
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 40), "AUTOSAR Classic Platform 4.4.0", fontsize=9)
    p2.insert_text((50, 100), "2. Port and Interface Specifications", fontsize=16)
    p2.insert_text((50, 140), "The RPort receives signals from sensor compo-\nnents across the RTE.", fontsize=11)
    p2.insert_text((50, 810), "Page 2 of 2", fontsize=9)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_clean_hyphenation():
    """Verify hyphen unwrapping across linebreaks while preserving inline hyphens."""
    cleaner = TextCleaner()

    # Broken across newline
    raw = "The sensor compo-\nnent communicates with the actuator."
    expected = "The sensor component communicates with the actuator."
    assert cleaner.clean_text(raw) == expected

    # Preserves inline domain hyphens
    inline_raw = "Sender-Receiver interface and Sensor-Actuator SWC."
    assert cleaner.clean_text(inline_raw) == inline_raw


def test_clean_whitespace_normalization():
    """Verify excessive newlines and spaces are collapsed cleanly."""
    cleaner = TextCleaner()

    raw = "Paragraph 1 line.\n   \n\n\n\nParagraph 2 line.  "
    cleaned = cleaner.clean_text(raw)
    assert cleaned == "Paragraph 1 line.\n\nParagraph 2 line."


def test_clean_document_end_to_end(synthetic_pdf_bytes: bytes):
    """Verify cleaning an extracted document strips headers/footers and unwraps hyphens."""
    extractor = PDFExtractor()
    extracted = extractor.extract(synthetic_pdf_bytes, filename="autosar_spec.pdf")

    cleaner = TextCleaner()
    cleaned_pages = cleaner.clean_document(extracted)

    assert len(cleaned_pages) == 2

    # Check Page 1
    cp1 = cleaned_pages[0]
    assert isinstance(cp1, CleanedPage)
    assert cp1.page_number == 1
    assert cp1.source_document == "autosar_spec.pdf"
    assert "1. Architecture Overview" in cp1.headings
    # Repeating header and footer stripped
    assert "AUTOSAR Classic Platform 4.4.0" not in cp1.text
    assert "Page 1 of 2" not in cp1.text

    # Check Page 2
    cp2 = cleaned_pages[1]
    assert cp2.page_number == 2
    assert "2. Port and Interface Specifications" in cp2.headings
    # Hyphen unwrapped
    assert "component" in cp2.text
    assert "compo-" not in cp2.text
    # Repeating header and footer stripped
    assert "AUTOSAR Classic Platform 4.4.0" not in cp2.text
    assert "Page 2 of 2" not in cp2.text


def test_page_metadata_and_blocks_preserved(synthetic_pdf_bytes: bytes):
    """Verify that cleaned pages retain spatial blocks and block indices."""
    extractor = PDFExtractor()
    extracted = extractor.extract(synthetic_pdf_bytes, filename="autosar_spec.pdf")

    cleaner = TextCleaner()
    cleaned_pages = cleaner.clean_document(extracted)

    for cp in cleaned_pages:
        assert len(cp.blocks) > 0
        for block in cp.blocks:
            assert block.block_index >= 0
            assert block.bbox is not None
            assert len(block.text) > 0
