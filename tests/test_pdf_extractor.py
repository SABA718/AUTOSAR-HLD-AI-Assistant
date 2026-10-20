"""Unit tests for PDF extraction module using synthetic multi-page PDFs (Phase 2)."""

from pathlib import Path
import pytest
import pymupdf

from core.ingestion.pdf_extractor import PDFExtractor
from core.ingestion.models import ExtractedDocument


@pytest.fixture
def synthetic_pdf_bytes() -> bytes:
    """Generate an in-memory 2-page synthetic AUTOSAR HLD PDF."""
    doc = pymupdf.open()

    # Page 1
    p1 = doc.new_page(width=595, height=842)
    # Header in top margin
    p1.insert_text((50, 40), "AUTOSAR Classic Platform 4.4.0", fontsize=9)
    # Section Heading
    p1.insert_text((50, 100), "1. Architecture Overview", fontsize=16)
    # Body text
    p1.insert_text((50, 140), "Software Components define boundaries for automotive control units.", fontsize=11)
    # Footer in bottom margin
    p1.insert_text((50, 810), "Page 1 of 2", fontsize=9)

    # Page 2
    p2 = doc.new_page(width=595, height=842)
    # Header in top margin
    p2.insert_text((50, 40), "AUTOSAR Classic Platform 4.4.0", fontsize=9)
    # Section Heading
    p2.insert_text((50, 100), "2. Port and Interface Specifications", fontsize=16)
    # Body text with hyphenation
    p2.insert_text((50, 140), "The RPort receives signals from sensor compo-\nnents across the RTE.", fontsize=11)
    # Footer in bottom margin
    p2.insert_text((50, 810), "Page 2 of 2", fontsize=9)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_pdf_extraction_metadata_and_pages(synthetic_pdf_bytes: bytes):
    """Verify that PDFExtractor correctly extracts document metadata and pages."""
    extractor = PDFExtractor(heading_min_font_size=13.0)
    extracted = extractor.extract(synthetic_pdf_bytes, filename="test_autosar_hld.pdf")

    assert isinstance(extracted, ExtractedDocument)
    assert extracted.metadata.filename == "test_autosar_hld.pdf"
    assert extracted.metadata.total_pages == 2
    assert len(extracted.pages) == 2

    # Check Page 1
    p1 = extracted.pages[0]
    assert p1.page_number == 1
    assert p1.source_document == "test_autosar_hld.pdf"
    assert p1.width == pytest.approx(595, rel=1e-2)
    assert p1.height == pytest.approx(842, rel=1e-2)
    assert len(p1.blocks) >= 2

    # Check Page 2
    p2 = extracted.pages[1]
    assert p2.page_number == 2
    assert p2.source_document == "test_autosar_hld.pdf"
    assert "Port and Interface" in p2.text


def test_heading_and_block_coordinate_preservation(synthetic_pdf_bytes: bytes):
    """Verify that headings are flagged and spatial bounding boxes are retained."""
    extractor = PDFExtractor(heading_min_font_size=13.0)
    extracted = extractor.extract(synthetic_pdf_bytes, filename="test_autosar_hld.pdf")

    p1 = extracted.pages[0]
    heading_blocks = [b for b in p1.blocks if b.is_heading]
    assert len(heading_blocks) >= 1

    first_heading = heading_blocks[0]
    assert "Architecture Overview" in first_heading.text
    assert first_heading.heading_level == 1
    assert first_heading.bbox[0] >= 0.0
    assert first_heading.font_size is not None and first_heading.font_size >= 13.0


def test_pdf_extraction_from_file_path(tmp_path: Path, synthetic_pdf_bytes: bytes):
    """Verify that PDFExtractor can load from a filesystem path."""
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(synthetic_pdf_bytes)

    extractor = PDFExtractor()
    extracted = extractor.extract(pdf_file)

    assert extracted.metadata.filename == "sample.pdf"
    assert extracted.metadata.source_path == str(pdf_file.resolve())
    assert extracted.metadata.total_pages == 2


def test_pdf_extraction_invalid_inputs():
    """Verify error handling on non-existent paths and corrupted bytes."""
    extractor = PDFExtractor()

    with pytest.raises(FileNotFoundError):
        extractor.extract(Path("non_existent_file.pdf"))

    with pytest.raises(ValueError):
        extractor.extract(b"not a valid pdf content", filename="corrupt.pdf")
