"""PDF layout extraction using PyMuPDF (Phase 2).

Extracts raw text, spatial block coordinates, page dimensions, font metadata,
and section heading markers from technical PDF documents.
"""

from pathlib import Path
import re
from typing import BinaryIO, Optional, Union
import pymupdf

from core.ingestion.models import (
    DocumentMetadata,
    ExtractedDocument,
    PageContent,
    TextBlock,
)
from core.logger import get_logger

logger = get_logger(__name__)

# Regex pattern for hierarchical section headings (e.g., "1.1 Architecture", "Section 2.3 Ports")
SECTION_PATTERN = re.compile(r"^(?:Section\s+)?(\d+(?:\.\d+)*)\s+([A-Z][\w\s\-]+)$", re.IGNORECASE)


class PDFExtractor:
    """Extracts structured text and layout metadata from PDF documents using PyMuPDF."""

    def __init__(self, heading_min_font_size: float = 13.0) -> None:
        """Initialize PDF extractor.

        Args:
            heading_min_font_size: Threshold font size (in points) above which text
                is suspected to be a section heading.
        """
        self.heading_min_font_size = heading_min_font_size

    def extract(
        self,
        source: Union[str, Path, bytes, BinaryIO],
        filename: Optional[str] = None,
    ) -> ExtractedDocument:
        """Extract all pages and structural metadata from a PDF.

        Args:
            source: File path, raw bytes, or file-like binary stream.
            filename: Optional custom filename for identification.

        Returns:
            ExtractedDocument containing DocumentMetadata and a list of PageContent models.

        Raises:
            FileNotFoundError: If source path does not exist.
            ValueError: If the file is not a valid PDF or cannot be opened.
        """
        doc = None
        source_path: Optional[str] = None
        file_size: Optional[int] = None
        resolved_name = filename or "document.pdf"

        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f"PDF file not found at: {path}")
                source_path = str(path.resolve())
                file_size = path.stat().st_size
                resolved_name = filename or path.name
                doc = pymupdf.open(str(path))
            elif isinstance(source, bytes):
                file_size = len(source)
                doc = pymupdf.open(stream=source, filetype="pdf")
            else:
                # Binary stream
                raw_bytes = source.read()
                file_size = len(raw_bytes)
                doc = pymupdf.open(stream=raw_bytes, filetype="pdf")

            logger.info("Opened PDF '%s' with %d pages", resolved_name, len(doc))

            # Extract document-level metadata
            raw_meta = doc.metadata or {}
            doc_metadata = DocumentMetadata(
                filename=resolved_name,
                total_pages=len(doc),
                file_size_bytes=file_size,
                title=raw_meta.get("title") or None,
                author=raw_meta.get("author") or None,
                creation_date=raw_meta.get("creationDate") or None,
                source_path=source_path,
                extra_metadata={
                    k: v for k, v in raw_meta.items()
                    if k not in ["title", "author", "creationDate"] and v
                },
            )

            pages: list[PageContent] = []
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_number = page_idx + 1  # 1-indexed

                page_content = self._extract_page(page, page_number, resolved_name)
                pages.append(page_content)

            return ExtractedDocument(metadata=doc_metadata, pages=pages)

        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            logger.error("Failed to extract PDF '%s': %s", resolved_name, str(exc))
            raise ValueError(f"Could not parse PDF '{resolved_name}': {exc}") from exc
        finally:
            if doc is not None:
                doc.close()

    def _extract_page(
        self,
        page: pymupdf.Page,
        page_number: int,
        source_document: str,
    ) -> PageContent:
        """Extract structured blocks, text, and tables from a single page."""
        rect = page.rect
        page_dict = page.get_text("dict")
        raw_full_text = page.get_text("text")

        # Discover tables if any
        tables_count = 0
        try:
            tabs = page.find_tables()
            tables_count = len(tabs.tables) if tabs else 0
        except Exception:
            tables_count = 0

        blocks: list[TextBlock] = []
        block_idx = 0

        for b in page_dict.get("blocks", []):
            if b.get("type") != 0:  # Only process text blocks (type 0)
                continue

            lines = b.get("lines", [])
            if not lines:
                continue

            block_text_lines = []
            max_font_size = 0.0
            primary_font_name = None

            for line in lines:
                line_spans = line.get("spans", [])
                line_str = "".join(span.get("text", "") for span in line_spans).strip()
                if line_str:
                    block_text_lines.append(line_str)

                for span in line_spans:
                    span_size = span.get("size", 0.0)
                    if span_size > max_font_size:
                        max_font_size = span_size
                        primary_font_name = span.get("font")

            block_text = "\n".join(block_text_lines).strip()
            if not block_text:
                continue

            bbox = tuple(b.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            is_heading, heading_level = self._detect_heading(block_text, max_font_size)

            # Detect table-like block (e.g. contains pipe or multiple tab characters)
            is_table_like = "\t" in block_text or ("|" in block_text and block_text.count("|") >= 2)

            blocks.append(
                TextBlock(
                    block_index=block_idx,
                    text=block_text,
                    bbox=bbox,  # type: ignore[arg-type]
                    is_heading=is_heading,
                    heading_level=heading_level,
                    font_size=max_font_size if max_font_size > 0 else None,
                    font_name=primary_font_name,
                    is_table_like=is_table_like,
                )
            )
            block_idx += 1

        return PageContent(
            page_number=page_number,
            width=rect.width,
            height=rect.height,
            text=raw_full_text,
            blocks=blocks,
            tables_count=tables_count,
            source_document=source_document,
            metadata={
                "has_tables": tables_count > 0,
                "block_count": len(blocks),
            },
        )

    def _detect_heading(self, text: str, font_size: float) -> tuple[bool, Optional[int]]:
        """Identify if a text block functions as a section or chapter heading."""
        first_line = text.split("\n")[0].strip()

        # Check explicit section number pattern
        match = SECTION_PATTERN.match(first_line)
        if match:
            dots_count = match.group(1).count(".")
            level = dots_count + 1
            return True, level

        # Check font size threshold for short single/double line text
        if font_size >= self.heading_min_font_size and len(text.split("\n")) <= 2 and len(text) < 120:
            if font_size >= 16.0:
                return True, 1
            if font_size >= 14.0:
                return True, 2
            return True, 3

        return False, None
