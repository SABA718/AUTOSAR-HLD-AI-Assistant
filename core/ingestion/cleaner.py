"""Text cleaning and structural metadata preservation module (Phase 2).

Cleans extracted text by resolving hyphenated linebreaks and filtering out
superficial headers/footers while preserving AUTOSAR section hierarchy,
spatial block structure, and page numbers.
"""

from collections import Counter
import re
from typing import Optional

from core.ingestion.models import (
    CleanedBlock,
    CleanedPage,
    ExtractedDocument,
    PageContent,
    TextBlock,
)
from core.logger import get_logger

logger = get_logger(__name__)

# Hyphen unwrapping across linebreaks: e.g. "compo-\nnent" -> "component"
HYPHEN_LINEBREAK_PATTERN = re.compile(r"(\b[a-zA-Z]{2,})-\n\s*([a-zA-Z]{2,}\b)")

# Common footer page number patterns: e.g. "Page 1 of 10", "Page 2", "- 3 -"
PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:page\s+\d+(?:\s*(?:of|/)\s*\d+)?|\d+|-\s*\d+\s*-)$",
    re.IGNORECASE,
)


class TextCleaner:
    """Normalizes raw extracted PDF pages and preserves architectural structural metadata."""

    def __init__(
        self,
        strip_page_numbers: bool = True,
        strip_repeating_headers: bool = True,
        header_margin_ratio: float = 0.08,  # top 8% of page considered header zone
        footer_margin_ratio: float = 0.08,  # bottom 8% of page considered footer zone
    ) -> None:
        self.strip_page_numbers = strip_page_numbers
        self.strip_repeating_headers = strip_repeating_headers
        self.header_margin_ratio = header_margin_ratio
        self.footer_margin_ratio = footer_margin_ratio

    def clean_text(self, text: str) -> str:
        """Perform text-level normalization without destroying paragraph or table boundaries."""
        if not text:
            return ""

        # Normalize non-breaking spaces and carriage returns
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")

        # Unwrap hyphenated linebreaks
        cleaned = HYPHEN_LINEBREAK_PATTERN.sub(r"\1\2", cleaned)

        # Strip excessive trailing whitespaces per line
        cleaned_lines = [line.rstrip() for line in cleaned.split("\n")]
        cleaned = "\n".join(cleaned_lines)

        # Collapse 3 or more consecutive newlines into 2 (paragraph break)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def clean_page(
        self,
        page: PageContent,
        repeating_headers: Optional[set[str]] = None,
    ) -> CleanedPage:
        """Clean an individual PageContent into a normalized CleanedPage."""
        header_threshold_y = page.height * self.header_margin_ratio
        footer_threshold_y = page.height * (1.0 - self.footer_margin_ratio)

        cleaned_blocks: list[CleanedBlock] = []
        headings: list[str] = []

        for block in page.blocks:
            bbox = block.bbox
            block_raw_text = block.text.strip()

            # 1. Check if block is in footer zone and is just a page number
            if self.strip_page_numbers and bbox[1] >= footer_threshold_y:
                if PAGE_NUMBER_PATTERN.match(block_raw_text):
                    logger.debug("Filtered page number footer on page %d: '%s'", page.page_number, block_raw_text)
                    continue

            # 2. Check if block is a recurring header in header zone
            if self.strip_repeating_headers and bbox[3] <= header_threshold_y:
                if repeating_headers and block_raw_text in repeating_headers:
                    logger.debug("Filtered repeating header on page %d: '%s'", page.page_number, block_raw_text)
                    continue
                if PAGE_NUMBER_PATTERN.match(block_raw_text):
                    continue

            # 3. Clean block text
            cleaned_block_text = self.clean_text(block.text)
            if not cleaned_block_text:
                continue

            # 4. Record section headings
            if block.is_heading:
                headings.append(cleaned_block_text)

            cleaned_blocks.append(
                CleanedBlock(
                    block_index=block.block_index,
                    text=cleaned_block_text,
                    is_heading=block.is_heading,
                    heading_level=block.heading_level,
                    bbox=block.bbox,
                    is_table_like=block.is_table_like,
                )
            )

        # Assemble unified cleaned page text from cleaned blocks
        page_text = "\n\n".join(b.text for b in cleaned_blocks)

        return CleanedPage(
            page_number=page.page_number,
            text=page_text,
            blocks=cleaned_blocks,
            headings=headings,
            source_document=page.source_document,
            metadata={
                **page.metadata,
                "cleaned_blocks_count": len(cleaned_blocks),
                "headings_count": len(headings),
            },
        )

    def clean_document(self, extracted_doc: ExtractedDocument) -> list[CleanedPage]:
        """Clean all pages of an ExtractedDocument and dynamically strip repeating headers."""
        repeating_headers: set[str] = set()

        # Discover repeating headers across pages (if 2 or more pages exist)
        if len(extracted_doc.pages) > 1 and self.strip_repeating_headers:
            header_candidates: list[str] = []
            for page in extracted_doc.pages:
                header_y_max = page.height * self.header_margin_ratio
                for block in page.blocks:
                    if block.bbox[3] <= header_y_max:
                        clean_line = block.text.strip()
                        if clean_line and not PAGE_NUMBER_PATTERN.match(clean_line):
                            header_candidates.append(clean_line)

            counts = Counter(header_candidates)
            for text, freq in counts.items():
                if freq >= 2:
                    repeating_headers.add(text)
                    logger.info("Detected repeating header (freq=%d): '%s'", freq, text)

        cleaned_pages = [
            self.clean_page(p, repeating_headers=repeating_headers)
            for p in extracted_doc.pages
        ]
        return cleaned_pages
