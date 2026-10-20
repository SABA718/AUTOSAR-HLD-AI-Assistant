"""AUTOSAR-Aware Hierarchical & Structural Document Chunker (Phase 3A).

Partitions CleanedPage sequences into bounded Chunk objects while preserving:
1. Section/chapter heading boundaries.
2. Contiguous architectural table blocks.
3. Configurable sentence-boundary sliding overlap for oversized sections.
4. Accurate character offsets, page citations, and metadata.
"""

from pathlib import Path
import re
from typing import Optional, Sequence

from core.chunking.models import Chunk, ChunkMetadata
from core.ingestion.models import CleanedBlock, CleanedPage
from core.logger import get_logger

logger = get_logger(__name__)

# Sentence boundary regex splitter
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class HierarchicalChunker:
    """Chunks cleaned pages according to document structure, tables, and section hierarchies."""

    def __init__(
        self,
        max_chunk_chars: int = 1200,
        min_chunk_chars: int = 100,
        overlap_chars: int = 200,
        respect_sentence_boundaries: bool = True,
    ) -> None:
        """Initialize chunker.

        Args:
            max_chunk_chars: Maximum character length for any single chunk.
            min_chunk_chars: Minimum character length to form a standalone chunk
                (smaller text blocks are combined with adjacent blocks unless they are tables or headings).
            overlap_chars: Fallback overlap size when splitting oversized text blocks.
            respect_sentence_boundaries: Whether to break oversized text at sentence ends.
        """
        if overlap_chars >= max_chunk_chars:
            raise ValueError(f"overlap_chars ({overlap_chars}) must be less than max_chunk_chars ({max_chunk_chars})")

        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.overlap_chars = overlap_chars
        self.respect_sentence_boundaries = respect_sentence_boundaries

    def chunk_document(
        self,
        pages: Sequence[CleanedPage],
        document_id: Optional[str] = None,
    ) -> list[Chunk]:
        """Transform a sequence of CleanedPage models into structured Chunks.

        Args:
            pages: List of CleanedPage instances.
            document_id: Optional unique identifier for the document.
                Defaults to the source document filename or 'doc_unknown'.

        Returns:
            List of structured Chunk instances with metadata and deterministic IDs.
        """
        if not pages:
            return []

        resolved_doc_id = document_id
        if not resolved_doc_id:
            for p in pages:
                if p.source_document:
                    # Clean filename into a stable identifier (e.g., 'AUTOSAR_HLD.pdf' -> 'AUTOSAR_HLD')
                    resolved_doc_id = Path(p.source_document).stem
                    break
        resolved_doc_id = resolved_doc_id or "doc_unknown"

        all_chunks: list[Chunk] = []
        global_chunk_idx = 0
        current_section: Optional[str] = None
        heading_hierarchy: list[str] = []

        for page in pages:
            page_chunks = self._chunk_page(
                page=page,
                document_id=resolved_doc_id,
                start_chunk_index=global_chunk_idx,
                current_section=current_section,
                heading_hierarchy=heading_hierarchy,
            )

            # Update section context carried over to the next page
            if page_chunks:
                last_chunk = page_chunks[-1]
                current_section = last_chunk.section_title
                heading_hierarchy = list(last_chunk.metadata.heading_hierarchy)

            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        logger.info(
            "Chunked document '%s' (%d pages) into %d chunks",
            resolved_doc_id,
            len(pages),
            len(all_chunks),
        )
        return all_chunks

    def _chunk_page(
        self,
        page: CleanedPage,
        document_id: str,
        start_chunk_index: int,
        current_section: Optional[str],
        heading_hierarchy: list[str],
    ) -> list[Chunk]:
        """Process an individual page into chunks respecting headings and tables."""
        page_chunks: list[Chunk] = []
        chunk_idx = start_chunk_index

        active_section = current_section
        active_hierarchy = list(heading_hierarchy)

        # Buffer for accumulating paragraph blocks under the active section
        buffer_text_blocks: list[str] = []
        buffer_char_count = 0

        def flush_buffer() -> None:
            nonlocal chunk_idx, buffer_text_blocks, buffer_char_count
            if not buffer_text_blocks:
                return

            combined_text = "\n\n".join(buffer_text_blocks).strip()
            buffer_text_blocks = []
            buffer_char_count = 0

            if not combined_text:
                return

            # If combined text exceeds max_chunk_chars, split using fallback sliding window
            if len(combined_text) > self.max_chunk_chars:
                sub_chunks = self._split_oversized_text(
                    text=combined_text,
                    document_id=document_id,
                    page=page,
                    section_title=active_section,
                    heading_hierarchy=active_hierarchy,
                    start_chunk_index=chunk_idx,
                    is_table=False,
                )
                page_chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
            else:
                chunk = self._create_chunk(
                    text=combined_text,
                    document_id=document_id,
                    page=page,
                    section_title=active_section,
                    heading_hierarchy=active_hierarchy,
                    chunk_index=chunk_idx,
                    is_table=False,
                )
                page_chunks.append(chunk)
                chunk_idx += 1

        # Iterate over spatial blocks on the page
        for block in page.blocks:
            block_text = block.text.strip()
            if not block_text:
                continue

            # Case A: Heading block -> Boundary trigger
            if block.is_heading:
                # Flush existing buffer before changing section
                flush_buffer()

                active_section = block_text
                # Maintain heading hierarchy based on heading level if present
                level = block.heading_level or 1
                if level <= len(active_hierarchy):
                    active_hierarchy = active_hierarchy[: level - 1]
                active_hierarchy.append(block_text)

                # Include the heading itself in the buffer to provide anchored context
                buffer_text_blocks.append(block_text)
                buffer_char_count += len(block_text)
                continue

            # Case B: Table block -> Preserve table contiguity
            if block.is_table_like or self._is_table_text(block_text):
                # Flush any preceding narrative text
                flush_buffer()

                if len(block_text) > self.max_chunk_chars:
                    # Oversized table: split along row boundaries
                    table_chunks = self._split_oversized_table(
                        table_text=block_text,
                        document_id=document_id,
                        page=page,
                        section_title=active_section,
                        heading_hierarchy=active_hierarchy,
                        start_chunk_index=chunk_idx,
                    )
                    page_chunks.extend(table_chunks)
                    chunk_idx += len(table_chunks)
                else:
                    # Contiguous table chunk
                    chunk = self._create_chunk(
                        text=block_text,
                        document_id=document_id,
                        page=page,
                        section_title=active_section,
                        heading_hierarchy=active_hierarchy,
                        chunk_index=chunk_idx,
                        is_table=True,
                    )
                    page_chunks.append(chunk)
                    chunk_idx += 1
                continue

            # Case C: Standard paragraph / narrative block
            projected_size = buffer_char_count + len(block_text) + (2 if buffer_text_blocks else 0)

            if projected_size > self.max_chunk_chars:
                if buffer_char_count >= self.min_chunk_chars:
                    flush_buffer()
                    buffer_text_blocks.append(block_text)
                    buffer_char_count = len(block_text)
                else:
                    # Buffer was small, but adding this block exceeds limit -> add and flush
                    buffer_text_blocks.append(block_text)
                    buffer_char_count += len(block_text)
                    flush_buffer()
            else:
                buffer_text_blocks.append(block_text)
                buffer_char_count = projected_size

        # Flush any remaining text on the page
        flush_buffer()

        return page_chunks

    def _split_oversized_text(
        self,
        text: str,
        document_id: str,
        page: CleanedPage,
        section_title: Optional[str],
        heading_hierarchy: list[str],
        start_chunk_index: int,
        is_table: bool = False,
    ) -> list[Chunk]:
        """Sliding-window fallback split for oversized blocks with configurable overlap."""
        chunks: list[Chunk] = []
        chunk_idx = start_chunk_index

        if self.respect_sentence_boundaries:
            sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text) if s.strip()]
        else:
            sentences = [line.strip() for line in text.split("\n") if line.strip()]

        if not sentences:
            sentences = [text]

        curr_window: list[str] = []
        curr_chars = 0
        i = 0

        while i < len(sentences):
            sentence = sentences[i]
            sentence_len = len(sentence)

            # If a single sentence is itself larger than max_chunk_chars, hard split by character window
            if sentence_len > self.max_chunk_chars:
                if curr_window:
                    window_text = " ".join(curr_window)
                    chunks.append(
                        self._create_chunk(
                            text=window_text,
                            document_id=document_id,
                            page=page,
                            section_title=section_title,
                            heading_hierarchy=heading_hierarchy,
                            chunk_index=chunk_idx,
                            is_table=is_table,
                        )
                    )
                    chunk_idx += 1
                    curr_window = []
                    curr_chars = 0

                # Split the single oversized sentence by character window with overlap
                start = 0
                while start < sentence_len:
                    end = min(start + self.max_chunk_chars, sentence_len)
                    part = sentence[start:end].strip()
                    chunks.append(
                        self._create_chunk(
                            text=part,
                            document_id=document_id,
                            page=page,
                            section_title=section_title,
                            heading_hierarchy=heading_hierarchy,
                            chunk_index=chunk_idx,
                            is_table=is_table,
                        )
                    )
                    chunk_idx += 1
                    if end >= sentence_len:
                        break
                    start += self.max_chunk_chars - self.overlap_chars

                i += 1
                continue

            # Standard sentence accumulation
            projected = curr_chars + sentence_len + (1 if curr_window else 0)
            if projected <= self.max_chunk_chars:
                curr_window.append(sentence)
                curr_chars = projected
                i += 1
            else:
                # Window is full -> emit chunk
                window_text = " ".join(curr_window)
                chunks.append(
                    self._create_chunk(
                        text=window_text,
                        document_id=document_id,
                        page=page,
                        section_title=section_title,
                        heading_hierarchy=heading_hierarchy,
                        chunk_index=chunk_idx,
                        is_table=is_table,
                    )
                )
                chunk_idx += 1

                # Step backwards to create overlap
                overlap_accum = 0
                overlap_window: list[str] = []
                for prev in reversed(curr_window):
                    if overlap_accum + len(prev) <= self.overlap_chars:
                        overlap_window.insert(0, prev)
                        overlap_accum += len(prev) + 1
                    else:
                        break

                curr_window = overlap_window
                curr_chars = sum(len(s) for s in curr_window) + max(0, len(curr_window) - 1)

        # Emit remaining window
        if curr_window:
            window_text = " ".join(curr_window)
            chunks.append(
                self._create_chunk(
                    text=window_text,
                    document_id=document_id,
                    page=page,
                    section_title=section_title,
                    heading_hierarchy=heading_hierarchy,
                    chunk_index=chunk_idx,
                    is_table=is_table,
                )
            )

        return chunks

    def _split_oversized_table(
        self,
        table_text: str,
        document_id: str,
        page: CleanedPage,
        section_title: Optional[str],
        heading_hierarchy: list[str],
        start_chunk_index: int,
    ) -> list[Chunk]:
        """Split an oversized table along newline/row boundaries preserving table context."""
        rows = table_text.split("\n")
        chunks: list[Chunk] = []
        chunk_idx = start_chunk_index

        header_row = rows[0] if rows else ""
        current_rows = [header_row] if header_row else []
        current_chars = len(header_row)

        for row in rows[1:]:
            row_len = len(row)
            if current_chars + row_len + 1 > self.max_chunk_chars:
                if current_rows:
                    part_text = "\n".join(current_rows)
                    chunks.append(
                        self._create_chunk(
                            text=part_text,
                            document_id=document_id,
                            page=page,
                            section_title=section_title,
                            heading_hierarchy=heading_hierarchy,
                            chunk_index=chunk_idx,
                            is_table=True,
                        )
                    )
                    chunk_idx += 1
                # Start new chunk with header row if available for table context
                current_rows = [header_row, row] if header_row else [row]
                current_chars = sum(len(r) for r in current_rows) + len(current_rows) - 1
            else:
                current_rows.append(row)
                current_chars += row_len + 1

        if current_rows:
            part_text = "\n".join(current_rows)
            chunks.append(
                self._create_chunk(
                    text=part_text,
                    document_id=document_id,
                    page=page,
                    section_title=section_title,
                    heading_hierarchy=heading_hierarchy,
                    chunk_index=chunk_idx,
                    is_table=True,
                )
            )

        return chunks

    def _create_chunk(
        self,
        text: str,
        document_id: str,
        page: CleanedPage,
        section_title: Optional[str],
        heading_hierarchy: list[str],
        chunk_index: int,
        is_table: bool,
    ) -> Chunk:
        """Construct a validated Chunk with deterministic chunk_id, offsets, and metadata."""
        # Calculate character offset within page text
        start_char = 0
        end_char = len(text)
        if page.text:
            idx = page.text.find(text)
            if idx != -1:
                start_char = idx
                end_char = idx + len(text)

        # Estimate tokens (average 4 chars per token)
        token_estimate = max(1, len(text) // 4)

        # Deterministic chunk ID: {document_id}_p{page_number}_c{chunk_index}
        chunk_id = f"{document_id}_p{page.page_number}_c{chunk_index}"

        metadata = ChunkMetadata(
            document_id=document_id,
            page_number=page.page_number,
            section_title=section_title,
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            token_count_estimate=token_estimate,
            is_table=is_table,
            heading_hierarchy=list(heading_hierarchy),
            extra_metadata={
                "source_document": page.source_document or "",
                "char_length": len(text),
            },
        )

        return Chunk(
            chunk_id=chunk_id,
            text=text,
            document_id=document_id,
            page_number=page.page_number,
            section_title=section_title,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata,
        )

    def _is_table_text(self, text: str) -> bool:
        """Heuristic check for tabular text format (pipes or tabs across multiple lines)."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) < 2:
            return False

        pipe_lines = sum(1 for line in lines if line.count("|") >= 2)
        tab_lines = sum(1 for line in lines if "\t" in line)

        return (pipe_lines >= 2) or (tab_lines >= 2)
