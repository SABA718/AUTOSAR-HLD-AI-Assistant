"""Data models for structural document chunking (Phase 3A)."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attributes associated with a document chunk."""

    document_id: str
    page_number: int  # 1-indexed
    section_title: Optional[str] = None
    start_char: int
    end_char: int
    chunk_index: int
    token_count_estimate: int = 0
    is_table: bool = False
    heading_hierarchy: list[str] = Field(default_factory=list)
    extra_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert metadata to a flat primitive dictionary suitable for vector storage."""
        flat: dict[str, Any] = {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "section_title": self.section_title or "",
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_index": self.chunk_index,
            "token_count_estimate": self.token_count_estimate,
            "is_table": self.is_table,
            "heading_hierarchy": " > ".join(self.heading_hierarchy) if self.heading_hierarchy else "",
        }
        for k, v in self.extra_metadata.items():
            if isinstance(v, (str, int, float, bool)):
                flat[k] = v
        return flat


class Chunk(BaseModel):
    """A bounded semantic text chunk with deterministic citation and location identifiers."""

    chunk_id: str
    text: str
    document_id: str
    page_number: int  # 1-indexed
    section_title: Optional[str] = None
    start_char: int
    end_char: int
    metadata: ChunkMetadata

    @property
    def is_table(self) -> bool:
        """Indicates whether this chunk represents an architectural table block."""
        return self.metadata.is_table

    @property
    def token_count_estimate(self) -> int:
        """Estimated token count."""
        return self.metadata.token_count_estimate
