"""Data models for semantic retrieval and relevance filtering (Stage 7)."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A ranked chunk retrieved from vector search with similarity and citation metadata."""

    chunk_id: str
    text: str
    document_id: str
    page_number: int  # 1-indexed
    section_title: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    similarity_score: float  # Cosine similarity in [0.0, 1.0]
    distance: float
    is_table: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def citation_source(self) -> str:
        """Human-readable citation label for engineering review."""
        sec = f", Section: {self.section_title}" if self.section_title else ""
        return f"[Doc: {self.document_id}, Page {self.page_number}{sec}]"


class RetrievalResult(BaseModel):
    """Container holding ranked retrieved chunks and query execution statistics."""

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_found: int = 0
    top_k: int = 5
    relevance_threshold: Optional[float] = None
