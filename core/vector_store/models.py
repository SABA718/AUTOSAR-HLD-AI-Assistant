"""Data models for ChromaDB vector store operations (Stage 6)."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class VectorSearchResult(BaseModel):
    """Result item returned from a vector similarity search."""

    chunk_id: str
    text: str
    document_id: str
    page_number: int  # 1-indexed
    section_title: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    distance: float
    similarity: float
    is_table: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStoreStats(BaseModel):
    """Diagnostic statistics for a vector collection."""

    collection_name: str
    total_chunks: int
    persist_directory: Optional[str] = None
