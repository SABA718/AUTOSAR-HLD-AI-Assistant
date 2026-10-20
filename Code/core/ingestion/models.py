"""Data models for PDF ingestion and text cleaning (Phase 2)."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata describing an ingested document."""

    filename: str
    total_pages: int
    file_size_bytes: Optional[int] = None
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[str] = None
    source_path: Optional[str] = None
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class TextBlock(BaseModel):
    """A single spatial text block extracted from a PDF page."""

    block_index: int
    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    is_heading: bool = False
    heading_level: Optional[int] = None
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    is_table_like: bool = False


class PageContent(BaseModel):
    """Raw content extracted from a single PDF page."""

    page_number: int  # 1-indexed
    width: float
    height: float
    text: str
    blocks: list[TextBlock] = Field(default_factory=list)
    tables_count: int = 0
    source_document: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedDocument(BaseModel):
    """Complete document extraction container."""

    metadata: DocumentMetadata
    pages: list[PageContent] = Field(default_factory=list)


class CleanedBlock(BaseModel):
    """A cleaned text block preserving structural semantics."""

    block_index: int
    text: str
    is_heading: bool = False
    heading_level: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    is_table_like: bool = False


class CleanedPage(BaseModel):
    """Normalized page representation with cleaned text and preserved structure."""

    page_number: int  # 1-indexed
    text: str
    blocks: list[CleanedBlock] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    source_document: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
