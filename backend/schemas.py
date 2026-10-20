"""Pydantic request and response schemas for FastAPI endpoints (Stage 13)."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = "ok"
    version: str = "0.1.0"
    llm_provider: str
    vector_store_chunks: int


class DocumentUploadResponse(BaseModel):
    """Response returned upon processing a PDF document."""

    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    extracted_swcs: int
    extracted_ports: int
    extracted_interfaces: int
    message: str = "Document processed successfully"


class RAGQueryRequest(BaseModel):
    """Engineering query payload."""

    question: str = Field(..., description="Engineering query or question", min_length=1)
    document_id: Optional[str] = Field(None, description="Optional document filter")
    top_k: int = Field(default=5, ge=1, le=20, description="Max context chunks to retrieve")
    relevance_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    llm_provider: Optional[str] = Field(default=None, description="Optional LLM provider override ('mock', 'gemini')")


class CitationItem(BaseModel):
    """Traceable citation model for query response."""

    chunk_id: str
    document_id: str
    page_number: int
    section_title: Optional[str] = None
    snippet: Optional[str] = None
    is_table: bool = False
    source_label: str


class RAGQueryResponse(BaseModel):
    """Response returned for an engineering query."""

    question: str
    answer: str
    is_grounded: bool
    has_insufficient_evidence: bool
    verified_citations: list[CitationItem]
    invalid_citations: list[str]
    grounding_warnings: list[str]
    provider_name: str
    model_name: str
    latency_seconds: float


class EntityItem(BaseModel):
    """Generic entity representation."""

    id: Optional[int] = None
    name: str
    document_id: str
    page_number: int
    section: Optional[str] = None
    extra_details: dict[str, Any] = Field(default_factory=dict)


class EntitiesResponse(BaseModel):
    """Response containing extracted architectural entities for a document."""

    document_id: str
    software_components: list[dict[str, Any]]
    ports: list[dict[str, Any]]
    interfaces: list[dict[str, Any]]
    total_entities: int
