"""Data models for grounded RAG prompt construction and citation verification (Stages 8 & 10)."""

from typing import Any, Optional, Union
from pydantic import BaseModel, Field

from core.chunking.models import Chunk
from core.retrieval.models import RetrievedChunk


class RAGPrompt(BaseModel):
    """Encapsulates formatted grounded prompts and reference metadata."""

    system_prompt: str
    user_prompt: str
    chunk_ids: list[str] = Field(default_factory=list)
    chunks: list[Union[RetrievedChunk, Chunk]] = Field(default_factory=list)
    estimated_tokens: int = 0
    insufficient_evidence_phrase: str = (
        "Based on the provided AUTOSAR specification, there is insufficient evidence to answer this question."
    )

    @property
    def full_prompt(self) -> str:
        """Combined prompt text suitable for single-prompt or completion models."""
        return f"{self.system_prompt}\n\n{self.user_prompt}"

    def to_messages(self) -> list[dict[str, str]]:
        """Format as standard role-based chat messages."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


class Citation(BaseModel):
    """A verified source citation linking a claim to an authentic retrieved chunk."""

    chunk_id: str
    document_id: str
    page_number: int  # 1-indexed
    section_title: Optional[str] = None
    snippet: Optional[str] = None
    is_table: bool = False
    source_label: str  # e.g., "[Doc: AUTOSAR_Braking, Page 1, Section: 1. Architecture Overview]"


class GroundedAnswer(BaseModel):
    """Verified engineering answer with factual grounding status and citation traceability."""

    answer_text: str
    verified_citations: list[Citation] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    is_grounded: bool = True
    grounding_warnings: list[str] = Field(default_factory=list)
    has_insufficient_evidence: bool = False
