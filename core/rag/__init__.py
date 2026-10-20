"""RAG pipeline package for AUTOSAR HLD AI Assistant."""

from core.rag.citation_engine import CitationEngine, CitationVerifier
from core.rag.models import Citation, GroundedAnswer, RAGPrompt
from core.rag.prompt_builder import (
    DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE,
    DEFAULT_SYSTEM_PROMPT,
    GroundedPromptBuilder,
)

__all__ = [
    "Citation",
    "CitationEngine",
    "CitationVerifier",
    "DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE",
    "DEFAULT_SYSTEM_PROMPT",
    "GroundedAnswer",
    "GroundedPromptBuilder",
    "RAGPrompt",
]
