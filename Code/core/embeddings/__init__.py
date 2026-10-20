"""Dense text embeddings package for AUTOSAR HLD AI Assistant."""

from core.embeddings.embedder import (
    BaseEmbedder,
    SentenceTransformerEmbedder,
    l2_normalize,
)

__all__ = [
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "l2_normalize",
]
