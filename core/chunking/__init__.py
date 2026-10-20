"""Structural and hierarchical chunking package for AUTOSAR HLD documents."""

from core.chunking.hierarchical_chunker import HierarchicalChunker
from core.chunking.models import Chunk, ChunkMetadata

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "HierarchicalChunker",
]
