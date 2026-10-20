"""ChromaDB vector store package for AUTOSAR HLD AI Assistant."""

from core.vector_store.chroma_manager import ChromaManager
from core.vector_store.models import VectorSearchResult, VectorStoreStats

__all__ = [
    "ChromaManager",
    "VectorSearchResult",
    "VectorStoreStats",
]
