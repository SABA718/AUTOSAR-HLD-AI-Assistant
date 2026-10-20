"""Persistent ChromaDB vector store manager (Stage 6).

Handles collection lifecycle, chunk upserts with dense embeddings,
metadata filtering, and semantic similarity search with preserved citations.
"""

from pathlib import Path
from typing import Any, Optional, Sequence, Union
import chromadb
from chromadb.api.models.Collection import Collection

from config.settings import get_settings
from core.chunking.models import Chunk
from core.logger import get_logger
from core.vector_store.models import VectorSearchResult, VectorStoreStats

logger = get_logger(__name__)

DEFAULT_COLLECTION_NAME = "autosar_chunks"


class ChromaManager:
    """Manages persistent vector indexing and similarity retrieval in ChromaDB."""

    def __init__(
        self,
        persist_directory: Optional[Union[Path, str]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Initialize persistent ChromaDB client and collection.

        Args:
            persist_directory: Local storage folder for ChromaDB SQLite & HNSW files.
                Defaults to settings.chroma_dir.
            collection_name: Name of the vector collection.
        """
        settings = get_settings()
        self.persist_directory = Path(persist_directory or settings.chroma_dir)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        logger.info(
            "Initializing ChromaDB PersistentClient at '%s' (collection: '%s')",
            self.persist_directory,
            self.collection_name,
        )

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[list[float]],
    ) -> int:
        """Add or update chunks with dense embeddings and metadata in the collection.

        Args:
            chunks: Sequence of Chunk instances to insert.
            embeddings: Sequence of corresponding normalized float vectors.

        Returns:
            Number of chunks upserted.

        Raises:
            ValueError: If chunks and embeddings lengths do not match.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        if not chunks:
            return 0

        ids: list[str] = [c.chunk_id for c in chunks]
        documents: list[str] = [c.text for c in chunks]
        metadatas: list[dict[str, Any]] = [c.metadata.to_flat_dict() for c in chunks]
        embedding_list: list[list[float]] = list(embeddings)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embedding_list,
        )

        logger.info(
            "Upserted %d chunks into collection '%s' (total collection count: %d)",
            len(chunks),
            self.collection_name,
            self.count(),
        )
        return len(chunks)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        """Perform nearest-neighbor similarity search using a query vector.

        Args:
            query_embedding: Dense embedding vector of the search query.
            top_k: Maximum number of nearest neighbors to retrieve.
            where: Optional metadata filter dict (e.g., {"document_id": "doc1"}).

        Returns:
            List of VectorSearchResult items ranked by cosine similarity.
        """
        total_items = self.count()
        if total_items == 0:
            return []

        n_results = min(top_k, total_items)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        search_results: list[VectorSearchResult] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            chunk_id = ids[i]
            text = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = float(distances[i]) if i < len(distances) else 1.0

            # For cosine distance: distance in [0, 2], similarity = 1.0 - distance
            similarity = max(0.0, min(1.0, 1.0 - dist))

            search_results.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    document_id=str(meta.get("document_id", "")),
                    page_number=int(meta.get("page_number", 1)),
                    section_title=meta.get("section_title") or None,
                    start_char=int(meta.get("start_char", 0)),
                    end_char=int(meta.get("end_char", 0)),
                    distance=dist,
                    similarity=similarity,
                    is_table=bool(meta.get("is_table", False)),
                    metadata=meta,
                )
            )

        return search_results

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> None:
        """Delete specific chunks by their IDs."""
        if not chunk_ids:
            return
        self.collection.delete(ids=list(chunk_ids))
        logger.info("Deleted %d chunk IDs from '%s'", len(chunk_ids), self.collection_name)

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks associated with a specific document identifier."""
        self.collection.delete(where={"document_id": document_id})
        logger.info("Deleted document '%s' from '%s'", document_id, self.collection_name)

    def reset_collection(self) -> None:
        """Purge and recreate the vector collection."""
        logger.warning("Resetting collection '%s'...", self.collection_name)
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return the number of stored vector chunks in the collection."""
        return self.collection.count()

    def get_stats(self) -> VectorStoreStats:
        """Return diagnostic statistics about the vector collection."""
        return VectorStoreStats(
            collection_name=self.collection_name,
            total_chunks=self.count(),
            persist_directory=str(self.persist_directory),
        )
