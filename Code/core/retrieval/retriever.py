"""Semantic retriever module with relevance thresholding and metadata filtering (Stage 7).

Coordinates dense query embedding with persistent vector search to deliver
ranked, grounded context chunks with full architectural citation tracking.
"""

from typing import Any, Optional
from core.embeddings.embedder import BaseEmbedder
from core.logger import get_logger
from core.retrieval.models import RetrievalResult, RetrievedChunk
from core.vector_store.chroma_manager import ChromaManager

logger = get_logger(__name__)


class SemanticRetriever:
    """Retrieves relevant document chunks from ChromaDB for engineering queries."""

    def __init__(
        self,
        vector_store: ChromaManager,
        embedder: BaseEmbedder,
        default_top_k: int = 5,
        default_relevance_threshold: Optional[float] = None,
    ) -> None:
        """Initialize the semantic retriever.

        Args:
            vector_store: ChromaManager instance managing chunk collections.
            embedder: BaseEmbedder instance for encoding queries into dense vectors.
            default_top_k: Default number of top nearest neighbor chunks to return.
            default_relevance_threshold: Optional minimum cosine similarity threshold
                below which chunks are filtered out as low-relevance noise.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.default_top_k = default_top_k
        self.default_relevance_threshold = default_relevance_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> RetrievalResult:
        """Retrieve ranked context chunks matching an engineering question.

        Args:
            query: Natural language engineering query string.
            top_k: Maximum number of chunks to retrieve (overrides default).
            relevance_threshold: Minimum cosine similarity score required (overrides default).
            document_id: Optional filter limiting search to a specific document.
            filters: Optional dictionary of additional metadata filter conditions
                (e.g., {"is_table": True} or {"page_number": 2}).

        Returns:
            RetrievalResult containing ranked RetrievedChunk instances and query metadata.
        """
        clean_query = query.strip() if query else ""
        effective_top_k = top_k if top_k is not None else self.default_top_k
        effective_threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else self.default_relevance_threshold
        )

        if not clean_query:
            logger.warning("Empty query passed to retriever.")
            return RetrievalResult(
                query=query,
                chunks=[],
                total_found=0,
                top_k=effective_top_k,
                relevance_threshold=effective_threshold,
            )

        # 1. Embed user query into dense vector
        query_embedding = self.embedder.embed_text(clean_query)

        # 2. Construct ChromaDB metadata filter
        where_clause = self._build_where_clause(
            document_id=document_id,
            additional_filters=filters,
        )

        # 3. Search vector store
        raw_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            where=where_clause,
        )

        # 4. Convert and apply relevance thresholding
        ranked_chunks: list[RetrievedChunk] = []
        for r in raw_results:
            if effective_threshold is not None and r.similarity < effective_threshold:
                logger.debug(
                    "Filtered chunk '%s' with similarity %.4f below threshold %.4f",
                    r.chunk_id,
                    r.similarity,
                    effective_threshold,
                )
                continue

            ranked_chunks.append(
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    document_id=r.document_id,
                    page_number=r.page_number,
                    section_title=r.section_title,
                    start_char=r.start_char,
                    end_char=r.end_char,
                    similarity_score=r.similarity,
                    distance=r.distance,
                    is_table=r.is_table,
                    metadata=r.metadata,
                )
            )

        # Sort descending by similarity score
        ranked_chunks.sort(key=lambda c: c.similarity_score, reverse=True)

        logger.info(
            "Retrieved %d chunks for query '%s' (top_k=%d, threshold=%s)",
            len(ranked_chunks),
            clean_query[:50],
            effective_top_k,
            str(effective_threshold),
        )

        return RetrievalResult(
            query=clean_query,
            chunks=ranked_chunks,
            total_found=len(ranked_chunks),
            top_k=effective_top_k,
            relevance_threshold=effective_threshold,
        )

    def _build_where_clause(
        self,
        document_id: Optional[str] = None,
        additional_filters: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Build ChromaDB compatible where clause from parameters."""
        conditions: dict[str, Any] = {}

        if document_id:
            conditions["document_id"] = document_id

        if additional_filters:
            for k, v in additional_filters.items():
                conditions[k] = v

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions

        # Multiple conditions: format as {"$and": [{k1: v1}, {k2: v2}]}
        return {"$and": [{k: v} for k, v in conditions.items()]}
