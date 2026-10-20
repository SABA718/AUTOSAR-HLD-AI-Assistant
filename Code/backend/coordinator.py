"""Shared pipeline coordinator reusing core services (Phase 7).

Orchestrates:
1. Ingestion: PDF parsing -> cleaning -> chunking -> embedding -> ChromaDB + entity extraction -> SQLite.
2. Retrieval & Grounded Q&A: Query -> dense embedding -> Chroma retrieval -> GroundedPromptBuilder -> LLM -> CitationVerifier.
3. Architecture Entities: SQLite DAO retrieval.
"""

from pathlib import Path
from typing import Any, BinaryIO, Optional, Union
from config.settings import get_settings
from core.chunking.hierarchical_chunker import HierarchicalChunker
from core.embeddings.embedder import SentenceTransformerEmbedder
from core.extraction.entity_extractor import EntityExtractor
from core.ingestion.cleaner import TextCleaner
from core.ingestion.pdf_extractor import PDFExtractor
from core.llm.base import BaseLLMProvider
from core.llm.factory import get_llm_provider
from core.logger import get_logger
from core.rag.citation_engine import CitationVerifier
from core.rag.models import GroundedAnswer
from core.rag.prompt_builder import GroundedPromptBuilder
from core.retrieval.retriever import SemanticRetriever
from core.vector_store.chroma_manager import ChromaManager
from storage.database import DatabaseManager

logger = get_logger(__name__)


class PipelineCoordinator:
    """Singleton-style coordinator for all AUTOSAR analysis subsystems."""

    def __init__(
        self,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        vector_store: Optional[ChromaManager] = None,
        db_manager: Optional[DatabaseManager] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ) -> None:
        """Initialize pipeline coordinator reusing core singleton services."""
        self.settings = get_settings()

        # Core services
        self.pdf_extractor = PDFExtractor()
        self.cleaner = TextCleaner()
        self.chunker = HierarchicalChunker()
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.vector_store = vector_store or ChromaManager()
        self.db_manager = db_manager or DatabaseManager()

        # RAG services
        self.retriever = SemanticRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            default_top_k=5,
        )
        self.prompt_builder = GroundedPromptBuilder()
        self.citation_verifier = CitationVerifier()

        # LLM Provider (defaults to mock unless specified in config or caller)
        self.default_llm_provider = llm_provider or get_llm_provider()

        # Entity Extractor
        self.entity_extractor = EntityExtractor(
            llm_provider=self.default_llm_provider,
            db_manager=self.db_manager,
        )

    def process_pdf(
        self,
        source: Union[str, Path, bytes, BinaryIO],
        filename: str,
        document_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ingest and index a PDF end-to-end.

        Steps:
        1. Extract text and blocks via PyMuPDF.
        2. Clean text and normalize line breaks.
        3. Partition into structural chunks.
        4. Compute vector embeddings.
        5. Upsert into ChromaDB vector store.
        6. Extract Software Components, Ports, and Interfaces and save to SQLite.
        """
        doc_id = document_id or Path(filename).stem
        logger.info("Processing PDF '%s' (doc_id='%s')...", filename, doc_id)

        # 1. Extract
        extracted_doc = self.pdf_extractor.extract(source=source, filename=filename)

        # 2. Clean
        cleaned_pages = self.cleaner.clean_document(extracted_doc)

        # 3. Chunk
        chunks = self.chunker.chunk_document(cleaned_pages, document_id=doc_id)

        # 4. Embed & Index in Vector Store
        if chunks:
            chunk_texts = [c.text for c in chunks]
            embeddings = self.embedder.embed_batch(chunk_texts)
            self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

        # 5. Extract Entities & Persist in SQLite
        entity_result = self.entity_extractor.extract_from_chunks(
            chunks=chunks,
            document_id=doc_id,
            use_llm=False,  # Use deterministic rules for speed; LLM can be triggered if desired
        )

        return {
            "document_id": doc_id,
            "filename": filename,
            "total_pages": len(cleaned_pages),
            "total_chunks": len(chunks),
            "extracted_swcs": len(entity_result.software_components),
            "extracted_ports": len(entity_result.ports),
            "extracted_interfaces": len(entity_result.interfaces),
        }

    def answer_query(
        self,
        question: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
        relevance_threshold: Optional[float] = None,
        provider_name: Optional[str] = None,
    ) -> tuple[GroundedAnswer, BaseLLMProvider, float]:
        """Execute full RAG pipeline for an engineering query.

        Returns:
            Tuple of (GroundedAnswer, provider_instance, latency_seconds).
        """
        provider = (
            get_llm_provider(provider_name=provider_name)
            if provider_name
            else self.default_llm_provider
        )

        # 1. Retrieve
        retrieval_result = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            document_id=document_id,
        )

        # 2. Build Grounded Prompt
        prompt = self.prompt_builder.build_prompt(
            question=question,
            chunks=retrieval_result.chunks,
        )

        # 3. Generate LLM Response
        llm_response = provider.generate(
            messages=prompt.to_messages(),
            temperature=0.0,
        )

        # 4. Verify Grounding & Citations
        grounded_answer = self.citation_verifier.verify(
            response=llm_response,
            retrieved_chunks=retrieval_result.chunks,
        )

        return grounded_answer, provider, llm_response.latency_seconds

    def get_entities(self, document_id: str) -> dict[str, Any]:
        """Fetch extracted architectural entities for a document from SQLite."""
        swcs = self.db_manager.get_software_components(document_id=document_id)
        ports = self.db_manager.get_ports(document_id=document_id)
        interfaces = self.db_manager.get_interfaces(document_id=document_id)

        return {
            "document_id": document_id,
            "software_components": swcs,
            "ports": ports,
            "interfaces": interfaces,
            "total_entities": len(swcs) + len(ports) + len(interfaces),
        }


# Lazy singleton instance
_coordinator_instance: Optional[PipelineCoordinator] = None


def get_coordinator() -> PipelineCoordinator:
    """Obtain or initialize the global pipeline coordinator."""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = PipelineCoordinator()
    return _coordinator_instance
