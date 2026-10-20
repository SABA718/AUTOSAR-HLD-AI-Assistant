"""FastAPI application for AUTOSAR HLD AI Assistant (Phase 7 Stage 13).

Exposes MVP REST endpoints:
- GET /health
- POST /documents/upload
- POST /rag/query
- GET /entities/{doc_id}
"""

from typing import Optional
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from backend.coordinator import PipelineCoordinator, get_coordinator
from backend.schemas import (
    CitationItem,
    DocumentUploadResponse,
    EntitiesResponse,
    HealthResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)
from core.logger import get_logger

logger = get_logger(__name__)


def create_app(coordinator: Optional[PipelineCoordinator] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        coordinator: Optional PipelineCoordinator instance (for testing/dependency injection).
    """
    app = FastAPI(
        title="AUTOSAR HLD Document Analysis Assistant API",
        version="0.1.0",
        description="Local RAG & Architectural Entity Extraction API for Automotive HLD Documents.",
    )

    # Enable CORS for local Streamlit UI and dev servers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store coordinator in app state
    app.state.coordinator = coordinator or get_coordinator()

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def get_health() -> HealthResponse:
        """Health check and subsystem status."""
        coord: PipelineCoordinator = app.state.coordinator
        return HealthResponse(
            status="ok",
            version="0.1.0",
            llm_provider=coord.default_llm_provider.provider_name,
            vector_store_chunks=coord.vector_store.count(),
        )

    @app.post(
        "/documents/upload",
        response_model=DocumentUploadResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Documents"],
    )
    async def upload_document(
        file: UploadFile = File(...),
        document_id: Optional[str] = Query(None, description="Optional custom document identifier"),
    ) -> DocumentUploadResponse:
        """Upload and index an AUTOSAR HLD PDF document."""
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files (.pdf) are supported.",
            )

        try:
            content = await file.read()
            if not content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty.",
                )

            coord: PipelineCoordinator = app.state.coordinator
            result = coord.process_pdf(
                source=content,
                filename=file.filename,
                document_id=document_id,
            )

            return DocumentUploadResponse(
                document_id=result["document_id"],
                filename=result["filename"],
                total_pages=result["total_pages"],
                total_chunks=result["total_chunks"],
                extracted_swcs=result["extracted_swcs"],
                extracted_ports=result["extracted_ports"],
                extracted_interfaces=result["extracted_interfaces"],
                message="Document ingested and indexed successfully.",
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to process uploaded document '%s': %s", file.filename, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document processing failed: {exc}",
            ) from exc

    @app.post("/rag/query", response_model=RAGQueryResponse, tags=["RAG"])
    def query_rag(payload: RAGQueryRequest) -> RAGQueryResponse:
        """Execute grounded RAG query with citation and evidence verification."""
        coord: PipelineCoordinator = app.state.coordinator
        try:
            grounded_answer, provider, latency = coord.answer_query(
                question=payload.question,
                document_id=payload.document_id,
                top_k=payload.top_k,
                relevance_threshold=payload.relevance_threshold,
                provider_name=payload.llm_provider,
            )

            citations_out = [
                CitationItem(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    page_number=c.page_number,
                    section_title=c.section_title,
                    snippet=c.snippet,
                    is_table=c.is_table,
                    source_label=c.source_label,
                )
                for c in grounded_answer.verified_citations
            ]

            return RAGQueryResponse(
                question=payload.question,
                answer=grounded_answer.answer_text,
                is_grounded=grounded_answer.is_grounded,
                has_insufficient_evidence=grounded_answer.has_insufficient_evidence,
                verified_citations=citations_out,
                invalid_citations=grounded_answer.invalid_citations,
                grounding_warnings=grounded_answer.grounding_warnings,
                provider_name=provider.provider_name,
                model_name=provider.model_name,
                latency_seconds=latency,
            )
        except Exception as exc:
            logger.error("Error evaluating RAG query '%s': %s", payload.question, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query execution failed: {exc}",
            ) from exc

    @app.get("/entities/{doc_id}", response_model=EntitiesResponse, tags=["Entities"])
    def get_document_entities(doc_id: str) -> EntitiesResponse:
        """Fetch extracted architectural entities (SWCs, Ports, Interfaces) for a document."""
        coord: PipelineCoordinator = app.state.coordinator
        try:
            entities = coord.get_entities(document_id=doc_id)
            return EntitiesResponse(
                document_id=entities["document_id"],
                software_components=entities["software_components"],
                ports=entities["ports"],
                interfaces=entities["interfaces"],
                total_entities=entities["total_entities"],
            )
        except Exception as exc:
            logger.error("Error retrieving entities for doc '%s': %s", doc_id, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Entity retrieval failed: {exc}",
            ) from exc

    return app


# Default ASGI entry point
app = create_app()
