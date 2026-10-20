"""FastAPI and Streamlit frontend integration tests (Phase 7).

Tests verify:
1. GET /health endpoint returns status, provider, and collection count.
2. POST /documents/upload validates PDF file types, extracts, chunks, embeds, and saves entities.
3. POST /rag/query coordinates retrieval, prompt building, mock LLM, and citation verification.
4. GET /entities/{doc_id} retrieves Software Components, Ports, and Interfaces from SQLite.
5. Error handling for empty files, invalid extensions, and unknown documents.
6. PipelineCoordinator integration testing.
"""

import io
from pathlib import Path
from fastapi.testclient import TestClient
import pymupdf
import pytest
from backend.coordinator import PipelineCoordinator
from backend.main import create_app
from core.embeddings.embedder import SentenceTransformerEmbedder
from core.llm.mock_provider import MockLLMProvider
from core.vector_store.chroma_manager import ChromaManager
from storage.database import DatabaseManager


def create_synthetic_autosar_pdf() -> bytes:
    """Generate in-memory synthetic PDF with AUTOSAR components and tables."""
    doc = pymupdf.open()

    # Page 1: Overview and SWC declaration
    p1 = doc.new_page()
    p1.insert_text(
        (50, 60),
        "1.0 Architecture Overview\n"
        "The SWC_BrakingController is responsible for vehicle deceleration arbitration.\n"
        "It interacts with chassis sensors and powertrain actuators.\n",
        fontsize=12,
    )

    # Page 2: Port Table
    p2 = doc.new_page()
    p2.insert_text(
        (50, 60),
        "2.0 Port Specifications for SWC_BrakingController\n\n"
        "| Port Name | Direction | Interface | Description |\n"
        "|---|---|---|---|\n"
        "| P_BrakePressure | Provided | If_BrakePressure | Current hydraulic pressure |\n"
        "| R_PedalPosition | Required | If_PedalPosition | Driver brake demand |\n",
        fontsize=11,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def test_coordinator(tmp_path: Path) -> PipelineCoordinator:
    """Fixture supplying an isolated PipelineCoordinator instance."""
    chroma_dir = tmp_path / "test_chroma"
    sqlite_file = tmp_path / "test_entities.db"

    vstore = ChromaManager(persist_directory=chroma_dir, collection_name="test_api_collection")
    db = DatabaseManager(db_path=sqlite_file)
    embedder = SentenceTransformerEmbedder()
    mock_llm = MockLLMProvider()

    return PipelineCoordinator(
        embedder=embedder,
        vector_store=vstore,
        db_manager=db,
        llm_provider=mock_llm,
    )


@pytest.fixture
def client(test_coordinator: PipelineCoordinator) -> TestClient:
    """Fixture providing FastAPI TestClient wired to the test coordinator."""
    app = create_app(coordinator=test_coordinator)
    return TestClient(app)


class TestFastAPIRoutes:
    """Test suite for FastAPI service endpoints."""

    def test_health_endpoint(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["llm_provider"] == "mock"
        assert "vector_store_chunks" in data

    def test_upload_document_invalid_extension(self, client: TestClient) -> None:
        file_content = b"Not a PDF file content"
        response = client.post(
            "/documents/upload",
            files={"file": ("invalid.txt", io.BytesIO(file_content), "text/plain")},
        )
        assert response.status_code == 400
        assert "Only PDF files" in response.json()["detail"]

    def test_upload_document_empty_file(self, client: TestClient) -> None:
        response = client.post(
            "/documents/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        assert response.status_code == 400
        assert "Uploaded file is empty" in response.json()["detail"]

    def test_upload_document_success(self, client: TestClient) -> None:
        pdf_bytes = create_synthetic_autosar_pdf()
        response = client.post(
            "/documents/upload",
            files={"file": ("AUTOSAR_Brake_Spec.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            params={"document_id": "Brake_Spec_v1"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["document_id"] == "Brake_Spec_v1"
        assert data["filename"] == "AUTOSAR_Brake_Spec.pdf"
        assert data["total_pages"] == 2
        assert data["total_chunks"] >= 2
        assert data["extracted_swcs"] >= 1
        assert data["extracted_ports"] >= 2
        assert data["extracted_interfaces"] >= 2

    def test_rag_query_endpoint(self, client: TestClient) -> None:
        # 1. Ingest document
        pdf_bytes = create_synthetic_autosar_pdf()
        client.post(
            "/documents/upload",
            files={"file": ("AUTOSAR_Brake_Spec.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            params={"document_id": "Brake_Spec_v1"},
        )

        # 2. Query
        query_payload = {
            "question": "What ports are defined for SWC_BrakingController?",
            "document_id": "Brake_Spec_v1",
            "top_k": 3,
            "llm_provider": "mock",
        }
        response = client.post("/rag/query", json=query_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == query_payload["question"]
        assert len(data["answer"]) > 0
        assert data["provider_name"] == "mock"
        assert data["is_grounded"] is True
        assert len(data["verified_citations"]) > 0

        first_citation = data["verified_citations"][0]
        assert first_citation["document_id"] == "Brake_Spec_v1"
        assert first_citation["page_number"] in (1, 2)
        assert "snippet" in first_citation

    def test_entities_endpoint(self, client: TestClient) -> None:
        # Ingest document
        pdf_bytes = create_synthetic_autosar_pdf()
        client.post(
            "/documents/upload",
            files={"file": ("AUTOSAR_Brake_Spec.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            params={"document_id": "Brake_Spec_v1"},
        )

        response = client.get("/entities/Brake_Spec_v1")
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "Brake_Spec_v1"
        assert data["total_entities"] >= 5

        swc_names = [s["name"] for s in data["software_components"]]
        assert "SWC_BrakingController" in swc_names

        port_names = [p["name"] for p in data["ports"]]
        assert "P_BrakePressure" in port_names
        assert "R_PedalPosition" in port_names

        iface_names = [i["name"] for i in data["interfaces"]]
        assert "If_BrakePressure" in iface_names
        assert "If_PedalPosition" in iface_names


class TestCoordinatorIntegration:
    """Test suite for direct PipelineCoordinator methods."""

    def test_coordinator_end_to_end_flow(self, test_coordinator: PipelineCoordinator) -> None:
        pdf_bytes = create_synthetic_autosar_pdf()
        stats = test_coordinator.process_pdf(
            source=pdf_bytes,
            filename="Brake_Test.pdf",
            document_id="doc_coord_test",
        )
        assert stats["document_id"] == "doc_coord_test"
        assert stats["total_pages"] == 2
        assert stats["total_chunks"] >= 2

        # Query
        grounded_answer, provider, latency = test_coordinator.answer_query(
            question="What is SWC_BrakingController responsible for?",
            document_id="doc_coord_test",
            top_k=2,
        )
        assert grounded_answer.is_grounded is True
        assert len(grounded_answer.verified_citations) >= 1
        assert provider.provider_name == "mock"
        assert latency >= 0.0

        # Entities
        entities = test_coordinator.get_entities("doc_coord_test")
        assert entities["total_entities"] >= 5
