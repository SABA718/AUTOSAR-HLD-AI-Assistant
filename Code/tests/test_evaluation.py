"""Automated evaluation tests for Phase 8.

Verifies:
1. Golden evaluation dataset loading and structure.
2. Pipeline evaluation across all 10 questions using MockLLM.
3. Retrieval success on answerable questions.
4. Citation validity and groundedness verification.
5. Negative grounding behavior when the document does not contain the answer.
"""

from pathlib import Path
import pymupdf
import pytest

from core.chunking.hierarchical_chunker import HierarchicalChunker
from core.embeddings.embedder import SentenceTransformerEmbedder
from core.evaluation.evaluator import PipelineEvaluator
from core.ingestion.cleaner import TextCleaner
from core.ingestion.pdf_extractor import PDFExtractor
from core.llm.mock_provider import MockLLMProvider
from core.rag.citation_engine import CitationVerifier
from core.rag.prompt_builder import DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE, GroundedPromptBuilder
from core.retrieval.retriever import SemanticRetriever
from core.vector_store.chroma_manager import ChromaManager


def generate_sample_hld_pdf() -> bytes:
    """Generate in-memory sample AUTOSAR HLD document matching the golden dataset."""
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
def indexed_test_pipeline(tmp_path: Path):
    """Fixture providing an indexed vector store and pipeline evaluator."""
    chroma_dir = tmp_path / "eval_chroma"
    vstore = ChromaManager(persist_directory=chroma_dir, collection_name="eval_collection")
    embedder = SentenceTransformerEmbedder()
    retriever = SemanticRetriever(vector_store=vstore, embedder=embedder, default_top_k=3)
    mock_llm = MockLLMProvider()

    # Ingest synthetic document
    pdf_bytes = generate_sample_hld_pdf()
    extractor = PDFExtractor()
    cleaner = TextCleaner()
    chunker = HierarchicalChunker()

    doc = extractor.extract(source=pdf_bytes, filename="AUTOSAR_HLD_Sample_BrakingSystem.pdf")
    pages = cleaner.clean_document(doc)
    chunks = chunker.chunk_document(pages, document_id="AUTOSAR_HLD_Sample_BrakingSystem")

    chunk_texts = [c.text for c in chunks]
    embeddings = embedder.embed_batch(chunk_texts)
    vstore.add_chunks(chunks=chunks, embeddings=embeddings)

    evaluator = PipelineEvaluator(
        retriever=retriever,
        prompt_builder=GroundedPromptBuilder(),
        citation_verifier=CitationVerifier(),
        llm_provider=mock_llm,
    )

    return evaluator, vstore


class TestPhase8Evaluation:
    """Evaluation test suite."""

    def test_golden_dataset_structure(self) -> None:
        """Verify the golden dataset JSON exists and contains 10 items."""
        dataset_path = Path(__file__).parent.parent / "data" / "golden_dataset.json"
        assert dataset_path.exists(), "golden_dataset.json must exist in data directory"

        import json
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        assert len(questions) == 10
        answerable = [q for q in questions if q["is_answerable"]]
        unanswerable = [q for q in questions if not q["is_answerable"]]
        assert len(answerable) == 7
        assert len(unanswerable) == 3

    def test_full_golden_dataset_evaluation(self, indexed_test_pipeline) -> None:
        """Run evaluator across all 10 questions and assert key metrics."""
        evaluator, _ = indexed_test_pipeline
        dataset_path = Path(__file__).parent.parent / "data" / "golden_dataset.json"

        report = evaluator.evaluate_dataset(dataset_path=dataset_path)

        assert report.total_questions == 10
        assert report.answerable_count == 7
        assert report.unanswerable_count == 3

        # Assert metrics meet acceptable thresholds
        assert report.retrieval_success_rate >= 0.8
        assert report.overall_groundedness_rate >= 0.8
        assert report.negative_grounding_accuracy == 1.0
        assert report.citation_validity_rate >= 0.8

        summary = report.to_summary_table()
        assert "Retrieval Success Rate" in summary
        assert "Negative Grounding Accuracy" in summary

    def test_negative_grounding_unanswerable_query(self, indexed_test_pipeline) -> None:
        """Explicit test for negative grounding when the document lacks the answer."""
        evaluator, _ = indexed_test_pipeline

        unanswerable_item = {
            "id": "q_neg_test",
            "question": "What is the AES-256 decryption key for the telematics unit?",
            "is_answerable": False,
            "expected_chunk_ids": [],
            "expected_keywords": [],
        }

        result = evaluator.evaluate_item(
            item=unanswerable_item,
            document_id="AUTOSAR_HLD_Sample_BrakingSystem",
            relevance_threshold=0.8,  # High threshold guarantees no false chunks
        )

        assert result.is_answerable is False
        assert result.negative_grounding_correct is True
        assert result.has_insufficient_evidence is True
        assert DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE.lower() in result.answer_text.lower()
        assert result.is_grounded is True
