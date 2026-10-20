"""Unit tests for Phase 4A: Semantic Retrieval using seeded test data."""

from pathlib import Path
from typing import Sequence
import numpy as np
import pytest

from core.chunking.models import Chunk, ChunkMetadata
from core.embeddings.embedder import BaseEmbedder, l2_normalize
from core.retrieval.models import RetrievalResult, RetrievedChunk
from core.retrieval.retriever import SemanticRetriever
from core.vector_store.chroma_manager import ChromaManager


class SeededMockEmbedder(BaseEmbedder):
    """Deterministic mock embedder mapping predefined terms to orthogonal test vectors."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        lower = text.lower()
        if "brake" in lower or "architecture" in lower:
            vec[0] = 1.0  # Direction 1
        elif "port" in lower or "table" in lower:
            vec[1] = 1.0  # Direction 2
        elif "light" in lower or "body" in lower:
            vec[0] = 0.7071
            vec[1] = 0.7071  # Diagonal
        else:
            vec[2] = 1.0  # Direction 3 (Distractor)

        return l2_normalize(vec)

    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


@pytest.fixture
def seeded_environment(tmp_path: Path):
    """Set up ChromaManager and SemanticRetriever pre-seeded with AUTOSAR test chunks."""
    chroma_dir = tmp_path / "seeded_chroma"
    manager = ChromaManager(persist_directory=chroma_dir, collection_name="seeded_test")
    embedder = SeededMockEmbedder()

    # Create 4 diverse chunks
    c1 = Chunk(
        chunk_id="AUTOSAR_Braking_p1_c0",
        text="Software Components define boundaries for automotive brake control units.",
        document_id="AUTOSAR_Braking",
        page_number=1,
        section_title="1. Architecture Overview",
        start_char=0,
        end_char=75,
        metadata=ChunkMetadata(
            document_id="AUTOSAR_Braking",
            page_number=1,
            section_title="1. Architecture Overview",
            start_char=0,
            end_char=75,
            chunk_index=0,
            is_table=False,
            heading_hierarchy=["1. Architecture Overview"],
            extra_metadata={"system": "Braking"},
        ),
    )

    c2 = Chunk(
        chunk_id="AUTOSAR_Braking_p2_c0",
        text="| Port | Direction | Interface |\n| R_WheelSpeed | Require | If_WheelSpeed |",
        document_id="AUTOSAR_Braking",
        page_number=2,
        section_title="2. Port Specifications",
        start_char=0,
        end_char=68,
        metadata=ChunkMetadata(
            document_id="AUTOSAR_Braking",
            page_number=2,
            section_title="2. Port Specifications",
            start_char=0,
            end_char=68,
            chunk_index=0,
            is_table=True,
            heading_hierarchy=["2. Port Specifications"],
            extra_metadata={"system": "Braking"},
        ),
    )

    c3 = Chunk(
        chunk_id="AUTOSAR_Body_p1_c0",
        text="Headlamp controller manages exterior lighting signals for body electronics.",
        document_id="AUTOSAR_BodyControl",
        page_number=1,
        section_title="1. Lighting Control SWC",
        start_char=0,
        end_char=76,
        metadata=ChunkMetadata(
            document_id="AUTOSAR_BodyControl",
            page_number=1,
            section_title="1. Lighting Control SWC",
            start_char=0,
            end_char=76,
            chunk_index=0,
            is_table=False,
            heading_hierarchy=["1. Lighting Control SWC"],
            extra_metadata={"system": "Body"},
        ),
    )

    c4 = Chunk(
        chunk_id="AUTOSAR_Body_p5_c0",
        text="Appendix and document revision history table.",
        document_id="AUTOSAR_BodyControl",
        page_number=5,
        section_title="5. Appendix",
        start_char=0,
        end_char=46,
        metadata=ChunkMetadata(
            document_id="AUTOSAR_BodyControl",
            page_number=5,
            section_title="5. Appendix",
            start_char=0,
            end_char=46,
            chunk_index=0,
            is_table=False,
            heading_hierarchy=["5. Appendix"],
            extra_metadata={"system": "Body"},
        ),
    )

    chunks = [c1, c2, c3, c4]
    embeddings = embedder.embed_batch([c.text for c in chunks])
    manager.add_chunks(chunks, embeddings)

    retriever = SemanticRetriever(
        vector_store=manager,
        embedder=embedder,
        default_top_k=3,
        default_relevance_threshold=None,
    )
    return retriever


def test_retrieval_ranking_and_metadata_preservation(seeded_environment: SemanticRetriever):
    """Verify that retrieval accurately ranks chunks and preserves all citation metadata."""
    result = seeded_environment.retrieve(query="brake control architecture", top_k=2)

    assert isinstance(result, RetrievalResult)
    assert result.total_found == 2
    assert len(result.chunks) == 2

    top = result.chunks[0]
    assert isinstance(top, RetrievedChunk)
    assert top.chunk_id == "AUTOSAR_Braking_p1_c0"
    assert top.document_id == "AUTOSAR_Braking"
    assert top.page_number == 1
    assert top.section_title == "1. Architecture Overview"
    assert top.start_char == 0
    assert top.end_char == 75
    assert top.is_table is False
    assert top.similarity_score == pytest.approx(1.0, rel=1e-3)
    assert "Doc: AUTOSAR_Braking, Page 1" in top.citation_source


def test_configurable_top_k(seeded_environment: SemanticRetriever):
    """Verify that top_k restricts the number of returned chunks."""
    res_1 = seeded_environment.retrieve(query="architecture", top_k=1)
    assert len(res_1.chunks) == 1

    res_3 = seeded_environment.retrieve(query="architecture", top_k=3)
    assert len(res_3.chunks) == 3


def test_relevance_threshold_filtering(seeded_environment: SemanticRetriever):
    """Verify that chunks with similarity below the threshold are discarded."""
    # Query for 'brake architecture' aligns with c1 (sim ≈ 1.0) and partially c3 (sim ≈ 0.707)
    # With threshold 0.85, only c1 should survive
    result = seeded_environment.retrieve(
        query="brake control architecture",
        top_k=4,
        relevance_threshold=0.85,
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "AUTOSAR_Braking_p1_c0"
    assert result.chunks[0].similarity_score >= 0.85

    # With very strict threshold 0.9999, non-exact matches are discarded
    result_strict = seeded_environment.retrieve(
        query="completely unknown unrelated query xyz",
        relevance_threshold=0.99,
    )
    # The unknown query points to Direction 3 which matches only distractor c4
    for c in result_strict.chunks:
        assert c.similarity_score >= 0.99


def test_document_metadata_filtering(seeded_environment: SemanticRetriever):
    """Verify filtering by document_id and additional metadata filters."""
    # 1. Filter by document_id
    res_braking = seeded_environment.retrieve(
        query="automotive software",
        top_k=5,
        document_id="AUTOSAR_Braking",
    )
    for c in res_braking.chunks:
        assert c.document_id == "AUTOSAR_Braking"

    # 2. Filter for tables only
    res_tables = seeded_environment.retrieve(
        query="port specification table",
        top_k=5,
        filters={"is_table": True},
    )
    assert len(res_tables.chunks) == 1
    assert res_tables.chunks[0].chunk_id == "AUTOSAR_Braking_p2_c0"
    assert res_tables.chunks[0].is_table is True

    # 3. Combined filter
    res_combined = seeded_environment.retrieve(
        query="port specification table",
        top_k=5,
        document_id="AUTOSAR_Braking",
        filters={"page_number": 2},
    )
    assert len(res_combined.chunks) == 1
    assert res_combined.chunks[0].page_number == 2


def test_empty_query_returns_empty_result(seeded_environment: SemanticRetriever):
    """Verify graceful handling of blank or whitespace-only queries."""
    res_empty = seeded_environment.retrieve(query="")
    assert res_empty.total_found == 0
    assert len(res_empty.chunks) == 0

    res_whitespace = seeded_environment.retrieve(query="   \t\n  ")
    assert res_whitespace.total_found == 0
    assert len(res_whitespace.chunks) == 0
