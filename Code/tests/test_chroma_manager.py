"""Unit tests for persistent ChromaDB vector store manager (Stage 6)."""

from pathlib import Path
import numpy as np
import pytest

from core.chunking.models import Chunk, ChunkMetadata
from core.vector_store.chroma_manager import ChromaManager
from core.vector_store.models import VectorSearchResult


def make_test_vector(seed_val: float, dim: int = 384) -> list[float]:
    """Helper to create a reproducible normalized 384-dimensional vector."""
    arr = np.full(dim, seed_val, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks with distinct IDs, pages, and metadata."""
    c1 = Chunk(
        chunk_id="AUTOSAR_HLD_p1_c0",
        text="The Software Component encapsulates application logic for brake torque arbitration.",
        document_id="AUTOSAR_HLD",
        page_number=1,
        section_title="1. Architecture Overview",
        start_char=0,
        end_char=80,
        metadata=ChunkMetadata(
            document_id="AUTOSAR_HLD",
            page_number=1,
            section_title="1. Architecture Overview",
            start_char=0,
            end_char=80,
            chunk_index=0,
            is_table=False,
            heading_hierarchy=["1. Architecture Overview"],
            extra_metadata={"system": "Braking"},
        ),
    )

    c2 = Chunk(
        chunk_id="AUTOSAR_HLD_p2_c0",
        text="| Port | Direction | Interface |\n| R_WheelSpeed | Require | If_WheelSpeed |",
        document_id="AUTOSAR_HLD",
        page_number=2,
        section_title="2. Port Interfaces",
        start_char=0,
        end_char=65,
        metadata=ChunkMetadata(
            document_id="AUTOSAR_HLD",
            page_number=2,
            section_title="2. Port Interfaces",
            start_char=0,
            end_char=65,
            chunk_index=0,
            is_table=True,
            heading_hierarchy=["2. Port Interfaces"],
            extra_metadata={"system": "Sensors"},
        ),
    )

    return [c1, c2]


def test_persistent_collection_lifecycle_and_restart(tmp_path: Path, sample_chunks: list[Chunk]):
    """Verify persistent local storage across client restarts."""
    chroma_dir = tmp_path / "chroma_store"
    manager1 = ChromaManager(persist_directory=chroma_dir, collection_name="test_autosar")
    assert manager1.count() == 0

    vec1 = make_test_vector(1.0)
    vec2 = make_test_vector(2.0)

    # Upsert chunks
    added = manager1.add_chunks(sample_chunks, [vec1, vec2])
    assert added == 2
    assert manager1.count() == 2

    # Instantiate a second manager pointing to the same folder to verify persistence across restarts
    manager2 = ChromaManager(persist_directory=chroma_dir, collection_name="test_autosar")
    assert manager2.count() == 2

    stats = manager2.get_stats()
    assert stats.total_chunks == 2
    assert stats.collection_name == "test_autosar"


def test_similarity_search_and_metadata_preservation(tmp_path: Path, sample_chunks: list[Chunk]):
    """Verify nearest-neighbor retrieval, cosine similarity score, and citation metadata."""
    chroma_dir = tmp_path / "chroma_search"
    manager = ChromaManager(persist_directory=chroma_dir, collection_name="test_search")

    vec1 = make_test_vector(1.0)
    vec2 = make_test_vector(5.0)
    manager.add_chunks(sample_chunks, [vec1, vec2])

    # Search with a query vector close to vec1
    query_vec = make_test_vector(1.0)
    results = manager.similarity_search(query_vec, top_k=2)

    assert len(results) == 2
    top = results[0]
    assert isinstance(top, VectorSearchResult)

    # Validate preserved chunk ID and citation metadata
    assert top.chunk_id == "AUTOSAR_HLD_p1_c0"
    assert top.document_id == "AUTOSAR_HLD"
    assert top.page_number == 1
    assert top.section_title == "1. Architecture Overview"
    assert "Software Component" in top.text
    assert top.is_table is False
    assert top.similarity == pytest.approx(1.0, rel=1e-3)
    assert top.distance == pytest.approx(0.0, abs=1e-3)


def test_metadata_filtering(tmp_path: Path, sample_chunks: list[Chunk]):
    """Verify filtered retrieval using where clauses (e.g., page_number or is_table)."""
    chroma_dir = tmp_path / "chroma_filter"
    manager = ChromaManager(persist_directory=chroma_dir, collection_name="test_filter")

    vec1 = make_test_vector(1.0)
    vec2 = make_test_vector(2.0)
    manager.add_chunks(sample_chunks, [vec1, vec2])

    # Filter for table chunks only
    table_results = manager.similarity_search(
        query_embedding=vec1,
        top_k=5,
        where={"is_table": True},
    )
    assert len(table_results) == 1
    assert table_results[0].chunk_id == "AUTOSAR_HLD_p2_c0"
    assert table_results[0].is_table is True
    assert table_results[0].page_number == 2


def test_deletion_and_reset(tmp_path: Path, sample_chunks: list[Chunk]):
    """Verify delete_by_ids, delete_document, and reset_collection."""
    chroma_dir = tmp_path / "chroma_delete"
    manager = ChromaManager(persist_directory=chroma_dir, collection_name="test_del")

    vec1 = make_test_vector(1.0)
    vec2 = make_test_vector(2.0)
    manager.add_chunks(sample_chunks, [vec1, vec2])
    assert manager.count() == 2

    # Delete single chunk by ID
    manager.delete_by_ids(["AUTOSAR_HLD_p1_c0"])
    assert manager.count() == 1

    # Re-add and delete by document_id
    manager.add_chunks([sample_chunks[0]], [vec1])
    assert manager.count() == 2
    manager.delete_document("AUTOSAR_HLD")
    assert manager.count() == 0

    # Reset collection
    manager.add_chunks(sample_chunks, [vec1, vec2])
    assert manager.count() == 2
    manager.reset_collection()
    assert manager.count() == 0


def test_add_chunks_validation(tmp_path: Path, sample_chunks: list[Chunk]):
    """Verify error on mismatched chunks and embeddings lengths."""
    chroma_dir = tmp_path / "chroma_val"
    manager = ChromaManager(persist_directory=chroma_dir)

    with pytest.raises(ValueError):
        manager.add_chunks(sample_chunks, [make_test_vector(1.0)])  # 2 chunks vs 1 vector
