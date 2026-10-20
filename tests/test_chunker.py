"""Unit tests for Phase 3A: Structural Document Chunking."""

from core.chunking.hierarchical_chunker import HierarchicalChunker
from core.chunking.models import Chunk
from core.ingestion.models import CleanedBlock, CleanedPage


def test_chunk_basic_structure_and_fields():
    """Verify that all required fields, metadata, and deterministic IDs are populated."""
    page = CleanedPage(
        page_number=1,
        text="1. Architecture Overview\n\nThe AUTOSAR classic platform architecture separates applications from hardware.",
        blocks=[
            CleanedBlock(
                block_index=0,
                text="1. Architecture Overview",
                is_heading=True,
                heading_level=1,
            ),
            CleanedBlock(
                block_index=1,
                text="The AUTOSAR classic platform architecture separates applications from hardware.",
                is_heading=False,
            ),
        ],
        headings=["1. Architecture Overview"],
        source_document="AUTOSAR_Specification.pdf",
    )

    chunker = HierarchicalChunker(max_chunk_chars=500, min_chunk_chars=50)
    chunks = chunker.chunk_document([page], document_id="AUTOSAR_Spec")

    assert len(chunks) == 1
    c = chunks[0]

    # Required attributes from specification
    assert isinstance(c, Chunk)
    assert c.chunk_id == "AUTOSAR_Spec_p1_c0"
    assert c.document_id == "AUTOSAR_Spec"
    assert c.page_number == 1
    assert c.section_title == "1. Architecture Overview"
    assert "1. Architecture Overview" in c.text
    assert "classic platform" in c.text
    assert c.start_char >= 0
    assert c.end_char > c.start_char
    assert not c.is_table
    assert c.token_count_estimate > 0

    # Required metadata
    meta = c.metadata
    assert meta.document_id == "AUTOSAR_Spec"
    assert meta.page_number == 1
    assert meta.section_title == "1. Architecture Overview"
    assert meta.heading_hierarchy == ["1. Architecture Overview"]

    flat = meta.to_flat_dict()
    assert flat["page_number"] == 1
    assert flat["is_table"] is False


def test_heading_boundaries_split_sections():
    """Verify that encountering a new heading triggers a chunk boundary."""
    page = CleanedPage(
        page_number=1,
        text="Section 1\nContent 1\nSection 2\nContent 2",
        blocks=[
            CleanedBlock(block_index=0, text="1. Software Components", is_heading=True, heading_level=1),
            CleanedBlock(
                block_index=1,
                text="Software Components (SWC) encapsulate functional automotive control algorithms.",
                is_heading=False,
            ),
            CleanedBlock(block_index=2, text="2. Port Specifications", is_heading=True, heading_level=1),
            CleanedBlock(
                block_index=3,
                text="Require Ports (RPort) and Provide Ports (PPort) establish typed communication endpoints.",
                is_heading=False,
            ),
        ],
        headings=["1. Software Components", "2. Port Specifications"],
    )

    chunker = HierarchicalChunker(max_chunk_chars=500, min_chunk_chars=20)
    chunks = chunker.chunk_document([page], document_id="doc_hld")

    assert len(chunks) == 2
    assert chunks[0].section_title == "1. Software Components"
    assert "SWC" in chunks[0].text
    assert chunks[1].section_title == "2. Port Specifications"
    assert "RPort" in chunks[1].text


def test_preserve_contiguous_table_block():
    """Verify that table blocks are preserved as cohesive, contiguous chunks tagged with is_table=True."""
    table_text = (
        "| Port Name | Direction | Interface | Description |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| R_WheelSpeed_FL | Require (RPort) | If_WheelSpeed | Front Left Wheel Speed sensor |\n"
        "| P_BrakeTorque_FL | Provide (PPort) | If_BrakeTorque | Actuator command request |"
    )

    page = CleanedPage(
        page_number=2,
        text=f"2.1 Port Table\n\nBelow is the port specification:\n\n{table_text}",
        blocks=[
            CleanedBlock(block_index=0, text="2.1 Port Table", is_heading=True, heading_level=2),
            CleanedBlock(block_index=1, text="Below is the port specification:", is_heading=False),
            CleanedBlock(block_index=2, text=table_text, is_heading=False, is_table_like=True),
        ],
        headings=["2.1 Port Table"],
    )

    chunker = HierarchicalChunker(max_chunk_chars=600, min_chunk_chars=30)
    chunks = chunker.chunk_document([page], document_id="doc_ports")

    # Table should be emitted as a distinct table chunk
    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 1
    tc = table_chunks[0]
    assert tc.is_table is True
    assert "R_WheelSpeed_FL" in tc.text
    assert "P_BrakeTorque_FL" in tc.text
    assert tc.section_title == "2.1 Port Table"


def test_oversized_section_fallback_overlap():
    """Verify that oversized narrative blocks are split with sentence-boundary overlap."""
    sentences = [
        f"Sentence number {i} provides detailed description of the electronic brake controller logic."
        for i in range(20)
    ]
    long_text = " ".join(sentences)

    page = CleanedPage(
        page_number=3,
        text=long_text,
        blocks=[CleanedBlock(block_index=0, text=long_text, is_heading=False)],
    )

    chunker = HierarchicalChunker(
        max_chunk_chars=400,
        min_chunk_chars=100,
        overlap_chars=120,
        respect_sentence_boundaries=True,
    )
    chunks = chunker.chunk_document([page], document_id="doc_long")

    assert len(chunks) > 1

    # Verify that all chunks adhere to max length
    for c in chunks:
        assert len(c.text) <= 400 + 50  # sentence boundary tolerance

    # Verify overlap exists between consecutive chunks
    c1_words = set(chunks[0].text.split())
    c2_words = set(chunks[1].text.split())
    shared_words = c1_words.intersection(c2_words)
    assert len(shared_words) > 3


def test_cross_page_section_continuity():
    """Verify section context is carried across page boundaries if no new heading is encountered."""
    page1 = CleanedPage(
        page_number=1,
        text="1. Runnables Specification\n\nRunnables are code snippets triggered by RTE events.",
        blocks=[
            CleanedBlock(block_index=0, text="1. Runnables Specification", is_heading=True),
            CleanedBlock(block_index=1, text="Runnables are code snippets triggered by RTE events."),
        ],
    )

    page2 = CleanedPage(
        page_number=2,
        text="Periodic runnables execute at 10ms intervals.",
        blocks=[
            CleanedBlock(block_index=0, text="Periodic runnables execute at 10ms intervals.", is_heading=False),
        ],
    )

    chunker = HierarchicalChunker(max_chunk_chars=500)
    chunks = chunker.chunk_document([page1, page2], document_id="doc_cont")

    p2_chunks = [c for c in chunks if c.page_number == 2]
    assert len(p2_chunks) == 1
    assert p2_chunks[0].section_title == "1. Runnables Specification"
