"""Unit tests for Phase 5: Citation and Grounding Verification."""

from core.llm.base import LLMResponse
from core.rag.citation_engine import CitationVerifier
from core.rag.models import Citation, GroundedAnswer
from core.rag.prompt_builder import DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE
from core.retrieval.models import RetrievedChunk


def make_test_chunk(
    chunk_id: str,
    doc_id: str = "AUTOSAR_Braking",
    page: int = 1,
    section: str = "1. Architecture Overview",
    text: str = "Software Component brake torque arbitration specification.",
    is_table: bool = False,
) -> RetrievedChunk:
    """Helper to construct realistic test chunks."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=doc_id,
        page_number=page,
        section_title=section,
        start_char=0,
        end_char=len(text),
        similarity_score=0.92,
        distance=0.08,
        is_table=is_table,
    )


def test_valid_single_citation():
    """Verify that a response with an authentic citation ID is marked grounded with resolved metadata."""
    chunk = make_test_chunk(
        chunk_id="AUTOSAR_Braking_p1_c0",
        doc_id="AUTOSAR_Braking",
        page=1,
        section="1. Architecture",
        text="SWC_BrakeTorqueArbitration calculates vehicle deceleration.",
    )

    llm_resp = LLMResponse(
        content="The SWC_BrakeTorqueArbitration manages braking deceleration [AUTOSAR_Braking_p1_c0].",
        provider_name="mock",
        model_name="mock-model",
    )

    verifier = CitationVerifier()
    answer = verifier.verify(llm_resp, retrieved_chunks=[chunk])

    assert isinstance(answer, GroundedAnswer)
    assert answer.is_grounded is True
    assert len(answer.verified_citations) == 1
    assert len(answer.invalid_citations) == 0
    assert len(answer.grounding_warnings) == 0

    cit = answer.verified_citations[0]
    assert isinstance(cit, Citation)
    assert cit.chunk_id == "AUTOSAR_Braking_p1_c0"
    assert cit.document_id == "AUTOSAR_Braking"
    assert cit.page_number == 1
    assert cit.section_title == "1. Architecture"
    assert "SWC_BrakeTorqueArbitration" in cit.snippet
    assert "[Doc: AUTOSAR_Braking, Page 1, Section: 1. Architecture]" == cit.source_label


def test_fabricated_citation_flagged_ungrounded():
    """Verify that an unknown or fabricated citation ID is caught and flagged as ungrounded."""
    chunk = make_test_chunk(chunk_id="real_chunk_p1_c0")

    llm_resp = LLMResponse(
        content="The system operates at 10ms cycle [hallucinated_chunk_999].",
        provider_name="mock",
        model_name="mock-model",
    )

    verifier = CitationVerifier()
    answer = verifier.verify(llm_resp, retrieved_chunks=[chunk])

    assert answer.is_grounded is False
    assert len(answer.verified_citations) == 0
    assert answer.invalid_citations == ["hallucinated_chunk_999"]
    assert len(answer.grounding_warnings) >= 1
    assert any("Fabricated or unretrieved citation" in w for w in answer.grounding_warnings)


def test_multiple_citations_mixed_valid_and_invalid():
    """Verify response containing multiple citations with partial validity."""
    c1 = make_test_chunk(chunk_id="c1", page=1, text="Component definitions.")
    c2 = make_test_chunk(chunk_id="c2", page=2, section="2. Ports", text="Port interfaces.", is_table=True)

    # Response cites c1 (valid), c2 (valid), and c_fake (invalid) in compound syntax
    response_text = (
        "Components define functional boundaries [c1]. "
        "Ports are typed by interfaces [c2, c_fake]."
    )

    verifier = CitationVerifier()
    answer = verifier.verify(response_text, retrieved_chunks=[c1, c2])

    assert answer.is_grounded is False  # Because c_fake is invalid
    assert len(answer.verified_citations) == 2
    assert [c.chunk_id for c in answer.verified_citations] == ["c1", "c2"]
    assert answer.invalid_citations == ["c_fake"]
    assert any("c_fake" in w for w in answer.grounding_warnings)


def test_multiple_all_valid_citations():
    """Verify response citing multiple authentic chunks is completely grounded."""
    c1 = make_test_chunk(chunk_id="c1", page=1)
    c2 = make_test_chunk(chunk_id="c2", page=2)

    response_text = "The design integrates brake torque [c1] and wheel speed sensors [c2]."

    verifier = CitationVerifier()
    answer = verifier.verify(response_text, retrieved_chunks=[c1, c2])

    assert answer.is_grounded is True
    assert len(answer.verified_citations) == 2
    assert len(answer.invalid_citations) == 0
    assert len(answer.grounding_warnings) == 0


def test_no_citations_detected():
    """Verify response with factual text but zero citation IDs is marked ungrounded with a warning."""
    c1 = make_test_chunk(chunk_id="c1")

    # Uncited response
    response_text = "The system uses 4 wheel speed sensors and provides anti-lock braking."

    verifier = CitationVerifier()
    answer = verifier.verify(response_text, retrieved_chunks=[c1])

    assert answer.is_grounded is False
    assert len(answer.verified_citations) == 0
    assert len(answer.invalid_citations) == 0
    assert any("no supporting citation IDs" in w for w in answer.grounding_warnings)


def test_insufficient_evidence_response_is_grounded():
    """Verify that admitting insufficient evidence is marked grounded without citation warnings."""
    c1 = make_test_chunk(chunk_id="c1")

    response_text = DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE

    verifier = CitationVerifier()
    answer = verifier.verify(response_text, retrieved_chunks=[c1])

    assert answer.is_grounded is True
    assert answer.has_insufficient_evidence is True
    assert len(answer.verified_citations) == 0
    assert len(answer.invalid_citations) == 0
    assert len(answer.grounding_warnings) == 0
    assert answer.answer_text == DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE
