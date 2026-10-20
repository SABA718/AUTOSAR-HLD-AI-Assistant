"""Citation and grounding verification engine (Stage 10).

Parses citation tags from LLM responses, deterministically verifies them against
retrieved context chunks, flags ungrounded hallucinations, and formats traceable citations.
"""

import re
from typing import Optional, Sequence, Union
from core.chunking.models import Chunk
from core.llm.base import LLMResponse
from core.logger import get_logger
from core.rag.models import Citation, GroundedAnswer
from core.rag.prompt_builder import DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE
from core.retrieval.models import RetrievedChunk

logger = get_logger(__name__)

# Matches bracketed citation tokens: e.g. [AUTOSAR_Braking_p1_c0], [chunk_1, chunk_2]
CITATION_BRACKET_PATTERN = re.compile(r"\[([a-zA-Z0-9_\-,\s;]+)\]")


class CitationVerifier:
    """Deterministically verifies LLM response claims and citations against retrieved chunks."""

    def __init__(
        self,
        insufficient_evidence_phrase: str = DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE,
        snippet_max_chars: int = 200,
    ) -> None:
        """Initialize the citation verifier.

        Args:
            insufficient_evidence_phrase: Standardized negative-grounding phrase.
            snippet_max_chars: Maximum length for snippet excerpt in verified citations.
        """
        self.insufficient_evidence_phrase = insufficient_evidence_phrase
        self.snippet_max_chars = snippet_max_chars

    def extract_citation_ids(self, text: str) -> list[str]:
        """Extract all candidate chunk citation IDs from text in order of appearance."""
        if not text:
            return []

        raw_matches = CITATION_BRACKET_PATTERN.findall(text)
        candidate_ids: list[str] = []

        for match_str in raw_matches:
            # Handle comma or semicolon separated IDs inside a single bracket: e.g. [c1, c2]
            parts = re.split(r"[,;]", match_str)
            for part in parts:
                clean_id = part.strip()
                # Ensure the token has valid identifier characters and is not purely a common word
                if clean_id and not clean_id.lower().startswith("page ") and clean_id not in candidate_ids:
                    candidate_ids.append(clean_id)

        return candidate_ids

    def verify(
        self,
        response: Union[LLMResponse, str],
        retrieved_chunks: Sequence[Union[RetrievedChunk, Chunk]],
    ) -> GroundedAnswer:
        """Verify an LLM response against the retrieved context chunks.

        Args:
            response: LLMResponse model instance or raw answer string.
            retrieved_chunks: Sequence of authentic retrieved chunks provided to the LLM.

        Returns:
            GroundedAnswer containing verification status, verified and invalid citations,
            and any grounding warnings.
        """
        answer_text = response.content if isinstance(response, LLMResponse) else str(response)
        clean_text = answer_text.strip()

        # 1. Check for standardized insufficient-evidence response
        if self._is_insufficient_evidence(clean_text):
            logger.info("Answer correctly identified insufficient evidence.")
            return GroundedAnswer(
                answer_text=clean_text,
                verified_citations=[],
                invalid_citations=[],
                is_grounded=True,
                grounding_warnings=[],
                has_insufficient_evidence=True,
            )

        # 2. Build lookup table of retrieved chunks by chunk_id
        chunk_map: dict[str, Union[RetrievedChunk, Chunk]] = {
            c.chunk_id: c for c in retrieved_chunks
        }

        # 3. Extract cited IDs from the answer
        extracted_ids = self.extract_citation_ids(clean_text)

        verified_citations: list[Citation] = []
        invalid_citations: list[str] = []
        warnings: list[str] = []

        for cid in extracted_ids:
            if cid in chunk_map:
                chunk = chunk_map[cid]
                sec_label = f", Section: {chunk.section_title}" if chunk.section_title else ""
                source_label = f"[Doc: {chunk.document_id}, Page {chunk.page_number}{sec_label}]"

                raw_snippet = chunk.text.strip()
                snippet = (
                    raw_snippet[: self.snippet_max_chars] + "..."
                    if len(raw_snippet) > self.snippet_max_chars
                    else raw_snippet
                )

                verified_citations.append(
                    Citation(
                        chunk_id=cid,
                        document_id=chunk.document_id,
                        page_number=chunk.page_number,
                        section_title=chunk.section_title,
                        snippet=snippet,
                        is_table=getattr(chunk, "is_table", False),
                        source_label=source_label,
                    )
                )
            else:
                invalid_citations.append(cid)
                warnings.append(f"Fabricated or unretrieved citation ID: '{cid}'.")

        # 4. Evaluate grounding integrity
        is_grounded = True

        if invalid_citations:
            is_grounded = False
            logger.warning("Detected %d fabricated citations in answer.", len(invalid_citations))

        if not verified_citations and not invalid_citations:
            # Answer made claims without attaching any citation IDs
            is_grounded = False
            warnings.append("Answer contains no supporting citation IDs for its architectural claims.")
            logger.warning("Answer contains zero citation IDs.")

        return GroundedAnswer(
            answer_text=clean_text,
            verified_citations=verified_citations,
            invalid_citations=invalid_citations,
            is_grounded=is_grounded,
            grounding_warnings=warnings,
            has_insufficient_evidence=False,
        )

    def _is_insufficient_evidence(self, text: str) -> bool:
        """Check whether answer represents the standardized insufficient-evidence admission."""
        normalized_phrase = self.insufficient_evidence_phrase.strip().lower()
        normalized_text = text.strip().lower()

        return (
            normalized_phrase in normalized_text
            or normalized_text.startswith("insufficient evidence")
            or "insufficient evidence to answer" in normalized_text
        )


# Convenient alias
CitationEngine = CitationVerifier
