"""Evaluation framework and metrics for AUTOSAR RAG pipeline (Phase 8 Stage 14).

Computes:
1. Retrieval Success: Whether expected chunks or evidence pages were retrieved.
2. Citation Validity: Ratio of cited chunks that actually exist in the retrieved context.
3. Groundedness Status: Whether the answer is strictly grounded without hallucinations.
4. Answer Relevance / Completeness: Deterministically measurable keyword overlap.
5. Negative Grounding Accuracy: Correctly identifying unanswerable questions.
"""

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union
from pydantic import BaseModel, Field

from core.llm.base import BaseLLMProvider
from core.logger import get_logger
from core.rag.citation_engine import CitationVerifier
from core.rag.models import GroundedAnswer
from core.rag.prompt_builder import DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE, GroundedPromptBuilder
from core.retrieval.retriever import SemanticRetriever

logger = get_logger(__name__)


class EvaluationItemResult(BaseModel):
    """Evaluation result for an individual golden test question."""

    question_id: str
    question: str
    is_answerable: bool

    # Retrieval metrics
    retrieval_success: bool
    retrieved_chunk_count: int
    retrieved_chunk_ids: list[str]

    # Citation & Grounding metrics
    is_grounded: bool
    has_insufficient_evidence: bool
    citation_validity_ratio: float
    verified_citation_count: int
    invalid_citation_count: int

    # Relevance & Negative grounding metrics
    answer_text: str
    expected_keywords_found: list[str]
    keyword_match_ratio: float
    negative_grounding_correct: bool


class EvaluationReport(BaseModel):
    """Aggregated evaluation summary report."""

    total_questions: int
    answerable_count: int
    unanswerable_count: int

    retrieval_success_rate: float
    overall_groundedness_rate: float
    citation_validity_rate: float
    negative_grounding_accuracy: float
    average_keyword_match_ratio: float

    item_results: list[EvaluationItemResult] = Field(default_factory=list)

    def to_summary_table(self) -> str:
        """Format a markdown summary table."""
        return (
            f"| Metric | Score |\n"
            f"|---|---|\n"
            f"| Total Test Cases | {self.total_questions} |\n"
            f"| Retrieval Success Rate | {self.retrieval_success_rate:.1%} |\n"
            f"| Groundedness Rate | {self.overall_groundedness_rate:.1%} |\n"
            f"| Citation Validity Rate | {self.citation_validity_rate:.1%} |\n"
            f"| Negative Grounding Accuracy | {self.negative_grounding_accuracy:.1%} |\n"
            f"| Average Keyword Match (Answerable) | {self.average_keyword_match_ratio:.1%} |"
        )


class PipelineEvaluator:
    """Evaluates the end-to-end RAG pipeline against a golden dataset."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        prompt_builder: Optional[GroundedPromptBuilder] = None,
        citation_verifier: Optional[CitationVerifier] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder or GroundedPromptBuilder()
        self.citation_verifier = citation_verifier or CitationVerifier()
        self.llm_provider = llm_provider

    def evaluate_item(
        self,
        item: dict[str, Any],
        document_id: Optional[str] = None,
        top_k: int = 5,
        relevance_threshold: Optional[float] = 0.3,
    ) -> EvaluationItemResult:
        """Run evaluation on a single golden dataset question item."""
        question_id = item["id"]
        question = item["question"]
        is_answerable = item["is_answerable"]
        expected_chunk_ids = item.get("expected_chunk_ids", [])
        expected_keywords = item.get("expected_keywords", [])
        doc_id = document_id or item.get("document_id")

        # 1. Retrieval
        retrieval_result = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            relevance_threshold=relevance_threshold if not is_answerable else None,
            document_id=doc_id,
        )
        retrieved_ids = [c.chunk_id for c in retrieval_result.chunks]

        if is_answerable:
            # Check if any expected chunk id or keyword-bearing chunk was retrieved
            retrieval_success = (
                any(eid in retrieved_ids for eid in expected_chunk_ids)
                if expected_chunk_ids
                else len(retrieval_result.chunks) > 0
            )
        else:
            # For unanswerable, low or empty retrieval is expected
            retrieval_success = True

        # 2. Prompt assembly & generation
        prompt = self.prompt_builder.build_prompt(
            question=question,
            chunks=retrieval_result.chunks,
        )

        if self.llm_provider is not None:
            llm_response = self.llm_provider.generate(prompt.to_messages())
            answer_text = llm_response.content
        else:
            # Deterministic fallback when no LLM is provided
            if not retrieval_result.chunks or "No relevant context" in prompt.user_prompt:
                answer_text = DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE
            else:
                first_id = retrieved_ids[0]
                answer_text = f"Based on the provided AUTOSAR specification, {question} [{first_id}]."

        # 3. Citation Verification
        grounded_answer: GroundedAnswer = self.citation_verifier.verify(
            response=answer_text,
            retrieved_chunks=retrieval_result.chunks,
        )

        total_extracted = len(grounded_answer.verified_citations) + len(grounded_answer.invalid_citations)
        citation_validity_ratio = (
            len(grounded_answer.verified_citations) / total_extracted
            if total_extracted > 0
            else (1.0 if grounded_answer.has_insufficient_evidence else 0.0)
        )

        # 4. Keyword relevance & Negative Grounding
        lower_ans = answer_text.lower()
        matched_keywords = [kw for kw in expected_keywords if kw.lower() in lower_ans]
        kw_ratio = len(matched_keywords) / len(expected_keywords) if expected_keywords else 1.0

        if not is_answerable:
            negative_grounding_correct = (
                grounded_answer.has_insufficient_evidence
                or DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE.lower() in lower_ans
            )
        else:
            negative_grounding_correct = True

        return EvaluationItemResult(
            question_id=question_id,
            question=question,
            is_answerable=is_answerable,
            retrieval_success=retrieval_success,
            retrieved_chunk_count=len(retrieved_ids),
            retrieved_chunk_ids=retrieved_ids,
            is_grounded=grounded_answer.is_grounded,
            has_insufficient_evidence=grounded_answer.has_insufficient_evidence,
            citation_validity_ratio=citation_validity_ratio,
            verified_citation_count=len(grounded_answer.verified_citations),
            invalid_citation_count=len(grounded_answer.invalid_citations),
            answer_text=answer_text,
            expected_keywords_found=matched_keywords,
            keyword_match_ratio=kw_ratio,
            negative_grounding_correct=negative_grounding_correct,
        )

    def evaluate_dataset(
        self,
        dataset_path: Union[str, Path],
        document_id: Optional[str] = None,
        top_k: int = 5,
        relevance_threshold: Optional[float] = 0.3,
    ) -> EvaluationReport:
        """Run complete evaluation over golden dataset file."""
        path = Path(dataset_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        resolved_doc_id = document_id or data.get("document_id")

        results: list[EvaluationItemResult] = []
        for q in questions:
            res = self.evaluate_item(
                q,
                document_id=resolved_doc_id,
                top_k=top_k,
                relevance_threshold=relevance_threshold,
            )
            results.append(res)

        total = len(results)
        if total == 0:
            return EvaluationReport(
                total_questions=0,
                answerable_count=0,
                unanswerable_count=0,
                retrieval_success_rate=0.0,
                overall_groundedness_rate=0.0,
                citation_validity_rate=0.0,
                negative_grounding_accuracy=0.0,
                average_keyword_match_ratio=0.0,
            )

        answerable = [r for r in results if r.is_answerable]
        unanswerable = [r for r in results if not r.is_answerable]

        retrieval_rate = sum(1 for r in results if r.retrieval_success) / total
        grounded_rate = sum(1 for r in results if r.is_grounded) / total
        citation_rate = sum(r.citation_validity_ratio for r in results) / total
        neg_rate = (
            sum(1 for r in unanswerable if r.negative_grounding_correct) / len(unanswerable)
            if unanswerable
            else 1.0
        )
        avg_kw = (
            sum(r.keyword_match_ratio for r in answerable) / len(answerable)
            if answerable
            else 1.0
        )

        return EvaluationReport(
            total_questions=total,
            answerable_count=len(answerable),
            unanswerable_count=len(unanswerable),
            retrieval_success_rate=retrieval_rate,
            overall_groundedness_rate=grounded_rate,
            citation_validity_rate=citation_rate,
            negative_grounding_accuracy=neg_rate,
            average_keyword_match_ratio=avg_kw,
            item_results=results,
        )
