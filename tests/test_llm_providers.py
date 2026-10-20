"""Unit tests for LLM provider abstraction and implementations (Stage 9)."""

import os
import pytest

from core.chunking.models import Chunk, ChunkMetadata
from core.llm.base import BaseLLMProvider, LLMResponse
from core.llm.factory import get_llm_provider
from core.llm.gemini_provider import GeminiLLMProvider
from core.llm.mock_provider import MockLLMProvider
from core.rag.prompt_builder import (
    DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE,
    GroundedPromptBuilder,
)
from core.retrieval.models import RetrievedChunk


def make_sample_retrieved_chunk(
    chunk_id: str = "AUTOSAR_Braking_p1_c0",
    text: str = "Software Component provides brake torque arbitration logic.",
) -> RetrievedChunk:
    """Helper to generate a test RetrievedChunk."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        document_id="AUTOSAR_Braking",
        page_number=1,
        section_title="1. Architecture Overview",
        start_char=0,
        end_char=len(text),
        similarity_score=0.95,
        distance=0.05,
        is_table=False,
    )


def test_mock_llm_with_rag_prompt_messages():
    """Verify that MockLLMProvider accepts RAGPrompt.to_messages() and returns structured output."""
    builder = GroundedPromptBuilder()
    chunk = make_sample_retrieved_chunk()
    prompt = builder.build_prompt(
        question="What does the Software Component provide?",
        chunks=[chunk],
    )

    messages = prompt.to_messages()
    assert isinstance(messages, list)
    assert len(messages) == 2

    provider = MockLLMProvider()
    assert isinstance(provider, BaseLLMProvider)
    assert provider.provider_name == "mock"

    response = provider.generate(messages)

    assert isinstance(response, LLMResponse)
    assert response.provider_name == "mock"
    assert response.model_name == "mock-autosar-v1"
    assert response.latency_seconds >= 0.0
    assert response.prompt_tokens is not None and response.prompt_tokens > 0
    assert response.completion_tokens is not None and response.completion_tokens > 0
    assert response.total_tokens == response.prompt_tokens + response.completion_tokens

    # Verify deterministic cited response referencing chunk_id
    assert "AUTOSAR_Braking_p1_c0" in response.content


def test_mock_llm_insufficient_evidence_response():
    """Verify that MockLLMProvider returns the exact insufficient-evidence phrase when context is empty."""
    builder = GroundedPromptBuilder()
    prompt = builder.build_prompt(question="What is the hydraulic valve pressure?", chunks=[])
    messages = prompt.to_messages()

    provider = MockLLMProvider()
    response = provider.generate(messages)

    assert response.content == DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE


def test_mock_llm_canned_response():
    """Verify that custom canned responses can be injected for deterministic test flows."""
    custom_text = "Verified: SWC_Brake requires R_WheelSpeed [AUTOSAR_Braking_p1_c0]."
    provider = MockLLMProvider(canned_response=custom_text)

    response = provider.generate([{"role": "user", "content": "Query"}])
    assert response.content == custom_text


def test_provider_factory_selection(monkeypatch: pytest.MonkeyPatch):
    """Verify that switching providers occurs via configuration without code changes."""
    # 1. Default configured provider is mock
    provider_default = get_llm_provider()
    assert isinstance(provider_default, MockLLMProvider)
    assert provider_default.provider_name == "mock"

    # 2. Explicit provider selection
    provider_explicit = get_llm_provider("mock")
    assert isinstance(provider_explicit, MockLLMProvider)

    # 3. Switching via environment override
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "gemini")
    provider_gemini = get_llm_provider(api_key="test_dummy_key")
    assert isinstance(provider_gemini, GeminiLLMProvider)
    assert provider_gemini.provider_name == "gemini"

    # 4. Unsupported provider raises clean ValueError
    with pytest.raises(ValueError, match="Unsupported LLM provider 'unknown_model'"):
        get_llm_provider("unknown_model")


def test_gemini_missing_api_key_raises_clean_error():
    """Verify that Gemini provider raises a clean ValueError when API key is missing."""
    provider = GeminiLLMProvider(api_key="")

    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        provider.generate([{"role": "user", "content": "Test"}])


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set in environment. Skipping real provider smoke test.",
)
def test_real_gemini_provider_smoke_test():
    """Live smoke test executed ONLY when a valid GEMINI_API_KEY is configured in the environment."""
    api_key = os.environ["GEMINI_API_KEY"]
    provider = GeminiLLMProvider(api_key=api_key)

    messages = [
        {"role": "system", "content": "You are a concise engineering tester."},
        {"role": "user", "content": "Reply with 'PONG' and nothing else."},
    ]

    response = provider.generate(messages, max_output_tokens=10)

    assert isinstance(response, LLMResponse)
    assert response.provider_name == "gemini"
    assert "PONG" in response.content.upper()
    assert response.latency_seconds > 0.0
