"""Deterministic Mock LLM Provider for unit testing and offline development (Stage 9)."""

import re
import time
from typing import Any, Optional, Sequence
from core.llm.base import BaseLLMProvider, LLMResponse
from core.rag.prompt_builder import DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE

CHUNK_ID_PATTERN = re.compile(r'<chunk\s+id="([^"]+)"')


class MockLLMProvider(BaseLLMProvider):
    """Deterministic, zero-token mock provider requiring no network access or API credentials."""

    def __init__(
        self,
        model_name: str = "mock-autosar-v1",
        canned_response: Optional[str] = None,
        simulated_latency: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """Initialize mock LLM provider.

        Args:
            model_name: Identifier for the mock model.
            canned_response: Optional fixed string response to return unconditionally.
            simulated_latency: Duration in seconds to simulate generation latency.
        """
        self._model_name = model_name
        self.canned_response = canned_response
        self.simulated_latency = simulated_latency

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a deterministic response based on provided chat messages."""
        start_time = time.time()
        if self.simulated_latency > 0:
            time.sleep(self.simulated_latency)

        # 1. If canned response is provided, return immediately
        if self.canned_response is not None:
            content = self.canned_response
        else:
            # 2. Extract user and system message content
            user_text = ""
            for m in messages:
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                    break

            # Check if context contains no relevant evidence
            if "No relevant context available." in user_text:
                content = DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE
            else:
                # Discover chunk IDs in context
                chunk_ids = CHUNK_ID_PATTERN.findall(user_text)
                if chunk_ids:
                    first_id = chunk_ids[0]
                    content = (
                        f"Based on the provided AUTOSAR specification, the components and interfaces "
                        f"operate according to the architectural design [{first_id}]."
                    )
                else:
                    content = DEFAULT_INSUFFICIENT_EVIDENCE_PHRASE

        elapsed = time.time() - start_time
        prompt_len = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = max(1, prompt_len // 4)
        completion_tokens = max(1, len(content) // 4)

        return LLMResponse(
            content=content,
            provider_name=self.provider_name,
            model_name=self.model_name,
            latency_seconds=round(elapsed, 4),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            metadata={"simulated": True},
        )
