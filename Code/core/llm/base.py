"""Base protocol and response models for LLM provider abstraction (Stage 9)."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Structured response container returned by all LLM providers."""

    content: str
    provider_name: str
    model_name: str
    latency_seconds: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract interface defining the contract for pluggable LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'mock', 'gemini')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying model (e.g., 'gemini-2.5-flash', 'mock-hld-model')."""
        pass

    @abstractmethod
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response given a sequence of role-based chat messages.

        Args:
            messages: List of chat messages, typically produced by RAGPrompt.to_messages().
                      Format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: Sampling temperature (default 0.0 for deterministic engineering answers).
            max_output_tokens: Optional maximum generation length.
            **kwargs: Provider-specific generation parameters.

        Returns:
            Structured LLMResponse instance.
        """
        pass
