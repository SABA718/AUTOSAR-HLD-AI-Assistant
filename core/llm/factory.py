"""Provider factory for instantiating LLM implementations (Stage 9)."""

import os
from typing import Any, Optional
from config.settings import get_settings
from core.llm.base import BaseLLMProvider
from core.llm.gemini_provider import GeminiLLMProvider
from core.llm.mock_provider import MockLLMProvider
from core.logger import get_logger

logger = get_logger(__name__)


def get_llm_provider(
    provider_name: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Instantiate and return an LLM provider based on configuration or explicit name.

    Args:
        provider_name: Optional provider identifier ('mock', 'gemini').
            If not specified, reads settings.default_llm_provider.
        **kwargs: Optional configuration parameters passed directly to the provider.

    Returns:
        Instance conforming to BaseLLMProvider.

    Raises:
        ValueError: If the requested provider is unknown or unsupported.
    """
    selected_name = provider_name or os.getenv("DEFAULT_LLM_PROVIDER") or get_settings().default_llm_provider
    selected = selected_name.strip().lower()

    logger.info("Instantiating LLM provider: '%s'", selected)

    if selected == "mock":
        return MockLLMProvider(**kwargs)

    if selected == "gemini":
        return GeminiLLMProvider(**kwargs)

    raise ValueError(
        f"Unsupported LLM provider '{selected}'. Supported providers are: 'mock', 'gemini'."
    )
