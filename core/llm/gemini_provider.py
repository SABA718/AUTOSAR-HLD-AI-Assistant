"""Google Gemini LLM provider implementation (Stage 9).

Integrates the official Gemini API for grounded engineering analysis,
handling authentication, system instructions, and token usage tracking cleanly.
"""

import time
from typing import Any, Optional, Sequence
from config.settings import get_settings
from core.llm.base import BaseLLMProvider, LLMResponse
from core.logger import get_logger

logger = get_logger(__name__)


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM provider using project environment configuration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize Gemini provider.

        Args:
            api_key: Optional explicit API key override. Defaults to settings.gemini_api_key.
            model_name: Optional model identifier. Defaults to settings.gemini_model.
            timeout_seconds: Request timeout in seconds.
        """
        settings = get_settings()
        self._api_key = api_key or settings.gemini_api_key
        self._model_name = model_name or settings.gemini_model or "gemini-3-flash-preview"
        self.timeout_seconds = timeout_seconds

        # Masked key representation for secure logging
        if self._api_key and len(self._api_key) >= 8:
            self._masked_key = f"{self._api_key[:4]}...{self._api_key[-4:]}"
        elif self._api_key:
            self._masked_key = "***"
        else:
            self._masked_key = "None"

    @property
    def provider_name(self) -> str:
        return "gemini"

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
        """Generate response from Google Gemini model."""
        if not self._api_key or not self._api_key.strip():
            logger.error("Gemini API key is not configured.")
            raise ValueError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in your .env file or environment."
            )

        system_instruction = None
        user_content = ""

        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                user_content = content

        if not user_content and messages:
            user_content = messages[-1].get("content", "")

        logger.info(
            "Calling Gemini model '%s' (key: %s, prompt_len: %d)",
            self._model_name,
            self._masked_key,
            len(user_content),
        )

        start_time = time.time()

        try:
            # Import google generative AI client
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)

            generation_config: dict[str, Any] = {
                "temperature": temperature,
            }
            if max_output_tokens is not None:
                generation_config["max_output_tokens"] = max_output_tokens

            # Initialize generative model with optional system instruction
            model_kwargs: dict[str, Any] = {"model_name": self._model_name}
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction

            model = genai.GenerativeModel(**model_kwargs)

            # Execute generation
            response = model.generate_content(
                user_content,
                generation_config=generation_config,
                request_options={"timeout": self.timeout_seconds},
            )

            elapsed = time.time() - start_time

            response_text = ""
            if response and hasattr(response, "text"):
                response_text = response.text
            elif response and response.candidates:
                response_text = response.candidates[0].content.parts[0].text

            # Extract token usage metadata if available
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

            usage = getattr(response, "usage_metadata", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_token_count", None)
                completion_tokens = getattr(usage, "candidates_token_count", None)
                total_tokens = getattr(usage, "total_token_count", None)

            return LLMResponse(
                content=response_text,
                provider_name=self.provider_name,
                model_name=self.model_name,
                latency_seconds=round(elapsed, 4),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                metadata={
                    "finish_reason": str(getattr(response.candidates[0], "finish_reason", "STOP"))
                    if response.candidates
                    else "STOP"
                },
            )

        except ValueError:
            raise
        except Exception as exc:
            # Ensure exception messages do not leak secrets
            err_msg = str(exc)
            if self._api_key in err_msg:
                err_msg = err_msg.replace(self._api_key, self._masked_key)

            logger.error("Gemini API call failed: %s", err_msg)
            raise RuntimeError(f"Gemini API generation failed: {err_msg}") from exc
