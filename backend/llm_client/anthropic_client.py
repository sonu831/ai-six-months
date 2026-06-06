"""
Anthropic provider client (Claude family).

Anthropic's SDK does not support embeddings natively; the embed() method
raises NotImplementedError so callers know to route embedding requests to
OpenAI or Ollama instead.
"""

from __future__ import annotations

from typing import AsyncIterator

import structlog
from anthropic import AsyncAnthropic
from anthropic import APIConnectionError as _AntConnectionError
from anthropic import AuthenticationError as _AntAuthError
from anthropic import BadRequestError as _AntBadRequest
from anthropic import NotFoundError as _AntNotFound
from anthropic import RateLimitError as _AntRateLimit
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from backend.core.config import get_settings
from backend.core.exceptions import (
    AuthenticationError,
    ContextLengthError,
    LLMClientError,
    ModelNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
)
from backend.llm_client.base import (
    BaseLLMClient,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ProviderInfo,
)

logger = structlog.get_logger(__name__)

_RETRYABLE = (ProviderUnavailableError, RateLimitError)


class AnthropicClient(BaseLLMClient):
    """
    Async Anthropic (Claude) client.

    Anthropic's messages API separates system prompts from the turn history;
    this client transparently extracts the system message so callers can use
    the unified ChatMessage model.
    """

    def __init__(self, sdk: AsyncAnthropic, default_model: str) -> None:
        self._sdk = sdk
        self._default_model = default_model
        self._log = structlog.get_logger(self.__class__.__name__)

    @classmethod
    def create(cls) -> "AnthropicClient":
        cfg = get_settings().anthropic
        sdk = AsyncAnthropic(
            api_key=cfg.api_key.get_secret_value(),
            max_retries=0,
            timeout=cfg.timeout_seconds,
        )
        return cls(sdk=sdk, default_model=cfg.default_model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_messages(
        request: ChatCompletionRequest,
    ) -> tuple[str | None, list[dict[str, str]]]:
        """
        Anthropic requires system prompts to be passed separately.
        Extracts the first system message (if any) and returns the rest.
        """
        system: str | None = None
        turns: list[dict[str, str]] = []
        for msg in request.messages:
            if msg.role == "system":
                system = (system or "") + msg.content
            else:
                turns.append({"role": msg.role, "content": msg.content})
        return system, turns

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        model = request.model or self._default_model
        cfg = get_settings().anthropic
        system, turns = self._split_messages(request)
        log = self._log.bind(model=model)

        create_kwargs: dict = {
            "model": model,
            "messages": turns,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or 4096,
        }
        if system:
            create_kwargs["system"] = system

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(_RETRYABLE),
                stop=stop_after_attempt(cfg.max_retries),
                wait=wait_exponential_jitter(initial=1, max=30),
                reraise=True,
            ):
                with attempt:
                    log.debug("anthropic.chat_complete.attempt", attempt=attempt.retry_state.attempt_number)
                    raw = await self._sdk.messages.create(**create_kwargs)
        except _AntRateLimit as exc:
            raise RateLimitError("Anthropic rate limit exceeded", {"model": model}) from exc
        except _AntAuthError as exc:
            raise AuthenticationError("Anthropic authentication failed") from exc
        except _AntNotFound as exc:
            raise ModelNotFoundError("Anthropic model not found", {"model": model}) from exc
        except _AntBadRequest as exc:
            msg = str(exc)
            if "prompt is too long" in msg or "context_window" in msg.lower():
                raise ContextLengthError("Prompt exceeds Anthropic context window", {"model": model}) from exc
            raise LLMClientError("Anthropic bad request", {"error": msg}) from exc
        except _AntConnectionError as exc:
            raise ProviderUnavailableError("Anthropic unreachable", {"error": str(exc)}) from exc

        content = "".join(
            block.text for block in raw.content if hasattr(block, "text")
        )
        log.info(
            "anthropic.chat_complete.done",
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
        )
        return ChatCompletionResponse(
            content=content,
            model=raw.model,
            prompt_tokens=raw.usage.input_tokens,
            completion_tokens=raw.usage.output_tokens,
            total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        model = request.model or self._default_model
        system, turns = self._split_messages(request)

        create_kwargs: dict = {
            "model": model,
            "messages": turns,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or 4096,
        }
        if system:
            create_kwargs["system"] = system

        try:
            async with self._sdk.messages.stream(**create_kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except _AntRateLimit as exc:
            raise RateLimitError("Anthropic rate limit (stream)", {"model": model}) from exc
        except _AntConnectionError as exc:
            raise ProviderUnavailableError("Anthropic stream failed", {"error": str(exc)}) from exc

    # ------------------------------------------------------------------
    # Embeddings — not supported by Anthropic
    # ------------------------------------------------------------------

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. "
            "Route embedding requests to OpenAIClient or OllamaClient."
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            await self._sdk.messages.create(
                model=self._default_model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:
            return False

    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="anthropic",
            default_model=self._default_model,
            embedding_model=None,
            supports_streaming=True,
        )

    async def close(self) -> None:
        await self._sdk.close()
        self._log.info("anthropic_client.closed")
