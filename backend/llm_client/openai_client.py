"""
OpenAI provider client.

Uses the official `openai` async SDK with a shared AsyncOpenAI instance.
Retry logic is handled by the SDK's built-in `max_retries` parameter plus
a tenacity wrapper for rate-limit back-off.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import structlog
from openai import AsyncOpenAI, APIConnectionError, RateLimitError as _OAIRateLimit
from openai import AuthenticationError as _OAIAuthError
from openai import BadRequestError, NotFoundError
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


class OpenAIClient(BaseLLMClient):
    """
    Async OpenAI client with connection-reuse and tenacity retry.

    Usage:
        client = OpenAIClient.create()          # synchronous factory
        response = await client.chat_complete(req)
        await client.close()
    """

    def __init__(self, sdk: AsyncOpenAI, default_model: str, embedding_model: str) -> None:
        self._sdk = sdk
        self._default_model = default_model
        self._embedding_model = embedding_model
        self._log = structlog.get_logger(self.__class__.__name__)

    @classmethod
    def create(cls) -> "OpenAIClient":
        cfg = get_settings().openai
        sdk = AsyncOpenAI(
            api_key=cfg.api_key.get_secret_value(),
            max_retries=0,      # we manage retries via tenacity
            timeout=cfg.timeout_seconds,
        )
        return cls(
            sdk=sdk,
            default_model=cfg.default_model,
            embedding_model=cfg.embedding_model,
        )

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        model = request.model or self._default_model
        cfg = get_settings().openai
        log = self._log.bind(model=model, temperature=request.temperature)

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(_RETRYABLE),
                stop=stop_after_attempt(cfg.max_retries),
                wait=wait_exponential_jitter(initial=1, max=30),
                reraise=True,
            ):
                with attempt:
                    log.debug("openai.chat_complete.attempt", attempt=attempt.retry_state.attempt_number)
                    raw = await self._sdk.chat.completions.create(
                        model=model,
                        messages=[m.model_dump() for m in request.messages],
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        stream=False,
                    )
        except _OAIRateLimit as exc:
            raise RateLimitError("OpenAI rate limit exceeded", {"model": model}) from exc
        except _OAIAuthError as exc:
            raise AuthenticationError("OpenAI authentication failed") from exc
        except NotFoundError as exc:
            raise ModelNotFoundError("OpenAI model not found", {"model": model}) from exc
        except BadRequestError as exc:
            if "context_length_exceeded" in str(exc) or "maximum context length" in str(exc):
                raise ContextLengthError("Prompt exceeds OpenAI context window", {"model": model}) from exc
            raise LLMClientError("OpenAI bad request", {"error": str(exc)}) from exc
        except APIConnectionError as exc:
            raise ProviderUnavailableError("OpenAI unreachable", {"error": str(exc)}) from exc

        choice = raw.choices[0]
        usage = raw.usage
        log.info(
            "openai.chat_complete.done",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
        return ChatCompletionResponse(
            content=choice.message.content or "",
            model=raw.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        model = request.model or self._default_model
        try:
            stream = await self._sdk.chat.completions.create(
                model=model,
                messages=[m.model_dump() for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except _OAIRateLimit as exc:
            raise RateLimitError("OpenAI rate limit (stream)", {"model": model}) from exc
        except APIConnectionError as exc:
            raise ProviderUnavailableError("OpenAI stream connection failed", {"error": str(exc)}) from exc

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model = request.model or self._embedding_model
        cfg = get_settings().openai
        log = self._log.bind(model=model, text_count=len(request.texts))

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(_RETRYABLE),
                stop=stop_after_attempt(cfg.max_retries),
                wait=wait_exponential_jitter(initial=1, max=30),
                reraise=True,
            ):
                with attempt:
                    raw = await self._sdk.embeddings.create(
                        model=model,
                        input=request.texts,
                        encoding_format="float",
                    )
        except _OAIRateLimit as exc:
            raise RateLimitError("OpenAI embedding rate limit", {"model": model}) from exc
        except APIConnectionError as exc:
            raise ProviderUnavailableError("OpenAI embedding unreachable", {"error": str(exc)}) from exc

        embeddings = [item.embedding for item in sorted(raw.data, key=lambda x: x.index)]
        log.debug("openai.embed.done", total_tokens=raw.usage.total_tokens)
        return EmbeddingResponse(
            embeddings=embeddings,
            model=raw.model,
            total_tokens=raw.usage.total_tokens,
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            models = await self._sdk.models.list()
            return len(list(models)) > 0
        except Exception:
            return False

    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="openai",
            default_model=self._default_model,
            embedding_model=self._embedding_model,
        )

    async def close(self) -> None:
        await self._sdk.close()
        self._log.info("openai_client.closed")
