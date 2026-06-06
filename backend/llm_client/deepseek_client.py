"""
DeepSeek provider client.

DeepSeek exposes an OpenAI-compatible Chat Completions API, so this client
wraps the official `openai` SDK pointed at the DeepSeek base URL.
Embeddings are not currently offered by DeepSeek; embed() raises NotImplementedError.

Env vars:
    DEEPSEEK_API_KEY       required
    DEEPSEEK_DEFAULT_MODEL optional (default: deepseek-chat)
    DEEPSEEK_BASE_URL      optional (default: https://api.deepseek.com/v1)
"""

from __future__ import annotations

from typing import AsyncIterator

import structlog
from openai import AsyncOpenAI, APIConnectionError, RateLimitError as _RateLimit
from openai import AuthenticationError as _AuthError, NotFoundError
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from backend.core.exceptions import (
    AuthenticationError,
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


class DeepSeekSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEEPSEEK_", extra="ignore")

    api_key: SecretStr = SecretStr("")
    default_model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/v1"
    max_retries: int = 3
    timeout_seconds: float = 60.0


class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek client using the OpenAI SDK with a custom base_url.

    Usage:
        client = DeepSeekClient.create()
        response = await client.chat_complete(request)
        await client.close()
    """

    def __init__(self, sdk: AsyncOpenAI, default_model: str) -> None:
        self._sdk = sdk
        self._default_model = default_model
        self._log = structlog.get_logger(self.__class__.__name__)

    @classmethod
    def create(cls) -> "DeepSeekClient":
        cfg = DeepSeekSettings()
        sdk = AsyncOpenAI(
            api_key=cfg.api_key.get_secret_value(),
            base_url=cfg.base_url,
            max_retries=0,
            timeout=cfg.timeout_seconds,
        )
        return cls(sdk=sdk, default_model=cfg.default_model)

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        model = request.model or self._default_model
        cfg = DeepSeekSettings()
        log = self._log.bind(model=model)
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(_RETRYABLE),
                stop=stop_after_attempt(cfg.max_retries),
                wait=wait_exponential_jitter(initial=1, max=30),
                reraise=True,
            ):
                with attempt:
                    log.debug("deepseek.chat_complete.attempt", n=attempt.retry_state.attempt_number)
                    raw = await self._sdk.chat.completions.create(
                        model=model,
                        messages=[m.model_dump() for m in request.messages],
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        stream=False,
                    )
        except _RateLimit as exc:
            raise RateLimitError("DeepSeek rate limit", {"model": model}) from exc
        except _AuthError as exc:
            raise AuthenticationError("DeepSeek authentication failed") from exc
        except NotFoundError as exc:
            raise ModelNotFoundError("DeepSeek model not found", {"model": model}) from exc
        except APIConnectionError as exc:
            raise ProviderUnavailableError("DeepSeek unreachable", {"error": str(exc)}) from exc

        choice = raw.choices[0]
        usage = raw.usage
        log.info("deepseek.chat_complete.done", total_tokens=usage.total_tokens if usage else 0)
        return ChatCompletionResponse(
            content=choice.message.content or "",
            model=raw.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

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
        except APIConnectionError as exc:
            raise ProviderUnavailableError("DeepSeek stream failed", {"error": str(exc)}) from exc

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise NotImplementedError(
            "DeepSeek does not provide an embeddings API. "
            "Use OpenAIClient or OllamaClient for embeddings."
        )

    async def health_check(self) -> bool:
        try:
            models = await self._sdk.models.list()
            return len(list(models)) > 0
        except Exception:
            return False

    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="deepseek",
            default_model=self._default_model,
            embedding_model=None,
            supports_streaming=True,
        )

    async def close(self) -> None:
        await self._sdk.close()
        self._log.info("deepseek_client.closed")
