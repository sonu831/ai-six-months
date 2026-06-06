"""
Ollama local LLM client — zero-cost fallback for development and offline work.

Talks to the Ollama REST API via httpx.AsyncClient.
Supports both chat completions and embeddings (nomic-embed-text, mxbai-embed, etc.).
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
import structlog

from backend.core.config import get_settings
from backend.core.exceptions import (
    LLMClientError,
    ModelNotFoundError,
    ProviderUnavailableError,
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


class OllamaClient(BaseLLMClient):
    """
    Async Ollama REST client.

    Ollama API reference: https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    def __init__(self, http: httpx.AsyncClient, base_url: str, default_model: str, embedding_model: str) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._embedding_model = embedding_model
        self._log = structlog.get_logger(self.__class__.__name__)

    @classmethod
    def create(cls) -> "OllamaClient":
        cfg = get_settings().ollama
        http = httpx.AsyncClient(
            base_url=cfg.base_url,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        return cls(
            http=http,
            base_url=cfg.base_url,
            default_model=cfg.default_model,
            embedding_model=cfg.embedding_model,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._http.post(path, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                "Ollama server is not reachable. Is `ollama serve` running?",
                {"base_url": self._base_url, "error": str(exc)},
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ModelNotFoundError(
                    "Ollama model not found — run `ollama pull <model>`",
                    {"status": 404, "path": path},
                ) from exc
            raise LLMClientError(
                "Ollama API error",
                {"status": exc.response.status_code, "body": exc.response.text[:200]},
            ) from exc

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        model = request.model or self._default_model
        log = self._log.bind(model=model)
        log.debug("ollama.chat_complete.start")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens

        data = await self._post("/api/chat", payload)

        content = data.get("message", {}).get("content", "")
        prompt_eval = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)
        log.info("ollama.chat_complete.done", prompt_tokens=prompt_eval, completion_tokens=eval_count)
        return ChatCompletionResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_eval,
            completion_tokens=eval_count,
            total_tokens=prompt_eval + eval_count,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        model = request.model or self._default_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": True,
            "options": {"temperature": request.temperature},
        }
        try:
            async with self._http.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError("Ollama stream connection failed", {"error": str(exc)}) from exc

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model = request.model or self._embedding_model
        log = self._log.bind(model=model, text_count=len(request.texts))
        log.debug("ollama.embed.start")

        embeddings: list[list[float]] = []
        total_tokens = 0
        for text in request.texts:
            data = await self._post("/api/embeddings", {"model": model, "prompt": text})
            embeddings.append(data["embedding"])
            total_tokens += len(text.split())  # Ollama doesn't return token counts

        log.debug("ollama.embed.done")
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            total_tokens=total_tokens,
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            response = await self._http.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="ollama",
            default_model=self._default_model,
            embedding_model=self._embedding_model,
            supports_streaming=True,
            max_context_tokens=8_192,
        )

    async def close(self) -> None:
        await self._http.aclose()
        self._log.info("ollama_client.closed")
