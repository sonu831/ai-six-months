"""
Abstract contract for all LLM provider clients.

Concrete implementations must honour:
    - async everywhere (no sync blocking calls)
    - retry logic internal to the client (callers never retry)
    - structured exceptions from backend.core.exceptions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EmbeddingRequest(BaseModel):
    texts: list[str]
    model: str | None = None


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    total_tokens: int


class ProviderInfo(BaseModel):
    name: str
    default_model: str
    embedding_model: str | None = None
    supports_streaming: bool = True
    max_context_tokens: int = Field(default=128_000)


class BaseLLMClient(ABC):
    """
    Interface every LLM provider client must satisfy.

    Lifecycle:
        client = await ConcreteClient.create()
        response = await client.chat_complete(request)
        await client.close()
    """

    @abstractmethod
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Single-turn completion. Returns the full response when done."""

    @abstractmethod
    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """
        Streaming completion. Yields text delta chunks as they arrive.
        Caller must async-for over the result.
        """

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed a batch of texts. Returned embeddings are L2-normalised."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and the API key is valid."""

    @abstractmethod
    def provider_info(self) -> ProviderInfo:
        """Static metadata about this provider."""

    @abstractmethod
    async def close(self) -> None:
        """Teardown: close HTTP sessions, release semaphores, flush caches."""
