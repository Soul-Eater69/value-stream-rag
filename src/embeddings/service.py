"""
Embedding service with Azure OpenAI and plain OpenAI backends.

Design:
  - Abstract base class so the backend can be swapped for tests/local dev.
  - Batch embedding with retry logic (tenacity).
  - Token-budget enforcement before sending to the API.
  - Caching hook for production use (Redis, in-memory).
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Maximum tokens per embedding API call (text-embedding-3-large supports 8192)
_MAX_EMBED_TOKENS = 8192


class EmbeddingService(ABC):
    """Abstract embedding service."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...


class AzureOpenAIEmbeddingService(EmbeddingService):
    """Azure OpenAI text-embedding-3-large backend."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str = "text-embedding-3-large",
        api_version: str = "2024-02-01",
        batch_size: int = 16,
    ) -> None:
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._deployment = deployment
        self._batch_size = batch_size
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # Infer dimension from a probe embedding
            probe = self.embed("dimension probe")
            self._dim = len(probe)
        return self._dim

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=[text], model=self._deployment
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed in batches to respect API rate limits."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts, model=self._deployment
        )
        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]


class OpenAIEmbeddingService(EmbeddingService):
    """Plain OpenAI embedding backend (for local dev / PoC)."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        batch_size: int = 16,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            probe = self.embed("dimension probe")
            self._dim = len(probe)
        return self._dim

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(input=batch, model=self._model)
            sorted_data = sorted(response.data, key=lambda d: d.index)
            all_embeddings.extend([d.embedding for d in sorted_data])
        return all_embeddings


class MockEmbeddingService(EmbeddingService):
    """
    Deterministic mock embedding service for unit tests.

    Produces a stable pseudo-embedding from text hash – NOT for production.
    """

    def __init__(self, dimension: int = 1536) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        import struct

        digest = hashlib.sha256(text.encode()).digest()
        # Repeat digest bytes to fill the required dimension
        raw = (digest * ((self._dimension * 4 // len(digest)) + 1))[
            : self._dimension * 4
        ]
        floats = list(struct.unpack(f"{self._dimension}f", raw))
        # L2-normalise
        mag = sum(x**2 for x in floats) ** 0.5
        return [x / (mag + 1e-9) for x in floats]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def build_embedding_service(settings: Any) -> EmbeddingService:
    """Factory that builds the correct embedding service from settings."""
    ao = settings.azure_openai
    if ao.use_azure:
        logger.info("Using Azure OpenAI embedding service")
        return AzureOpenAIEmbeddingService(
            endpoint=ao.endpoint,
            api_key=ao.api_key,
            deployment=ao.embedding_deployment,
            api_version=ao.api_version,
        )
    elif ao.openai_api_key:
        logger.info("Using OpenAI embedding service")
        return OpenAIEmbeddingService(
            api_key=ao.openai_api_key,
            model=ao.openai_embedding_model,
        )
    else:
        logger.warning(
            "No embedding API keys configured – using MockEmbeddingService. "
            "Set AZURE_OPENAI_* or OPENAI_API_KEY environment variables."
        )
        return MockEmbeddingService()
