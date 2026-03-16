"""Embedding service abstraction."""

from .service import EmbeddingService, AzureOpenAIEmbeddingService, OpenAIEmbeddingService

__all__ = [
    "EmbeddingService",
    "AzureOpenAIEmbeddingService",
    "OpenAIEmbeddingService",
]
