"""
Unit tests for the MockEmbeddingService.

Real Azure/OpenAI tests live in tests/integration/.
"""

from __future__ import annotations

import pytest

from src.embeddings.service import MockEmbeddingService


class TestMockEmbeddingService:
    def test_embed_returns_correct_dimension(self):
        svc = MockEmbeddingService(dimension=128)
        vec = svc.embed("hello world")
        assert len(vec) == 128

    def test_embed_is_normalised(self):
        import math
        svc = MockEmbeddingService(dimension=256)
        vec = svc.embed("test text")
        magnitude = math.sqrt(sum(x**2 for x in vec))
        assert abs(magnitude - 1.0) < 1e-3

    def test_embed_batch_returns_correct_count(self):
        svc = MockEmbeddingService(dimension=64)
        texts = ["alpha", "beta", "gamma"]
        vectors = svc.embed_batch(texts)
        assert len(vectors) == 3

    def test_embed_deterministic(self):
        svc = MockEmbeddingService()
        v1 = svc.embed("same text")
        v2 = svc.embed("same text")
        assert v1 == v2

    def test_different_texts_different_vectors(self):
        svc = MockEmbeddingService()
        v1 = svc.embed("apple")
        v2 = svc.embed("orange")
        assert v1 != v2
