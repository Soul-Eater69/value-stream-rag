"""
VSCatalogue: in-memory Value Stream catalogue loaded once at startup.

Replaces per-request Azure AI Search calls for VS candidate discovery.
With only ~50 Value Streams, scoring all of them via cosine similarity
is faster and more reliable than a top-K index search that may silently
omit VSes that only appear in historical (ChromaDB) evidence.

Startup behaviour:
  - Calls AzureValueStreamSearcher.list_all() to fetch all VS documents.
  - Embeds each VS description (or name if description is empty).
  - Caches both the ValueStream objects and their embeddings in memory.

Per-request behaviour:
  - score_all(query_embedding) computes cosine similarity against every
    cached VS embedding in O(N) time (~50 dot products, negligible cost).
  - Returns a score dict keyed by VS ID.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from src.models.domain import ValueStream

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VSCatalogue:
    """
    Holds all Value Streams in memory with their pre-computed embeddings.

    Must be initialised by calling load() before the retriever is used.
    In the ServiceContainer this is done at cached_property access time.
    """

    def __init__(self, azure_searcher, embedding_service) -> None:
        self._azure = azure_searcher
        self._embedder = embedding_service
        self._vs_map: dict[str, ValueStream] = {}
        self._vs_embeddings: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Fetch all Value Streams from Azure and pre-embed their descriptions.

        Call once at service startup.  Subsequent calls refresh the catalogue
        (useful for hot-reload without restarting).
        """
        all_vs = self._azure.list_all()
        if not all_vs:
            logger.warning("VSCatalogue: list_all() returned 0 Value Streams")
            return

        texts = [vs.description or vs.name for vs in all_vs]
        embeddings = self._embedder.embed_batch(texts)

        self._vs_map.clear()
        self._vs_embeddings.clear()
        for vs, emb in zip(all_vs, embeddings):
            self._vs_map[vs.id] = vs
            self._vs_embeddings[vs.id] = emb

        logger.info("VSCatalogue loaded %d value streams", len(self._vs_map))

    # ------------------------------------------------------------------
    # Query-time helpers
    # ------------------------------------------------------------------

    def all(self) -> dict[str, ValueStream]:
        """Return the full {vs_id: ValueStream} map."""
        return self._vs_map

    def score_all(self, query_embedding: list[float]) -> dict[str, float]:
        """
        Cosine similarity between query_embedding and every pre-embedded VS.

        Returns a dict {vs_id: score ∈ [0, 1]}.  Embeddings are assumed to
        come from the same model (text-embedding-ada-002 or equivalent) so
        scores are directly comparable.
        """
        scores: dict[str, float] = {}
        for vs_id, vs_emb in self._vs_embeddings.items():
            scores[vs_id] = self._cosine(query_embedding, vs_emb)
        return scores

    @property
    def size(self) -> int:
        return len(self._vs_map)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(0.0, dot / (norm_a * norm_b))
