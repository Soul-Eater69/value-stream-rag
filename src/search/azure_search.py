"""
Azure AI Search client for Value Stream retrieval.

Supports:
  - Pure vector search (approximate KNN)
  - Hybrid search (keyword + vector with RRF fusion)
  - Semantic re-ranking (Azure native semantic ranker)
  - Metadata filtering by domain, keywords, etc.
"""

from __future__ import annotations

import logging
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.models.domain import Stage, StageSequence, ValueStream

logger = logging.getLogger(__name__)


class AzureValueStreamSearcher:
    """
    Retrieves Value Streams from the Azure AI Search index.

    The index is assumed to already contain Value Stream embeddings and
    all relevant properties.  This class is read-only with respect to
    the Value Stream index.
    """

    # Field names in the Azure AI Search index (adjust to match your schema)
    FIELD_ID = "id"
    FIELD_NAME = "name"
    FIELD_DESCRIPTION = "description"
    FIELD_DOMAIN = "domain"
    FIELD_KEYWORDS = "keywords"
    FIELD_STAGES = "stages"
    FIELD_PROPERTIES = "properties"
    FIELD_EMBEDDING = "embedding"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str,
        embedding_deployment: str,
        api_version: str = "2024-05-01-preview",
        semantic_config_name: str = "default",
    ) -> None:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.models import VectorizedQuery

        self._VectorizedQuery = VectorizedQuery
        self._client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key),
        )
        self._semantic_config = semantic_config_name
        logger.info(
            "AzureValueStreamSearcher connected to index=%s endpoint=%s",
            index_name,
            endpoint,
        )

    # ------------------------------------------------------------------
    # Primary retrieval methods
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        domain_filter: str | None = None,
    ) -> list[ValueStream]:
        """Pure vector search against Value Stream embeddings."""
        vector_query = self._VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields=self.FIELD_EMBEDDING,
        )

        filter_expr = self._build_filter(domain=domain_filter)
        results = self._client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expr,
            top=top_k,
            select=self._all_fields(),
        )
        return [self._doc_to_value_stream(r) for r in results]

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 10,
        domain_filter: str | None = None,
        alpha: float = 0.5,
    ) -> list[ValueStream]:
        """
        Hybrid search: keyword BM25 + vector, fused via Reciprocal Rank Fusion.

        alpha: weight for vector component (0=keyword only, 1=vector only).
        Azure AI Search handles RRF automatically when both search_text and
        vector_queries are provided.
        """
        vector_query = self._VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k * 2,  # oversample, then RRF
            fields=self.FIELD_EMBEDDING,
            weight=alpha,
        )

        filter_expr = self._build_filter(domain=domain_filter)
        results = self._client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            filter=filter_expr,
            top=top_k,
            select=self._all_fields(),
            query_type="semantic" if self._semantic_config else "full",
            semantic_configuration_name=self._semantic_config or None,
        )
        return [self._doc_to_value_stream(r) for r in results]

    def semantic_rerank(
        self,
        query_text: str,
        candidates: list[ValueStream],
        top_n: int = 5,
    ) -> list[ValueStream]:
        """
        Apply Azure semantic ranker to re-rank an existing candidate list.

        Note: Azure semantic ranker works at search time. This method is a
        convenience wrapper that re-issues a search with semantic ranking
        restricted to the candidate IDs.
        """
        if not candidates:
            return candidates

        candidate_ids = [vs.id for vs in candidates]
        id_filter = " or ".join(f"id eq '{vid}'" for vid in candidate_ids)

        results = self._client.search(
            search_text=query_text,
            filter=id_filter,
            top=top_n,
            select=self._all_fields(),
            query_type="semantic",
            semantic_configuration_name=self._semantic_config,
        )
        return [self._doc_to_value_stream(r) for r in results]

    def get_by_id(self, value_stream_id: str) -> ValueStream | None:
        """Fetch a single Value Stream by its ID."""
        try:
            doc = self._client.get_document(key=value_stream_id)
            return self._doc_to_value_stream(doc)
        except Exception:
            logger.warning("Value Stream not found: %s", value_stream_id)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_fields(self) -> list[str]:
        return [
            self.FIELD_ID,
            self.FIELD_NAME,
            self.FIELD_DESCRIPTION,
            self.FIELD_DOMAIN,
            self.FIELD_KEYWORDS,
            self.FIELD_STAGES,
            self.FIELD_PROPERTIES,
        ]

    @staticmethod
    def _build_filter(domain: str | None) -> str | None:
        if domain:
            return f"domain eq '{domain}'"
        return None

    def _doc_to_value_stream(self, doc: dict[str, Any]) -> ValueStream:
        """Map an Azure AI Search document dict to a ValueStream domain object."""
        score = doc.get("@search.score", 0.0)
        reranker_score = doc.get("@search.reranker_score")

        # Parse stages if present
        stage_sequence = None
        raw_stages = doc.get(self.FIELD_STAGES) or []
        if raw_stages:
            stages = [
                Stage(
                    id=s.get("id", ""),
                    name=s.get("name", ""),
                    description=s.get("description", ""),
                    sequence_order=s.get("sequence_order", idx + 1),
                    keywords=s.get("keywords", []),
                )
                for idx, s in enumerate(raw_stages)
            ]
            stage_sequence = StageSequence(
                value_stream_id=doc.get(self.FIELD_ID, ""), stages=stages
            )

        return ValueStream(
            id=doc.get(self.FIELD_ID, ""),
            name=doc.get(self.FIELD_NAME, ""),
            description=doc.get(self.FIELD_DESCRIPTION, ""),
            domain=doc.get(self.FIELD_DOMAIN, ""),
            keywords=doc.get(self.FIELD_KEYWORDS) or [],
            properties=doc.get(self.FIELD_PROPERTIES) or {},
            stage_sequence=stage_sequence,
            similarity_score=float(score),
            rerank_score=float(reranker_score) if reranker_score is not None else None,
        )
