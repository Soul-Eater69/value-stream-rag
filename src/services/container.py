"""
Dependency injection container.

Assembles all services from configuration at startup.
This is the single place where concrete implementations are wired together.
"""

from __future__ import annotations

import logging
from functools import cached_property

from src.config.settings import Settings, get_settings
from src.embeddings.service import build_embedding_service
from src.models.database import get_session_factory
from src.ranking.ranker import RankingConfig, ValueStreamRanker
from src.search.azure_search import AzureValueStreamSearcher
from src.search.chroma_store import ChromaVectorStore
from src.search.retriever import HybridRetriever
from src.search.vs_catalogue import VSCatalogue

logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Lazy-initialised service container.

    Services are created once on first access and cached for the lifetime
    of the application.  In tests, this container can be replaced with a
    mock container using dependency overrides.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @cached_property
    def embedding_service(self):
        return build_embedding_service(self._settings)

    @cached_property
    def azure_searcher(self) -> AzureValueStreamSearcher:
        cfg = self._settings.azure_search
        ao = self._settings.azure_openai
        return AzureValueStreamSearcher(
            endpoint=cfg.endpoint,
            api_key=cfg.api_key,
            index_name=cfg.value_stream_index,
            embedding_deployment=ao.embedding_deployment,
            api_version=cfg.api_version,
            semantic_config_name=cfg.semantic_config_name,
        )

    @cached_property
    def vs_catalogue(self) -> VSCatalogue:
        """
        Pre-loaded Value Stream catalogue.

        Calls list_all() on first access, embedding all VS descriptions once.
        All subsequent per-request retrievals use the in-memory cosine scores.
        """
        catalogue = VSCatalogue(
            azure_searcher=self.azure_searcher,
            embedding_service=self.embedding_service,
        )
        catalogue.load()
        return catalogue

    @cached_property
    def chroma_store(self) -> ChromaVectorStore:
        cfg = self._settings.chroma
        cfg.persist_dir.mkdir(parents=True, exist_ok=True)
        return ChromaVectorStore(
            persist_dir=cfg.persist_dir,
            collection_name=cfg.historical_collection,
        )

    @cached_property
    def retriever(self) -> HybridRetriever:
        r = self._settings.retrieval
        return HybridRetriever(
            vs_catalogue=self.vs_catalogue,
            chroma_store=self.chroma_store,
            embedding_service=self.embedding_service,
            top_k_hist=r.top_k_historical,
            similarity_threshold=r.similarity_threshold,
        )

    @cached_property
    def ranker(self) -> ValueStreamRanker:
        return ValueStreamRanker(RankingConfig(top_n=self._settings.retrieval.rerank_top_n))

    @cached_property
    def db_session_factory(self):
        return get_session_factory(self._settings.database_url)

    @cached_property
    def llm_client(self):
        ao = self._settings.azure_openai
        if ao.use_azure:
            from openai import AzureOpenAI
            return AzureOpenAI(
                azure_endpoint=ao.endpoint,
                api_key=ao.api_key,
                api_version=ao.api_version,
            )
        else:
            from openai import OpenAI
            return OpenAI(api_key=ao.openai_api_key)

    @cached_property
    def ingestion_pipeline(self):
        from src.chunking.chunker import ChunkingConfig
        from src.ingestion.pipeline import IngestionPipeline
        cfg = self._settings.chunking
        return IngestionPipeline(
            embedding_service=self.embedding_service,
            vector_store=self.chroma_store,
            db_session_factory=self.db_session_factory,
            chunking_config=ChunkingConfig(
                max_tokens=cfg.max_tokens,
                overlap_tokens=cfg.overlap_tokens,
            ),
        )

    @cached_property
    def recommendation_graph(self):
        from src.graph.nodes.synthesis_nodes import SynthesisNodes
        from src.graph.workflows.recommendation_graph import build_recommendation_graph
        synthesis = SynthesisNodes(llm_client=self.llm_client)
        return build_recommendation_graph(
            embedding_service=self.embedding_service,
            retriever=self.retriever,
            ranker=self.ranker,
            synthesis_nodes=synthesis,
            db_session_factory=self.db_session_factory,
        )

    @cached_property
    def ingestion_graph(self):
        from src.graph.workflows.ingestion_graph import build_ingestion_graph
        return build_ingestion_graph(self.ingestion_pipeline)
