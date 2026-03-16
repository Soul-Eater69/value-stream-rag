"""
LangGraph recommendation workflow.

Graph topology:
  START
    │
    ▼
  parse_ppt ──(error)──► error_handler ──► END
    │
    ▼
  chunk_ppt ──(error)──► error_handler ──► END
    │
    ▼
  embed_chunks ──(error)──► error_handler ──► END
    │
    ▼
  summarise_ppt (parallel with retrieve)
    │              │
    │         retrieve_candidates
    │              │
    └──────────────┘
             │
             ▼
         rank_candidates
             │
             ▼
       synthesise_recommendation
             │
             ▼
           persist_trace
             │
             ▼
            END
"""

from __future__ import annotations

import functools
import logging
from datetime import datetime

from langgraph.graph import END, START, StateGraph

from src.graph.nodes.parse_nodes import (
    chunk_uploaded_ppt,
    embed_query_chunks,
    parse_uploaded_ppt,
)
from src.graph.nodes.retrieval_nodes import rank_candidates, retrieve_candidates
from src.graph.nodes.synthesis_nodes import SynthesisNodes
from src.graph.state import RecommendationState

logger = logging.getLogger(__name__)


def build_recommendation_graph(
    embedding_service,
    retriever,
    ranker,
    synthesis_nodes: SynthesisNodes,
    db_session_factory=None,
) -> StateGraph:
    """
    Build and return the compiled LangGraph recommendation workflow.

    Dependencies are injected so the graph is testable without real services.
    """

    # ── Bind service dependencies to nodes (partial application)
    def _embed(state: RecommendationState) -> RecommendationState:
        return embed_query_chunks(state, embedding_service)

    def _retrieve(state: RecommendationState) -> RecommendationState:
        return retrieve_candidates(state, retriever)

    def _rank(state: RecommendationState) -> RecommendationState:
        return rank_candidates(state, ranker)

    def _summarise(state: RecommendationState) -> RecommendationState:
        return synthesis_nodes.summarise_ppt(state)

    def _synthesise(state: RecommendationState) -> RecommendationState:
        return synthesis_nodes.synthesise_recommendation(state)

    def _persist(state: RecommendationState) -> RecommendationState:
        if db_session_factory and state.get("recommendation"):
            _persist_trace(state, db_session_factory)
        return {**state, "status": "done"}

    def _error_handler(state: RecommendationState) -> RecommendationState:
        logger.error(
            "Recommendation workflow error | upload_id=%s | errors=%s",
            state.get("upload_id"),
            state.get("errors"),
        )
        return {**state, "status": "error"}

    # ── Routing logic
    def route_after_parse(state: RecommendationState) -> str:
        return "error" if state.get("status") == "error" else "chunk"

    def route_after_chunk(state: RecommendationState) -> str:
        return "error" if state.get("status") == "error" else "embed"

    def route_after_embed(state: RecommendationState) -> str:
        return "error" if state.get("status") == "error" else "summarise"

    def route_after_retrieve(state: RecommendationState) -> str:
        return "error" if state.get("status") == "error" else "rank"

    def route_after_rank(state: RecommendationState) -> str:
        return "error" if state.get("status") == "error" else "synthesise"

    # ── Build graph
    graph = StateGraph(RecommendationState)

    graph.add_node("parse", parse_uploaded_ppt)
    graph.add_node("chunk", chunk_uploaded_ppt)
    graph.add_node("embed", _embed)
    graph.add_node("summarise", _summarise)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("rank", _rank)
    graph.add_node("synthesise", _synthesise)
    graph.add_node("persist", _persist)
    graph.add_node("error", _error_handler)

    # Edges
    graph.add_edge(START, "parse")
    graph.add_conditional_edges("parse", route_after_parse, {"chunk": "chunk", "error": "error"})
    graph.add_conditional_edges("chunk", route_after_chunk, {"embed": "embed", "error": "error"})
    graph.add_conditional_edges("embed", route_after_embed, {"summarise": "summarise", "error": "error"})

    # Summarise and retrieve run sequentially (summarise first for LLM context)
    graph.add_edge("summarise", "retrieve")
    graph.add_conditional_edges(
        "retrieve", route_after_retrieve, {"rank": "rank", "error": "error"}
    )
    graph.add_conditional_edges(
        "rank", route_after_rank, {"synthesise": "synthesise", "error": "error"}
    )
    graph.add_edge("synthesise", "persist")
    graph.add_edge("persist", END)
    graph.add_edge("error", END)

    return graph.compile()


def _persist_trace(state: RecommendationState, session_factory) -> None:
    """Persist recommendation trace to SQLite for audit and evaluation."""
    from src.models.database import RecommendationTraceORM

    rec = state.get("recommendation")
    if rec is None:
        return

    session = session_factory()
    try:
        trace = RecommendationTraceORM(
            id=str(rec.recommendation_id),
            upload_id=state["upload_id"],
            original_filename=state.get("original_filename", ""),
            query_summary=rec.query_summary,
            recommended_vs_ids=[vs.id for vs in rec.recommended_value_streams],
            recommended_vs_names=[vs.name for vs in rec.recommended_value_streams],
            reasoning=rec.reasoning,
            confidence=rec.confidence,
            processing_duration_ms=rec.processing_duration_ms,
            full_result_json=rec.model_dump(mode="json"),
        )
        session.add(trace)
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Failed to persist recommendation trace")
    finally:
        session.close()
