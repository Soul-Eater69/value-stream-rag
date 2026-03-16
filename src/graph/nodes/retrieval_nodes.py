"""
LangGraph nodes for the retrieval and ranking stage.
"""

from __future__ import annotations

import logging

from src.graph.state import RecommendationState

logger = logging.getLogger(__name__)


def retrieve_candidates(state: RecommendationState, retriever) -> RecommendationState:
    """
    Node: Run multi-source retrieval (Azure AI Search + ChromaDB).

    Input  state keys: query_chunks, domain_hint
    Output state keys: retrieval_context
    """
    logger.info("Node: retrieve_candidates | upload_id=%s", state.get("upload_id"))

    try:
        chunks = state.get("query_chunks", [])
        domain = state.get("domain_hint")

        ctx = retriever.retrieve(query_chunks=chunks, domain_filter=domain)

        logger.info(
            "Retrieved %d VS candidates | %d historical hits",
            len(ctx.vs_candidates),
            len(ctx.historical_hits),
        )
        return {**state, "retrieval_context": ctx, "status": "retrieved"}

    except Exception as exc:
        logger.exception("retrieve_candidates failed")
        errors = state.get("errors", [])
        errors.append(f"RetrievalError: {exc}")
        return {**state, "status": "error", "errors": errors}


def rank_candidates(state: RecommendationState, ranker) -> RecommendationState:
    """
    Node: Rank retrieved Value Stream candidates.

    Input  state keys: retrieval_context, query_chunks
    Output state keys: ranked_value_streams
    """
    logger.info("Node: rank_candidates | upload_id=%s", state.get("upload_id"))

    try:
        ctx = state.get("retrieval_context")
        if ctx is None:
            raise ValueError("No retrieval context; retrieve node must run first.")

        # Build query text for stage matching
        chunks = state.get("query_chunks", [])
        query_text = " ".join(c.content[:200] for c in chunks[:10])

        ranked = ranker.rank(ctx=ctx, query_text=query_text)

        return {**state, "ranked_value_streams": ranked, "status": "ranked"}

    except Exception as exc:
        logger.exception("rank_candidates failed")
        errors = state.get("errors", [])
        errors.append(f"RankingError: {exc}")
        return {**state, "status": "error", "errors": errors}
