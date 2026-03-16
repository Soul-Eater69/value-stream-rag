"""
LangGraph nodes for the LLM synthesis and recommendation generation stage.
"""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from src.graph.state import RecommendationState
from src.models.domain import RecommendationResult, ValueStream

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """You are an enterprise AI assistant. Read the following idea-card content
extracted from a PowerPoint presentation and write a concise 2-3 sentence summary
that captures the business problem, proposed solution, and key value delivered.

Idea-card content:
{content}

Summary:"""

_RECOMMENDATION_PROMPT = """You are an enterprise Value Stream expert. Given an idea-card summary and a
list of candidate Value Streams with their descriptions and scores, explain which
Value Streams are most relevant and why.

Idea-card summary:
{summary}

Top Value Stream candidates:
{candidates}

Write a clear, executive-level explanation (3-5 sentences) that:
1. Identifies the top 1-3 most relevant Value Streams.
2. Explains WHY each is relevant to the idea card.
3. Notes if any Value Streams are only marginally relevant.

Explanation:"""


class SynthesisNodes:
    """
    LLM-backed synthesis nodes that generate summaries and explanations.
    """

    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    def summarise_ppt(self, state: RecommendationState) -> RecommendationState:
        """
        Node: Generate a concise summary of the uploaded PPT using an LLM.

        This summary is used both for display and as the basis for the
        recommendation reasoning.
        """
        logger.info("Node: summarise_ppt | upload_id=%s", state.get("upload_id"))

        try:
            chunks = state.get("query_chunks", [])
            if not chunks:
                return {**state, "query_summary": "", "status": "summarised"}

            # Select most informative chunks (titles + text, skip notes/tables)
            content_parts = []
            for c in chunks:
                from src.models.domain import ChunkType
                if c.chunk_type in (ChunkType.SLIDE_TITLE, ChunkType.SLIDE_TEXT):
                    content_parts.append(c.content)
                if len("\n".join(content_parts)) > 3000:
                    break

            content = "\n\n".join(content_parts)[:3000]
            prompt = _SUMMARY_PROMPT.format(content=content)

            summary = self._call_llm(prompt)
            return {**state, "query_summary": summary, "status": "summarised"}

        except Exception as exc:
            logger.exception("summarise_ppt failed")
            errors = state.get("errors", [])
            errors.append(f"SummaryError: {exc}")
            # Non-fatal: continue with empty summary
            return {**state, "query_summary": "", "status": "summarised", "errors": errors}

    def synthesise_recommendation(
        self, state: RecommendationState
    ) -> RecommendationState:
        """
        Node: Generate the final recommendation with LLM-generated reasoning.

        Input  state keys: ranked_value_streams, query_summary, retrieval_context
        Output state keys: recommendation
        """
        logger.info(
            "Node: synthesise_recommendation | upload_id=%s", state.get("upload_id")
        )

        try:
            ranked = state.get("ranked_value_streams", [])
            summary = state.get("query_summary", "")
            ctx = state.get("retrieval_context")
            start = state.get("_start_time", datetime.utcnow().timestamp())

            if not ranked:
                reasoning = "No matching Value Streams were found for the uploaded idea card."
                confidence = 0.0
            else:
                # Format candidates for the LLM
                candidates_text = self._format_candidates(ranked[:5])
                prompt = _RECOMMENDATION_PROMPT.format(
                    summary=summary or "No summary available.",
                    candidates=candidates_text,
                )
                reasoning = self._call_llm(prompt)
                confidence = self._compute_confidence(ranked)

            recommendation = RecommendationResult(
                recommendation_id=uuid4(),
                upload_id=state["upload_id"],
                query_summary=summary,
                recommended_value_streams=ranked,
                evidence_by_value_stream=ctx.evidence if ctx else {},
                reasoning=reasoning,
                confidence=confidence,
                processing_duration_ms=int(
                    (datetime.utcnow().timestamp() - start) * 1000
                ),
            )

            return {
                **state,
                "recommendation": recommendation,
                "reasoning": reasoning,
                "status": "done",
            }

        except Exception as exc:
            logger.exception("synthesise_recommendation failed")
            errors = state.get("errors", [])
            errors.append(f"SynthesisError: {exc}")
            return {**state, "status": "error", "errors": errors}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the response text."""
        response = self._llm.chat.completions.create(
            model="gpt-4o",  # Will be overridden by deployment config
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _format_candidates(value_streams: list[ValueStream]) -> str:
        lines = []
        for i, vs in enumerate(value_streams, 1):
            score = f"{vs.final_score:.3f}" if vs.final_score is not None else "N/A"
            stage_names = ""
            if vs.stage_sequence:
                stage_names = " → ".join(vs.stage_sequence.stage_names[:5])
            lines.append(
                f"{i}. {vs.name} (score={score})\n"
                f"   Domain: {vs.domain or 'N/A'}\n"
                f"   Description: {vs.description[:200] if vs.description else 'N/A'}\n"
                f"   Key stages: {stage_names or 'N/A'}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _compute_confidence(ranked: list[ValueStream]) -> float:
        """Estimate confidence from the score distribution of top candidates."""
        if not ranked:
            return 0.0
        scores = [vs.final_score or 0.0 for vs in ranked[:3]]
        if not scores:
            return 0.0
        top = scores[0]
        # Confidence is high if the top score is well above the rest
        if len(scores) > 1:
            gap = top - scores[1]
            return round(min(1.0, top * (1 + gap)), 3)
        return round(min(1.0, top), 3)
