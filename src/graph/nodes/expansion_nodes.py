"""
LangGraph node for structured query expansion.

Converts raw slide chunks into a structured expansion dict:
  {
    "summary":     str,            # 2-3 sentence business summary
    "keywords":    list[str],      # 5-15 domain/business keywords
    "domain_hint": str | None,     # primary business domain if detectable
  }

This dict is stored in state["query_expansion"] and used by downstream
retrieval and ranking nodes as a richer query signal.

Hardening:
  - JSON schema validation on LLM output (no free-form parsing).
  - 3-second timeout on the LLM call; guaranteed not to block the pipeline.
  - Deterministic non-LLM fallback: extract keywords from slide titles +
    term-frequency analysis of slide content.
  - Bypass rule: skip LLM entirely for short/simple decks
    (≤ BYPASS_SLIDE_THRESHOLD slides AND ≤ BYPASS_CONTENT_CHARS total chars).
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re

from src.graph.state import RecommendationState
from src.models.domain import ChunkType

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
BYPASS_SLIDE_THRESHOLD = 8       # slides; below this, skip LLM
BYPASS_CONTENT_CHARS = 1_500     # total chars; below this, skip LLM
EXPANSION_TIMEOUT_S = 3.0        # hard timeout on LLM call

# ── English stop words (minimal set sufficient for keyword extraction) ─────────
_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of with by is are was were be "
    "been being have has had do does did will would could should may might "
    "this that these those it its we our they their not from as if when "
    "so then also into about than more some any all".split()
)

# ── LLM expansion prompt ──────────────────────────────────────────────────────
_EXPANSION_PROMPT = """\
Analyse this idea-card content and return a JSON object.

Return ONLY valid JSON — no markdown fences, no explanation.

Required fields:
  "summary"    : string, 2-3 sentence business summary (max 400 chars)
  "keywords"   : array of 5-15 specific technical/business keywords (strings)
  "domain_hint": string or null — primary business domain if clearly identifiable

Content:
{content}"""


# ── Public node ───────────────────────────────────────────────────────────────


def expand_query(
    state: RecommendationState,
    llm_client=None,
) -> RecommendationState:
    """
    Node: Produce a structured query expansion for the uploaded PPT.

    Uses LLM when available and deck is above the bypass threshold.
    Falls back to deterministic extraction on timeout, parse error, or bypass.

    Input  state keys: query_chunks, _parsed_document
    Output state keys: query_expansion
    """
    logger.info("Node: expand_query | upload_id=%s", state.get("upload_id"))

    chunks = state.get("query_chunks", [])
    parsed = state.get("_parsed_document")
    slide_count = getattr(parsed, "slide_count", len(chunks))

    content, total_chars = _build_content(chunks)

    try:
        bypass = (
            llm_client is None
            or (slide_count <= BYPASS_SLIDE_THRESHOLD and total_chars <= BYPASS_CONTENT_CHARS)
        )

        if bypass:
            logger.debug(
                "expand_query: bypassing LLM (slide_count=%d, chars=%d)",
                slide_count,
                total_chars,
            )
            expansion = _deterministic_expansion(chunks, content)
        else:
            try:
                expansion = _llm_expansion(content, llm_client)
            except Exception as exc:
                logger.warning(
                    "expand_query: LLM expansion failed (%s); using deterministic fallback",
                    exc,
                )
                expansion = _deterministic_expansion(chunks, content)

    except Exception as exc:
        logger.exception("expand_query failed entirely: %s", exc)
        expansion = {"summary": "", "keywords": [], "domain_hint": None}

    return {**state, "query_expansion": expansion}


# ── LLM path ──────────────────────────────────────────────────────────────────


def _llm_expansion(content: str, llm_client) -> dict:
    """Call LLM with a hard timeout; raises on timeout or parse failure."""
    prompt = _EXPANSION_PROMPT.format(content=content[:3000])

    def _call() -> str:
        resp = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_call)
        try:
            raw = future.result(timeout=EXPANSION_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"LLM expansion timed out after {EXPANSION_TIMEOUT_S}s"
            )

    return _validate_expansion(raw)


def _validate_expansion(raw: str) -> dict:
    """Parse and validate LLM JSON output; raises ValueError on bad schema."""
    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.M).strip()

    data = json.loads(raw)  # raises json.JSONDecodeError if not valid JSON

    if not isinstance(data.get("summary"), str):
        raise ValueError("Missing or non-string 'summary' field")
    if not isinstance(data.get("keywords"), list):
        raise ValueError("Missing or non-list 'keywords' field")

    return {
        "summary": str(data["summary"])[:400],
        "keywords": [str(k)[:80] for k in data["keywords"][:15]],
        "domain_hint": str(data["domain_hint"])[:100] if data.get("domain_hint") else None,
    }


# ── Deterministic fallback ────────────────────────────────────────────────────


def _deterministic_expansion(chunks, content: str) -> dict:
    """
    Extract a query expansion without any LLM call.

    Summary: concatenated unique slide titles (up to 5).
    Keywords: top-15 terms by frequency, excluding stop words.
    domain_hint: None (cannot infer without LLM).
    """
    titles = [
        c.content.strip()
        for c in chunks
        if c.chunk_type == ChunkType.SLIDE_TITLE and c.content.strip()
    ]

    summary = "; ".join(dict.fromkeys(titles)[:5]) or content[:300]

    keywords = _extract_keywords(content, top_n=15)

    return {
        "summary": summary[:400],
        "keywords": keywords,
        "domain_hint": None,
    }


def _extract_keywords(text: str, top_n: int = 15) -> list[str]:
    """Term-frequency keyword extraction, ignoring stop words."""
    words = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", text)
    freq: dict[str, int] = {}
    for w in words:
        wl = w.lower()
        if wl not in _STOP_WORDS:
            freq[wl] = freq.get(wl, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:top_n]]


def _build_content(chunks) -> tuple[str, int]:
    """Concatenate title and text chunks up to ~3000 chars."""
    parts: list[str] = []
    total = 0
    for c in chunks:
        if c.chunk_type in (ChunkType.SLIDE_TITLE, ChunkType.SLIDE_TEXT):
            parts.append(c.content)
            total += len(c.content)
            if total >= 3000:
                break
    return "\n".join(parts), total
