# Retrieval Redesign — VSCatalogue & Silent-Drop Fix

> **Status:** Implemented
> **Affected files:** `vs_catalogue.py` (new), `azure_search.py`, `retriever.py`, `ranker.py`, `container.py`

---

## Problem Statement

The original retrieval design had two bugs that partially cancelled each other:

### Bug 1 — Azure search was the only VS discovery mechanism

`HybridRetriever` called `azure_searcher.hybrid_search(top_k=10)` on every request.
Any VS outside the top-10 Azure result window was never even considered, regardless of how strong the historical evidence for it was.

### Bug 2 — Ranker silently dropped historically-matched VSes

`ranker.py:73-76` (old) iterated all VS IDs from both sources but only scored those already in `vs_map`, which was built exclusively from Azure candidates:

```python
vs = vs_map.get(vs_id)
if vs is None:
    continue   # VS only found via ChromaDB — silently dropped
```

**Result:** The historical signal (`mapped_vs_ids` in ChromaDB metadata) existed but was structurally prevented from ever influencing results.  Both document types — historical PPTs and idea cards — store `mapped_vs_ids` in their ChromaDB chunk metadata, making Azure the redundant bottleneck.

---

## Key Insight

> Both document types ingested into ChromaDB (historical PPTs **and** idea cards) carry `mapped_vs_ids` in chunk metadata, written at ingest time.
> ChromaDB alone is a complete source of VS-ID signals.
> Azure is only needed as a **metadata store** (VS names, descriptions, stage sequences), not as a search engine.
> With only ~50 VSes, scoring all of them in memory is trivially fast.

---

## New End-to-End Flow

### Startup (once per service lifetime)

```
ServiceContainer.vs_catalogue (cached_property)
        │
        ▼
VSCatalogue.load()
        │
        ├─ AzureValueStreamSearcher.list_all()
        │   → fetch ALL ~50 VS documents from Azure index (search_text="*")
        │   → returns list[ValueStream] with full metadata
        │
        └─ EmbeddingService.embed_batch(vs.description for each vs)
            → store {vs_id: embedding} in memory
            → store {vs_id: ValueStream}  in memory
```

Azure is touched **once** at startup.  Every subsequent request is Azure-free.

---

### Per Request

```
Upload PPT
    │
    ▼
Parse + Chunk  →  list[Chunk]
    │
    ▼
HybridRetriever.retrieve(query_chunks)
    │
    ├─ 1. Embed all chunks  →  chunk_embeddings[]
    │
    ├─ 2. Mean-pool chunk_embeddings  →  agg_embedding
    │
    ├─ 3. VSCatalogue.score_all(agg_embedding)
    │       → cosine(agg_embedding, vs_emb) for ALL ~50 VSes
    │       → {vs_id: score}  (O(50) dot products, ~microseconds)
    │       → ctx.vs_candidates = all 50 VSes, each with similarity_score
    │
    └─ 4. Per-chunk ChromaDB query
            → top-K historical chunk hits
            → filter by similarity_threshold
            → for each hit: extract hit["mapped_vs_ids"]
            → ctx.historically_matched_vs_ids (set of VS IDs)
            → ctx.evidence[vs_id] += RetrievalEvidence (HISTORICAL_PPT_INDEX)
    │
    ▼
ValueStreamRanker.rank(ctx, query_text)
    │
    ├─ vs_map = {vs.id: vs  for vs in ctx.vs_candidates}
    │          (now contains ALL 50 VSes — nothing can be missing)
    │
    ├─ all_vs_ids = vs_map.keys() ∪ historically_matched_vs_ids
    │
    ├─ hist_boost[vs_id] = mean(hit.similarity) × log_count_factor
    │                       for each VS referenced in historical hits
    │
    ├─ stage_scores[vs_id] = keyword_overlap(vs.stage_keywords, query_text)
    │                         (now operates on full vs_map, not just Azure hits)
    │
    └─ final_score = 0.40 × cosine_score
                   + 0.35 × historical_boost
                   + 0.15 × rerank_score (= cosine if no reranker)
                   + 0.10 × stage_match
    │
    ▼
top-5 ValueStreams  →  SynthesisNode  →  LLM explanation  →  RecommendationResult
```

---

## Component Changes

### New: `src/search/vs_catalogue.py`

| Method | Purpose |
|--------|---------|
| `load()` | Calls `azure_searcher.list_all()` + `embed_batch()`. Called once at startup. |
| `score_all(query_embedding)` | Returns `{vs_id: cosine_score}` for all VSes in memory. |
| `all()` | Returns `{vs_id: ValueStream}` map for the ranker. |

### Modified: `src/search/azure_search.py`

Added `list_all(max_results=1000)` — issues a `search_text="*"` query to retrieve all VS documents.  No per-request impact.

### Modified: `src/search/retriever.py`

| Before | After |
|--------|-------|
| Constructor takes `azure_searcher` | Constructor takes `vs_catalogue` |
| Step 1: `azure_searcher.hybrid_search(top_k=10)` — returns at most 10 VSes | Step 1: `catalogue.score_all(agg_embedding)` — scores all 50 |
| `top_k_vs` parameter | Removed (not needed; all VSes are scored) |
| `hybrid_alpha` parameter | Removed (not needed) |

### Modified: `src/ranking/ranker.py`

| Before | After |
|--------|-------|
| `vs_map` built from Azure top-K only | `vs_map` built from `ctx.vs_candidates` (all 50) |
| `if vs is None: continue` — silent drop | Same guard kept as **safety warning log**, but should never fire |
| `_compute_stage_scores` iterated `ctx.vs_candidates` (Azure only) | Iterates full `vs_map` parameter (all 50) |

### Modified: `src/services/container.py`

Added `vs_catalogue` cached property.  `VSCatalogue.load()` is called on first access (startup).  `retriever` now receives `vs_catalogue` instead of `azure_searcher`.

---

## Design Decisions

### D1 — Load all VSes at startup, not per-request

**Decision:** Pre-load all ~50 VSes once and cache in memory.
**Rationale:** 50 documents is trivially small.  Scoring all of them via dot product is O(50 × embedding_dim) ≈ microseconds, far cheaper than an Azure network round-trip.  Startup latency is acceptable (one-time cost).
**Trade-off:** If the VS catalogue changes, the service must restart (or call `catalogue.load()` again).  For a PoC with a stable catalogue, this is acceptable.  For production, a periodic background refresh can be added.

### D2 — Cosine in Python, not Azure RRF

**Decision:** Implement cosine similarity locally in `VSCatalogue._cosine()` rather than using Azure's hybrid RRF scoring.
**Rationale:** RRF (Reciprocal Rank Fusion) is useful when combining BM25 and vector scores across a large corpus where you can't score everything.  With 50 VSes, exact cosine is strictly better — no approximation, no threshold cutoff, no BM25 noise.
**Trade-off:** Lose BM25 keyword matching.  Mitigated by the stage keyword overlap scorer in the ranker (`weight_stage_match = 0.10`), which provides lexical signal for VS stage names and keywords.

### D3 — `mapped_vs_ids` as the canonical VS-ID signal in ChromaDB

**Decision:** Both historical PPTs and idea cards write `mapped_vs_ids` (JSON list) to every chunk's ChromaDB metadata at ingest time.  The retriever reads this field from every hit.
**Rationale:** This makes the historical signal self-contained — the retriever doesn't need a separate database lookup to know which VSes a historical chunk belonged to.  The ground-truth VS mapping (from JIRA/JITS) travels with the chunk.

### D4 — Keep Azure as metadata source only, not search engine

**Decision:** Azure AI Search remains in the architecture but only as the backing store for VS metadata (names, descriptions, stages).  `list_all()` is the only read call made at runtime.
**Rationale:** Avoids duplicating VS metadata in a second config file or database, while eliminating per-request Azure latency.  The Azure index is still the single source of truth for VS data; the in-memory catalogue is just a cache of it.

### D5 — `model_copy()` before mutating `similarity_score`

**Decision:** `retriever.py` calls `vs.model_copy()` before setting `similarity_score` on the catalogue object.
**Rationale:** `VSCatalogue._vs_map` is shared across all concurrent requests.  Mutating the shared object would cause race conditions.  The copy is cheap (Pydantic shallow copy); score fields are the only thing written.

### D6 — Warning log instead of silent drop in ranker

**Decision:** The `if vs is None` guard in the ranker now emits a `logger.warning` before `continue`, rather than silently skipping.
**Rationale:** With a fully loaded catalogue this branch should never execute.  If it does, it indicates a data inconsistency (a historical chunk references a VS ID that doesn't exist in Azure) — worth surfacing.

---

## What Azure Is Still Used For

| Use | Mechanism |
|-----|-----------|
| VS metadata (name, description, stages, keywords) | `list_all()` at startup |
| Future: semantic reranker | `AzureValueStreamSearcher.semantic_rerank()` (available, not wired) |
| Historical PPT / idea-card storage | Not in Azure — ChromaDB only |

---

## Scoring Weight Reference

| Component | Weight | Source |
|-----------|--------|--------|
| Cosine similarity (VS description vs upload) | 0.40 | `VSCatalogue.score_all()` |
| Historical boost (idea cards that used this VS) | 0.35 | ChromaDB `mapped_vs_ids` |
| Rerank score (falls back to cosine if absent) | 0.15 | Azure semantic ranker (optional) |
| Stage keyword overlap | 0.10 | `vs.stage_sequence.stages[*].keywords` |
