# Retrieval Redesign — VSCatalogue, Hardening & Scope

> **Status:** Implemented
> **Affected files:** `vs_catalogue.py` (new), `baseline.py` (new), `expansion_nodes.py` (new),
> `azure_search.py`, `retriever.py`, `ranker.py`, `container.py`, `parse_nodes.py`,
> `synthesis_nodes.py`, `recommendations.py`, `evaluator.py`, `security/sanitizer.py`

---

## Canonical Retrieval & Ranking Algorithm

**This is the single authoritative specification. All implementation must match it exactly.**

```
RECOMMEND(uploaded_ppt):

═══ PHASE 0: LONG-DOC GATE ════════════════════════════════════════════
  slide_count = len(uploaded_ppt.slides)

  if slide_count > LONG_DOC_SLIDE_THRESHOLD (= 40):
      query_chunks = [c for c in all_chunks
                        if c.type in {SLIDE_TITLE, SYNTHETIC_SUMMARY}]
      if not query_chunks:
          query_chunks = first_chunk_per_slide(all_chunks)[:40]
      long_doc_mode = True
  else:
      query_chunks = all_chunks
      long_doc_mode = False

═══ PHASE 1: EMBED ═════════════════════════════════════════════════════
  for each chunk c in query_chunks:
      chunk_embeddings[c.id] = EMBED(c.enriched_content or c.content)

  agg_embedding = MEAN_POOL(chunk_embeddings.values())
  # agg_embedding is used for VS scoring (Phase 2)
  # chunk_embeddings are used for ChromaDB search (Phase 3)

═══ PHASE 2: SCORE ALL VALUE STREAMS (exhaustive cosine, O(50)) ════════
  for each vs in CATALOGUE.all_50:
      vs_scores[vs.id] = COSINE(agg_embedding, CATALOGUE.vs_embedding[vs.id])

  # Key invariant: every VS gets a score, nothing is dropped here.
  # Domain filter applied post-scoring: remove vs where vs.domain ≠ domain_filter.

═══ PHASE 3: HISTORICAL EVIDENCE (per-chunk ChromaDB search) ═══════════
  if long_doc_mode:
      hist_boost = {}   ← skip; per-chunk search is too noisy on trimmed chunks
  else:
      per_chunk_k = max(1, TOP_K_HIST // len(query_chunks))   # e.g. 20 // N

      for each (chunk_c, emb_c) in zip(query_chunks, chunk_embeddings):
          hits = CHROMA_QUERY(emb_c, top_k=per_chunk_k)
          hits = [h for h in hits if h.similarity >= THRESHOLD]   # e.g. 0.6

          for hit in hits:
              for vs_id in hit.metadata.mapped_vs_ids:
                  hist_sim_sum[vs_id] += hit.similarity
                  hist_count[vs_id]   += 1

      for vs_id in hist_sim_sum:
          hist_boost[vs_id] = min(1.0,
              (hist_sim_sum[vs_id] / hist_count[vs_id])
              * (1 + 0.1 * log1p(hist_count[vs_id])))

═══ PHASE 4: STAGE KEYWORD OVERLAP (lexical, NOT semantic) ════════════
  query_text = join(c.content[:200] for c in query_chunks[:10])

  for each vs in CATALOGUE.all_50:
      stage_keywords = flatten(stage.keywords + [stage.name]
                               for stage in vs.stage_sequence.stages)
      if stage_keywords:
          matched = count(kw for kw in stage_keywords if kw.lower() in query_text.lower())
          stage_scores[vs.id] = min(1.0, matched / len(stage_keywords))
      else:
          stage_scores[vs.id] = 0.0

  # Note: "stage keyword overlap" is LEXICAL string matching, not embedding cosine.
  # It contributes 0.10 weight and rewards VS-specific terminology in the upload.

═══ PHASE 5: FINAL SCORING ════════════════════════════════════════════
  for each vs_id in ALL_VS_IDS:
      rerank = vs.rerank_score if vs.rerank_score is not None else vs_scores[vs_id]

      final_score[vs_id] = (
          0.40 * vs_scores.get(vs_id, 0.0)   # cosine: upload vs VS description
        + 0.35 * hist_boost.get(vs_id, 0.0)  # historical: similar cards mapped here
        + 0.15 * rerank                       # Azure semantic reranker (optional)
        + 0.10 * stage_scores.get(vs_id, 0.0)# lexical keyword overlap
      )

  return TOP_5(final_score, descending=True)

═══ PHASE 6: LLM SYNTHESIS ════════════════════════════════════════════
  sanitised_chunks = [sanitize_slide_content(c.content) for c in query_chunks]
  summary = LLM_SUMMARISE(sanitised_chunks)   ← 3s timeout, fallback to titles
  reasoning = LLM_EXPLAIN(summary, top_5_vs)
  confidence = top_score * (1 + score_gap)   ← to be replaced by calibrated percentile
```

### Disambiguation of terms

| Term | Meaning | Phase |
|------|---------|-------|
| **Embedding similarity** | Cosine of two embedding vectors; used for VS scoring | 2 |
| **Keyword overlap** | String-contains matching of stage keywords in query text; NOT cosine | 4 |
| **Mean-pool** | Average of all chunk embedding vectors; produces one query vector | 1 |
| **Per-chunk** | Each chunk's individual embedding; used only for ChromaDB search | 3 |
| **Historical boost** | Aggregated similarity of historically similar idea-cards mapped to a VS | 3 |

---

## Version Scope Lock

### V1 — Current (PoC) Scope

**In scope:**
- Value Stream recommendation: given a PPT, return top-5 VS IDs with scores
- Evidence: which historical chunks + VS descriptions support each recommendation
- LLM synthesis: natural language explanation of the top recommendations
- Offline evaluation: Hit@K, MRR, NDCG against historical ground-truth

**Explicitly NOT in scope for V1:**
- Stage recommendation within a VS (which stage does this idea map to?)
- VS description generation (auto-writing descriptions for new VSes)
- Multi-tenant isolation / per-user VS catalogues
- Real-time human feedback loop closing back into ranking weights
- Streaming responses

### V2 — Next milestone

- Stage recommendation: given a recommendation result, predict the most relevant stage within each VS
- Confidence calibration: replace heuristic confidence with percentile-rank from `ConfidenceCalibrator`
- Azure semantic reranker wired in for `rerank_score` (currently falls back to cosine)
- Human feedback loop: `RecommendationTraceORM.human_approved` fed back into evaluation

### V3 — Production hardening

- Multi-tenant: per-tenant ChromaDB collection + VS catalogue partition
- RBAC on evidence endpoints (evidence may contain confidential slide content)
- VS description auto-generation from stage documents
- Retention / deletion: upload files purged after configurable TTL
- Re-ingestion detection: stale embeddings flagged via `embedding_model` version mismatch

---

## Problem Statement

The original design had two bugs that partially cancelled each other:

**Bug 1 — Azure search was the only VS discovery mechanism.**
`HybridRetriever` called `azure_searcher.hybrid_search(top_k=10)` per request.  Any VS outside the top-10 window was never considered regardless of historical evidence.

**Bug 2 — Ranker silently dropped historically-matched VSes.**
`ranker.py` iterated all VS IDs from both sources but only scored those in `vs_map`, built exclusively from Azure candidates:

```python
vs = vs_map.get(vs_id)
if vs is None:
    continue   # VS only found via ChromaDB — silently dropped ← BUG
```

**Root cause:** Both document types (historical PPTs and idea cards) write `mapped_vs_ids` to ChromaDB chunk metadata at ingest time — ChromaDB alone is a complete source of VS-ID signals. Azure is only needed as a metadata store, not a per-request search engine.

---

## Component Changes

### New: `src/search/vs_catalogue.py`
Pre-loads all VSes from Azure at startup. Exposes `score_all(query_embedding)` → cosine scores for all 50 in memory (microseconds).

### New: `src/search/baseline.py` — ExhaustiveBaselineRetriever
Pure cosine ranking of all 50 VSes. No ChromaDB. No historical boost. Use in offline eval to measure whether historical + stage scoring beats cosine-only. If it doesn't, retire the complexity.

### New: `src/graph/nodes/expansion_nodes.py` — expand_query
Structured query expansion with:
- JSON schema validation on LLM output
- 3-second hard timeout on LLM call
- Deterministic fallback: slide title concatenation + term-frequency keyword extraction
- Bypass rule: skip LLM for decks ≤ 8 slides or ≤ 1500 total chars

### Modified: `src/search/azure_search.py`
Added `list_all()` for startup bulk fetch.

### Modified: `src/search/retriever.py`
- Removed `azure_searcher`, `top_k_vs`, `hybrid_alpha` — replaced by `vs_catalogue`
- Added `skip_historical: bool` param — set True in long-doc mode
- `ctx.vs_candidates` now = all 50 VSes (nothing dropped)

### Modified: `src/ranking/ranker.py`
- Silent `continue` → warning log (safety guard, should never fire)
- `_compute_stage_scores` operates on full `vs_map` (all 50), not just Azure hits

### Modified: `src/graph/nodes/parse_nodes.py`
Added `apply_long_doc_gate` node:
- Triggers at `slide_count > 40`
- Trims `query_chunks` to SLIDE_TITLE + SYNTHETIC_SUMMARY only
- Sets `state["long_doc_mode"] = True`

### Modified: `src/evaluation/evaluator.py`
Added `ConfidenceCalibrator`:
- `fit(eval_results)` — builds distribution from `(confidence, hit_at_5)` pairs
- `percentile_rank(score)` — normalises raw score to population percentile
- `approval_rate_by_band()` — reveals whether confidence actually predicts correctness

### Modified: `src/graph/nodes/synthesis_nodes.py`
All slide content passed to LLM prompts is now sanitised via `sanitize_slide_content()`.

### New: `src/security/sanitizer.py`
- `sanitize_slide_content(text)`: strips control chars, redacts prompt injection patterns, truncates
- `validate_filename(filename)`: rejects null bytes and path traversal
- `check_pptx_magic_bytes(data)`: confirms PK ZIP header independent of filename extension

### Modified: `src/api/routers/recommendations.py`
- Added magic-byte check (not just extension check)
- Added filename path-traversal validation
- Removed `source_document_path` from `EvidenceOut` (internal server path, must not be exposed)

### Modified: `src/embeddings/service.py` + `chroma_store.py` + `database.py` + `pipeline.py`
Embedding model versioning:
- `EmbeddingService.model_id` property added to all implementations
- `ChromaVectorStore.upsert_chunks(embedding_model=...)` — stored in metadata
- `ChunkORM.embedding_model` column — stored in SQLite
- `IngestionPipeline` passes `self._embedder.model_id` to both stores

---

## Design Decisions

### D1 — Load all VSes at startup, not per-request
With 50 VSes, cosine against pre-embedded vectors is microseconds. Startup cost is a one-time network call. Trade-off: service restart needed on catalogue change (V3: add background refresh).

### D2 — Cosine in Python, not Azure RRF
RRF is useful at thousands of documents. At 50, exact cosine is strictly better. BM25 keyword signal is preserved via stage keyword overlap (weight 0.10).

### D3 — `mapped_vs_ids` as canonical VS-ID signal in ChromaDB
Both historical PPTs and idea cards write `mapped_vs_ids` to every chunk's ChromaDB metadata at ingest. The retriever reads this field from every hit — no separate DB lookup needed.

### D4 — Azure as metadata store only
Azure touched once at startup (`list_all()`), never per-request. Eliminates per-request latency. Azure is still the single source of truth for VS metadata; in-memory catalogue is a cache.

### D5 — `model_copy()` before mutating similarity_score
`VSCatalogue._vs_map` is shared across concurrent requests. `model_copy()` prevents race conditions when setting `similarity_score`.

### D6 — Long-doc gate at 40 slides, not token count
Token count requires tokenising all content. Slide count is available immediately from the parser. A 40-slide deck at average 150 tokens/slide = 6000 tokens; mean-pooling that noise floor is measurable. Revisit threshold after running offline eval on large decks.

### D7 — 3-second expansion timeout, not async
LangGraph nodes run synchronously in the PoC. `concurrent.futures.ThreadPoolExecutor` gives a hard timeout without async overhead. If this becomes a bottleneck, replace with `asyncio.wait_for` when the graph is async.

### D8 — Prompt injection redaction, not rejection
Rejecting an entire upload for injection patterns in slides would create a denial-of-service vector (adversarial file). Redacting the pattern and logging a warning is safer: the upload still produces a recommendation, the attack is neutralised, and the incident is visible in logs.

### D9 — `source_document_path` stripped from API response
Internal file paths reveal server directory structure. `EvidenceOut` deliberately omits this field. If clients need evidence provenance, return a `source_document_id` (UUID) instead and resolve the path server-side.

### D10 — ConfidenceCalibrator is offline-only for V1
The calibrator requires labelled data (`human_approved` or eval ground-truth) to fit. In V1 the live endpoint continues to use the heuristic confidence score. In V2, once feedback data accumulates, the calibrator can be serialised and loaded at startup.

---

## What Azure Is Still Used For

| Use | Mechanism |
|-----|-----------|
| VS metadata (name, description, stages, keywords) | `list_all()` at startup |
| Future: semantic reranker | `AzureValueStreamSearcher.semantic_rerank()` (available, not wired) |
| Historical PPT / idea-card chunk storage | ChromaDB only |

---

## Scoring Weight Reference

| Component | Weight | Source | Phase |
|-----------|--------|--------|-------|
| Cosine: upload vs VS description | 0.40 | `VSCatalogue.score_all()` | 2 |
| Historical boost: similar cards mapped to this VS | 0.35 | ChromaDB `mapped_vs_ids` | 3 |
| Rerank score (falls back to cosine if absent) | 0.15 | Azure semantic ranker (V2) | 5 |
| Stage keyword overlap (lexical) | 0.10 | `vs.stage_sequence.stages[*].keywords` | 4 |
