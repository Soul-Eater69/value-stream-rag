# Mermaid Diagram Reference

All diagrams in this system are authored as Mermaid.js markup and are embedded
in `ARCHITECTURE.md`. Below is a standalone reference with each diagram labelled.

---

## Diagram 1: System Architecture

```mermaid
graph TB
    subgraph Client
        U["User / Portal"]
    end
    subgraph API_Layer["API Layer"]
        FA["FastAPI App"]
    end
    subgraph Orchestration
        IG["Ingestion Graph"]
        RG["Recommendation Graph"]
    end
    subgraph Storage
        AZS[("Azure AI Search — Value Stream Index")]
        CDB[("ChromaDB — Historical PPT Index")]
        SDB[("SQLite Metadata DB")]
    end
    subgraph Retrieval
        HYB["Hybrid Retriever"]
        RNK["Ranker"]
        SYN["LLM Synthesiser"]
    end

    U --> FA
    FA --> RG --> HYB --> AZS
    HYB --> CDB
    HYB --> RNK --> SYN --> FA
    FA --> IG --> CDB
    IG --> SDB
```

---

## Diagram 2: Ingestion Flow

```mermaid
sequenceDiagram
    participant CLI as CLI / API
    participant G as Ingestion Graph
    participant P as PPTParser
    participant C as Chunker
    participant E as Embedder
    participant CH as ChromaDB
    participant DB as SQLite

    CLI->>G: ingest_batch(records)
    loop For each PPT record
        G->>P: parse(file_path)
        P-->>G: ParsedDocument (slides + tables)
        G->>C: chunk_document(doc)
        C-->>G: list[Chunk]
        G->>E: embed_batch(texts)
        E-->>G: list[list[float]]
        G->>CH: upsert_chunks(chunks, embeddings, vs_ids)
        G->>DB: persist(card, chunks)
    end
    G-->>CLI: IngestionResult[]
```

---

## Diagram 3: Runtime Recommendation Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as FastAPI
    participant G as Recommendation Graph
    participant AZ as Azure AI Search
    participant CH as ChromaDB
    participant L as GPT-4o

    U->>A: POST /upload (PPTX)
    A->>G: invoke(state)
    G->>G: parse → chunk → embed (in memory)
    G->>L: summarise PPT content
    L-->>G: 2-3 sentence summary
    G->>AZ: hybrid_search(agg_embedding, query_text)
    AZ-->>G: top-K Value Streams
    G->>CH: query(per-chunk embeddings)
    CH-->>G: historical chunks + mapped VS IDs
    G->>G: rank (weighted scoring)
    G->>L: synthesise reasoning
    L-->>G: natural language explanation
    G-->>A: RecommendationResult
    A-->>U: JSON response
```

---

## Diagram 4: LangGraph Workflow

```mermaid
stateDiagram-v2
    [*] --> parse
    parse --> chunk : parsed
    parse --> error : ParseError

    chunk --> embed : chunked
    chunk --> error : ChunkError

    embed --> summarise : embedded
    embed --> error : EmbedError

    summarise --> retrieve
    retrieve --> rank : retrieved
    retrieve --> error : RetrievalError

    rank --> synthesise : ranked
    rank --> error : RankingError

    synthesise --> persist
    persist --> [*]
    error --> [*]
```

---

## Diagram 5: Data Model Relationships

```mermaid
erDiagram
    HistoricalIdeaCard ||--o{ Chunk : "contains"
    HistoricalIdeaCard }o--o{ ValueStream : "mapped_to"
    ValueStream ||--o| StageSequence : "has"
    StageSequence ||--o{ Stage : "ordered_stages"
    RecommendationResult }o--o{ ValueStream : "recommends"
    RecommendationResult ||--o{ RetrievalEvidence : "supported_by"
    RetrievalEvidence }o--o| Chunk : "references"
    UploadedPPT ||--o{ Chunk : "chunked_into"
    UploadedPPT ||--o| RecommendationResult : "produces"
```
