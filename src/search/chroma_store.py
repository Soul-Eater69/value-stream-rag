"""
ChromaDB vector store for historical idea-card chunks.

Acts as the local vector index for historical PPT embeddings.
Stores chunk content + rich metadata so retrieval results can be traced
back to their source documents and ground-truth Value Stream mappings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from src.models.domain import Chunk, ChunkType

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Wrapper around ChromaDB for persisting and querying historical PPT chunks.

    Schema (stored per chunk):
      - document: embedding_text (enriched content)
      - id: str(chunk.id)
      - metadata:
          source_document_id   : str
          source_document_path : str
          slide_index          : int
          slide_title          : str
          chunk_type           : str
          section_label        : str
          table_index          : int | -1
          mapped_vs_ids        : json-encoded list[str]  (JSON string)
    """

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = "historical_idea_cards",
        embedding_function=None,  # Optional chromadb EF; if None, embeddings are pre-computed
    ) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function,
        )
        logger.info(
            "ChromaVectorStore ready: collection=%s persist_dir=%s",
            collection_name,
            persist_dir,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        mapped_value_stream_ids: list[str],
    ) -> None:
        """Insert or update chunk embeddings and metadata."""
        import json

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )

        ids = [str(c.id) for c in chunks]
        documents = [c.embedding_text for c in chunks]
        metadatas = [
            {
                "source_document_id": c.source_document_id,
                "source_document_path": c.source_document_path,
                "slide_index": c.slide_index,
                "slide_title": c.slide_title,
                "chunk_type": c.chunk_type.value,
                "section_label": c.section_label,
                "table_index": c.table_index if c.table_index is not None else -1,
                "mapped_vs_ids": json.dumps(mapped_value_stream_ids),
            }
            for c in chunks
        ]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.debug("Upserted %d chunks into ChromaDB", len(chunks))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query the collection and return enriched result dicts.

        Each result dict contains:
          id, document, distance, metadata (source_document_id, etc.)
        """
        import json

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results: list[dict[str, Any]] = []
        for i, (cid, doc, meta, dist) in enumerate(
            zip(
                raw["ids"][0],
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            )
        ):
            # Convert cosine distance → similarity
            similarity = 1.0 - float(dist)
            vs_ids = json.loads(meta.get("mapped_vs_ids", "[]"))
            results.append(
                {
                    "id": cid,
                    "document": doc,
                    "similarity": similarity,
                    "source_document_id": meta.get("source_document_id", ""),
                    "source_document_path": meta.get("source_document_path", ""),
                    "slide_index": meta.get("slide_index", 0),
                    "slide_title": meta.get("slide_title", ""),
                    "chunk_type": ChunkType(meta.get("chunk_type", "slide_text")),
                    "section_label": meta.get("section_label", ""),
                    "table_index": meta.get("table_index", -1),
                    "mapped_vs_ids": vs_ids,
                }
            )

        return results

    def count(self) -> int:
        return self._collection.count()

    def delete_by_document(self, source_document_id: str) -> None:
        """Remove all chunks belonging to a source document."""
        self._collection.delete(
            where={"source_document_id": {"$eq": source_document_id}}
        )
