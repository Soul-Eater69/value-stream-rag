"""
Integration tests for the ingestion pipeline.

These tests use MockEmbeddingService and an in-memory ChromaDB instance
so no real external services are needed.  They validate end-to-end flow:
  PPTX file → parse → chunk → embed → upsert → query
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.chunking.chunker import ChunkingConfig
from src.embeddings.service import MockEmbeddingService
from src.ingestion.pipeline import IngestionPipeline, IngestionRecord
from src.search.chroma_store import ChromaVectorStore


@pytest.fixture
def tmp_chroma(tmp_path):
    return ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_collection",
    )


@pytest.fixture
def mock_pipeline(tmp_chroma, tmp_path):
    """Build an IngestionPipeline with all mocked/local dependencies."""
    from src.models.database import get_session_factory
    db_url = f"sqlite:///{tmp_path}/test.db"
    session_factory = get_session_factory(db_url)

    return IngestionPipeline(
        embedding_service=MockEmbeddingService(dimension=128),
        vector_store=tmp_chroma,
        db_session_factory=session_factory,
        chunking_config=ChunkingConfig(max_tokens=256),
    )


@pytest.fixture
def sample_pptx(tmp_path) -> Path:
    """Create a minimal PPTX file using python-pptx for testing."""
    try:
        from pptx import Presentation
        from pptx.util import Inches

        prs = Presentation()
        slide_layout = prs.slide_layouts[1]  # Title and Content layout

        # Slide 1: Problem Statement
        slide1 = prs.slides.add_slide(slide_layout)
        slide1.shapes.title.text = "Problem Statement"
        slide1.placeholders[1].text = (
            "We face a significant bottleneck in our order management process. "
            "This leads to customer dissatisfaction and revenue loss."
        )

        # Slide 2: Proposed Solution
        slide2 = prs.slides.add_slide(slide_layout)
        slide2.shapes.title.text = "Proposed Solution"
        slide2.placeholders[1].text = (
            "Implement an automated inventory tracking system "
            "integrated with the ERP platform."
        )

        path = tmp_path / "test_idea_card.pptx"
        prs.save(str(path))
        return path

    except ImportError:
        pytest.skip("python-pptx not installed")


@pytest.mark.integration
class TestIngestionPipelineIntegration:
    def test_ingest_single_success(self, mock_pipeline, sample_pptx):
        record = IngestionRecord(
            file_path=str(sample_pptx),
            mapped_value_stream_ids=["VS001", "VS002"],
            mapped_value_stream_names=["Order Management", "Inventory Control"],
            jira_ticket="JIRA-1234",
            title="Test Idea Card",
            domain="Supply Chain",
        )
        result = mock_pipeline.ingest_single(record)

        assert result.success is True
        assert result.slide_count >= 1
        assert result.chunk_count > 0

    def test_ingest_creates_chroma_entries(self, mock_pipeline, tmp_chroma, sample_pptx):
        record = IngestionRecord(
            file_path=str(sample_pptx),
            mapped_value_stream_ids=["VS001"],
        )
        result = mock_pipeline.ingest_single(record)

        assert result.success is True
        count = tmp_chroma.count()
        assert count == result.chunk_count

    def test_query_returns_mapped_vs_ids(self, mock_pipeline, tmp_chroma, sample_pptx):
        """After ingestion, querying with a relevant embedding should return
        chunks whose metadata contains the mapped VS IDs."""
        from src.embeddings.service import MockEmbeddingService

        record = IngestionRecord(
            file_path=str(sample_pptx),
            mapped_value_stream_ids=["VS001"],
        )
        mock_pipeline.ingest_single(record)

        embedder = MockEmbeddingService(dimension=128)
        query_vec = embedder.embed("order management problem")
        hits = tmp_chroma.query(query_embedding=query_vec, top_k=5)

        assert len(hits) > 0
        all_vs_ids = [vid for h in hits for vid in h["mapped_vs_ids"]]
        assert "VS001" in all_vs_ids

    def test_ingest_batch(self, mock_pipeline, sample_pptx, tmp_path):
        """Ingest two records in a batch."""
        # Make a copy of the PPTX for the second record
        second = tmp_path / "second.pptx"
        import shutil
        shutil.copy(str(sample_pptx), str(second))

        records = [
            IngestionRecord(
                file_path=str(sample_pptx),
                mapped_value_stream_ids=["VS001"],
            ),
            IngestionRecord(
                file_path=str(second),
                mapped_value_stream_ids=["VS002"],
            ),
        ]
        results = mock_pipeline.ingest_batch(records)

        assert len(results) == 2
        assert all(r.success for r in results)

    def test_ingest_nonexistent_file_fails_gracefully(self, mock_pipeline):
        record = IngestionRecord(
            file_path="/nonexistent/fake.pptx",
            mapped_value_stream_ids=["VS001"],
        )
        result = mock_pipeline.ingest_single(record)
        assert result.success is False
        assert result.error is not None
