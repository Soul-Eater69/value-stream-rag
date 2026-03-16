"""PPT ingestion pipeline – parsing, chunking, and indexing."""

from .parser import PPTParser, ParsedSlide, ParsedDocument
from .pipeline import IngestionPipeline

__all__ = ["PPTParser", "ParsedSlide", "ParsedDocument", "IngestionPipeline"]
