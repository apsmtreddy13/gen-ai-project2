"""
RAG Engine for Academic and Research Papers.
Provides layout-aware PDF ingestion, section-aware semantic chunking,
dense vector retrieval, context-grounded generation, and source citation attribution.
"""

from .loader import PaperLoader, ExtractedDocument, DocumentPage, PaperSection
from .chunker import SectionAwareChunker, TextChunk
from .vector_store import VectorIndex, SearchResult
from .generator import GroundedGenerator, GenerationResult
from .citation import CitationParser, Citation

__all__ = [
    "PaperLoader",
    "ExtractedDocument",
    "DocumentPage",
    "PaperSection",
    "SectionAwareChunker",
    "TextChunk",
    "VectorIndex",
    "SearchResult",
    "GroundedGenerator",
    "GenerationResult",
    "CitationParser",
    "Citation",
]
