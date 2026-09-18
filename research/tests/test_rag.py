"""
Comprehensive unit and integration tests for Research Paper RAG pipeline.
"""

import os
import pytest
from pathlib import Path
from sample_paper import generate_sample_paper
from rag.loader import PaperLoader, ExtractedDocument
from rag.chunker import SectionAwareChunker, TextChunk
from rag.vector_store import VectorIndex, SimpleBM25Fallback
from rag.citation import CitationParser
from rag.generator import GroundedGenerator


@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory) -> str:
    temp_dir = tmp_path_factory.mktemp("papers")
    pdf_path = temp_dir / "test_paper.pdf"
    generate_sample_paper(str(pdf_path))
    return str(pdf_path)


def test_sample_paper_generation(sample_pdf_path):
    assert Path(sample_pdf_path).exists()
    assert Path(sample_pdf_path).stat().st_size > 1000


def test_loader_pdf_extraction(sample_pdf_path):
    loader = PaperLoader()
    doc = loader.load_pdf(sample_pdf_path)

    assert isinstance(doc, ExtractedDocument)
    assert doc.total_pages == 3
    assert len(doc.pages) == 3
    assert "DeepScale" in doc.title or "DeepScale" in doc.full_text

    # Verify key sections detected
    section_canonicals = [s.canonical_type for s in doc.sections]
    assert "Introduction" in section_canonicals or any("intro" in s.name.lower() for s in doc.sections)
    assert "Methodology" in section_canonicals or any("method" in s.name.lower() for s in doc.sections)
    assert "Experiments & Datasets" in section_canonicals or any("dataset" in s.name.lower() for s in doc.sections)
    assert "Results & Findings" in section_canonicals or any("finding" in s.name.lower() for s in doc.sections)
    assert "Discussion & Limitations" in section_canonicals or any("limitation" in s.name.lower() for s in doc.sections)


def test_chunker_section_preservation(sample_pdf_path):
    loader = PaperLoader()
    doc = loader.load_pdf(sample_pdf_path)

    chunker = SectionAwareChunker(chunk_size=600, chunk_overlap=100)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 4
    for c in chunks:
        assert isinstance(c, TextChunk)
        assert 1 <= c.page_number <= 3
        assert len(c.content) > 0
        assert c.char_count == len(c.content)
        assert "[" in c.citation_label and "]" in c.citation_label


def test_bm25_search_retrieval(sample_pdf_path):
    loader = PaperLoader()
    doc = loader.load_pdf(sample_pdf_path)
    chunker = SectionAwareChunker()
    chunks = chunker.chunk_document(doc)

    bm25 = SimpleBM25Fallback()
    bm25.index(chunks)

    # Search for datasets
    res_datasets = bm25.search("What datasets were used?", top_k=3)
    assert len(res_datasets) > 0
    joined_text = " ".join([r.chunk.content for r in res_datasets]).lower()
    assert "pubmedqa" in joined_text or "bioasq" in joined_text or "dataset" in joined_text

    # Search for limitations
    res_limits = bm25.search("What are the limitations?", top_k=3)
    assert len(res_limits) > 0
    joined_limits = " ".join([r.chunk.content for r in res_limits]).lower()
    assert "limitation" in joined_limits or "latency" in joined_limits or "ocr" in joined_limits


def test_citation_parsing():
    mock_chunks = [
        TextChunk("c1", "DeepScale", 1, "Introduction", "Introduction", "The objective is to fix hallucinations.", 40),
        TextChunk("c2", "DeepScale", 2, "Experiments and Datasets", "Experiments & Datasets", "PubMedQA and BioASQ datasets were evaluated.", 48),
        TextChunk("c3", "DeepScale", 3, "Discussion and Limitations", "Discussion & Limitations", "A key limitation is latency overhead.", 38),
    ]

    answer = (
        "The objective of the paper is to resolve hallucinations [Page 1, Introduction]. "
        "The authors tested on PubMedQA and BioASQ [Page 2, Experiments and Datasets]. "
        "However, latency remains a limitation [Page 3]."
    )

    citations = CitationParser.extract_citations(answer, mock_chunks)
    assert len(citations) >= 3

    p1 = next((c for c in citations if c.page_number == 1), None)
    assert p1 is not None
    assert p1.matched_chunk.chunk_id == "c1"

    p2 = next((c for c in citations if c.page_number == 2), None)
    assert p2 is not None
    assert p2.matched_chunk.chunk_id == "c2"

    p3 = next((c for c in citations if c.page_number == 3), None)
    assert p3 is not None
    assert p3.matched_chunk.chunk_id == "c3"


def test_five_core_questions_end_to_end(sample_pdf_path):
    loader = PaperLoader()
    doc = loader.load_pdf(sample_pdf_path)
    chunker = SectionAwareChunker()
    chunks = chunker.chunk_document(doc)

    index = VectorIndex()
    index.index_chunks(chunks)

    generator = GroundedGenerator(provider="local")

    test_questions = [
        "What is the objective of the paper?",
        "What methodology was used?",
        "What datasets were used?",
        "What are the major findings?",
        "What are the limitations?",
    ]

    for q in test_questions:
        retrieved = index.search(q, top_k=3)
        assert len(retrieved) > 0

        gen_result = generator.generate(q, retrieved)
        assert len(gen_result.answer) > 20
        assert len(gen_result.citations) > 0
        # Citation page numbers must be between 1 and 3
        for cit in gen_result.citations:
            assert 1 <= cit.page_number <= 3
