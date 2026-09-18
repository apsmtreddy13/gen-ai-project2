"""
Generates a realistic 3-page synthetic academic research paper PDF for testing and demonstration.
Covers all 5 core question areas:
1. Objective
2. Methodology
3. Datasets
4. Major Findings
5. Limitations
"""

import os
from pathlib import Path


def generate_sample_paper(output_path: str = "sample_paper.pdf") -> str:
    """Generate a realistic multi-page academic paper PDF."""
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        return _generate_with_pymupdf(output_path)
    except ImportError:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            return _generate_with_reportlab(output_path)
        except ImportError:
            raise RuntimeError("Neither PyMuPDF nor reportlab is available to generate sample PDF.")


def _generate_with_pymupdf(output_path: str) -> str:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    doc = fitz.open()

    # --- Page 1: Title, Abstract, Introduction (Objective), Methodology ---
    page1 = doc.new_page(width=612, height=792)  # Standard Letter size

    # Header / Title
    page1.insert_text((54, 60), "DeepScale: Modular Retrieval-Augmented Generation for Scientific Literature", fontsize=15, fontname="helv", color=(0.1, 0.1, 0.3))
    page1.insert_text((54, 82), "Alex Rivera, Priya Sharma, and David Chen", fontsize=10, fontname="helv", color=(0.3, 0.3, 0.3))
    page1.insert_text((54, 96), "Institute for Advanced Machine Learning & Neural Systems", fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))

    # Abstract
    page1.insert_text((54, 125), "Abstract", fontsize=11, fontname="helv", color=(0.1, 0.1, 0.1))
    abstract_text = (
        "Analyzing lengthy, multi-column scientific literature requires accurate factual provenance and "
        "robust reasoning. In this work, we present DeepScale, a modular retrieval-augmented generation (RAG) "
        "framework tailored for academic document understanding. DeepScale integrates layout-aware text "
        "extraction, hierarchical section chunking, and dual-stage dense retrieval with context-grounded LLM synthesis. "
        "Our empirical evaluation demonstrates an 18.4% improvement in citation accuracy and an overall factual "
        "faithfulness score of 94.2% across multiple benchmark datasets."
    )
    page1.insert_textbox(fitz.Rect(54, 135, 558, 195), abstract_text, fontsize=9.5, fontname="times-roman")

    # Section 1: Introduction (Objective)
    page1.insert_text((54, 220), "1. Introduction", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.3))
    intro_text = (
        "The primary objective of this paper is to resolve the acute hallucination problem in academic literature "
        "search and synthesize factual answers with verifiable page-level source citations. Traditional language models "
        "frequently confabulate experimental statistics or attribute claims to fictitious papers. "
        "We introduce DeepScale to bridge this gap by enforcing strict provenance checking and context-grounded generation. "
        "The overarching goal is to enable researchers to query complex multi-page preprints and immediately inspect "
        "the exact paragraphs, figures, and datasets that justify the model's generated conclusions."
    )
    page1.insert_textbox(fitz.Rect(54, 232, 558, 310), intro_text, fontsize=9.5, fontname="times-roman")

    # Section 2: Methodology
    page1.insert_text((54, 335), "2. Methodology", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.3))
    method_text = (
        "DeepScale employs a three-tiered architectural pipeline:\n\n"
        "1. Layout-Aware PDF Ingestion: We utilize block-coordinate parsing to reconstruct reading flows across "
        "multi-column academic layouts, isolating section headers from page headers and footnotes.\n\n"
        "2. Section-Bounded Semantic Chunking: Documents are chunked into contiguous windows of 800 to 1,000 characters "
        "with 150-character semantic overlaps. Chunks are strictly constrained within section boundaries (e.g. Introduction, "
        "Methodology, Experiments) and tagged with immutable metadata including physical page numbers.\n\n"
        "3. Dual-Stage Dense & Re-ranking Retrieval: Embeddings are generated using BAAI/bge-small-en-v1.5 dense representations "
        "stored in ChromaDB. Queries undergo BM25 lexical expansion followed by dense cosine similarity matching and "
        "section-priority boosting."
    )
    page1.insert_textbox(fitz.Rect(54, 348, 558, 510), method_text, fontsize=9.5, fontname="times-roman")

    # Footer
    page1.insert_text((290, 760), "1", fontsize=9, fontname="helv")

    # --- Page 2: Datasets & Major Findings ---
    page2 = doc.new_page(width=612, height=792)

    # Section 3: Experiments and Datasets
    page2.insert_text((54, 60), "3. Experiments and Datasets", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.3))
    dataset_text = (
        "To rigorously benchmark DeepScale, we evaluated the framework on four standardized research datasets:\n\n"
        "- PubMedQA: A biomedical question-answering dataset containing 211.3k research abstracts labeled with reasoning steps.\n"
        "- BioASQ 10b: An expert-curated biomedical benchmark testing complex query comprehension and multi-hop fact retrieval.\n"
        "- ArXiv-CS: A corpus of 15,000 computer science preprints spanning machine learning, distributed systems, and computer vision.\n"
        "- SCIDOCS: A multi-task scientific benchmark consisting of 1,000 query papers evaluated across citation recommendation and classification.\n\n"
        "In total, over 35,000 experimental query-answer pairs were evaluated using automated LLM-as-a-Judge and human expert review."
    )
    page2.insert_textbox(fitz.Rect(54, 75, 558, 230), dataset_text, fontsize=9.5, fontname="times-roman")

    # Section 4: Results & Major Findings
    page2.insert_text((54, 255), "4. Results and Major Findings", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.3))
    findings_text = (
        "Our extensive experimental results yielded three major empirical findings:\n\n"
        "1. Superior Factual Grounding: DeepScale achieved a factual consistency score of 94.2% on PubMedQA and 91.8% on ArXiv-CS, "
        "outperforming standard naive RAG baselines by 23.6 percentage points.\n\n"
        "2. Significant Improvement in Generation Quality: Grounded generation with section-bounded chunks yielded an 18.4% "
        "relative improvement in ROUGE-L and a 14.1% increase in BERTScore (F1) over non-chunked context injection.\n\n"
        "3. High Source Citation Precision: The automated citation alignment module achieved 96.5% page-level precision. "
        "In 965 out of 1,000 sampled test citations, the cited page and paragraph directly supported the synthesized claim without distortion."
    )
    page2.insert_textbox(fitz.Rect(54, 270, 558, 440), findings_text, fontsize=9.5, fontname="times-roman")

    # Footer
    page2.insert_text((290, 760), "2", fontsize=9, fontname="helv")

    # --- Page 3: Discussion, Limitations, and Conclusion ---
    page3 = doc.new_page(width=612, height=792)

    # Section 5: Discussion and Limitations
    page3.insert_text((54, 60), "5. Discussion and Limitations", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.3))
    limitations_text = (
        "Despite promising empirical gains, our study is subject to several important limitations:\n\n"
        "1. Latency Overhead on Very Large Documents: Parsing and embedding massive monographs (e.g. over 150 pages) creates a "
        "noticeable latency bottleneck during initial ingestion, averaging 14.2 seconds per 100 pages on standard CPU hardware.\n\n"
        "2. Dependence on Clean OCR Quality: When applied to scanned archival manuscripts with degraded optical character recognition (OCR) "
        "or complex non-standard font ligatures, text extraction accuracy dropped by 12.3%, leading to occasional chunk boundary misalignments.\n\n"
        "3. Complex Multi-Modal Elements: DeepScale currently prioritizes textual and tabular sections; equations formatted in bitmap raster images "
        "and embedded vector graphics diagrams are not yet converted into semantic embeddings, potentially omitting mathematical proof details.\n\n"
        "4. High Memory Consumption for Extended Contexts: Maintaining dense vector caches for multi-document comparisons can require significant RAM "
        "when scaling to thousands of concurrent papers."
    )
    page3.insert_textbox(fitz.Rect(54, 75, 558, 290), limitations_text, fontsize=9.5, fontname="times-roman")

    # Section 6: Conclusion
    page3.insert_text((54, 315), "6. Conclusion", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.3))
    conclusion_text = (
        "We presented DeepScale, a grounded retrieval-augmented generation system designed specifically for research papers. "
        "By pairing layout-aware multi-column ingestion with section-aware chunking and verified page citations, DeepScale demonstrates "
        "that hallucination in scientific question answering can be dramatically curtailed. Future work will investigate end-to-end "
        "multi-modal chart and equation grounding."
    )
    page3.insert_textbox(fitz.Rect(54, 330, 558, 420), conclusion_text, fontsize=9.5, fontname="times-roman")

    # Section 7: References
    page3.insert_text((54, 445), "7. References", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.3))
    refs = (
        "[1] Lewis et al., 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', NeurIPS 2020.\n"
        "[2] Jin et al., 'PubMedQA: A Dataset for Biomedical Research Question Answering', EMNLP 2019.\n"
        "[3] Cohan et al., 'SPECTER: Document-level Representation Learning using Citation-informed Transformers', ACL 2020."
    )
    page3.insert_textbox(fitz.Rect(54, 460, 558, 550), refs, fontsize=8.5, fontname="times-roman")

    page3.insert_text((290, 760), "3", fontsize=9, fontname="helv")

    # Save to disk
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(target))
    doc.close()
    return str(target.resolve())


def _generate_with_reportlab(output_path: str) -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(output_path, pagesize=letter)
    c.drawString(100, 750, "Sample Research Paper")
    c.save()
    return output_path


if __name__ == "__main__":
    path = generate_sample_paper("sample_paper.pdf")
    print(f"Sample research paper successfully created: {Path(path).name}")
