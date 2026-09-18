"""
Command-line interface for the Research Paper RAG system.
Usage:
    python cli.py sample_paper.pdf --question "What datasets were used?"
    python cli.py sample_paper.pdf --preset all
"""

import argparse
import sys
from pathlib import Path
from rag.loader import PaperLoader
from rag.chunker import SectionAwareChunker
from rag.vector_store import VectorIndex
from rag.generator import GroundedGenerator

PRESET_QUESTIONS = {
    "objective": "What is the objective of the paper?",
    "methodology": "What methodology was used?",
    "datasets": "What datasets were used?",
    "findings": "What are the major findings?",
    "limitations": "What are the limitations?",
}


def run_rag_cli(
    pdf_path: str,
    question: str = None,
    preset: str = None,
    provider: str = "gemini",
    api_key: str = None,
    top_k: int = 4,
):
    path = Path(pdf_path)
    if not path.exists():
        print(f"Error: File not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[1/4] Ingesting and parsing PDF: {path.name}...")
    loader = PaperLoader()
    doc = loader.load_pdf(path)
    print(f"      Detected Title: {doc.title}")
    print(f"      Total Pages:    {doc.total_pages}")
    print(f"      Sections Found: {', '.join([s.name for s in doc.sections])}")

    print(f"\n[2/4] Chunking document with section boundaries...")
    chunker = SectionAwareChunker()
    chunks = chunker.chunk_document(doc)
    print(f"      Created {len(chunks)} contextual chunks.")

    print(f"\n[3/4] Indexing chunks into vector store...")
    index = VectorIndex()
    index.index_chunks(chunks)
    print(f"      Vector indexing complete.")

    generator = GroundedGenerator(provider=provider, api_key=api_key)

    questions_to_ask = []
    if preset == "all":
        questions_to_ask = list(PRESET_QUESTIONS.values())
    elif preset in PRESET_QUESTIONS:
        questions_to_ask = [PRESET_QUESTIONS[preset]]
    elif question:
        questions_to_ask = [question]
    else:
        questions_to_ask = list(PRESET_QUESTIONS.values())

    print(f"\n[4/4] Generating context-grounded responses...")
    print("=" * 80)

    for q in questions_to_ask:
        print(f"\nQUERY: {q}")
        print("-" * 80)
        retrieved = index.search(q, top_k=top_k)
        result = generator.generate(q, retrieved)

        print(result.answer)
        print("\nVerified Citations:")
        if result.citations:
            for cit in result.citations:
                snip = f" -> \"{cit.excerpt_snippet}\"" if cit.excerpt_snippet else ""
                print(f"  * [Page {cit.page_number} | Section: {cit.section}]{snip}")
        else:
            print("  * None explicitly cited.")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Research Paper Grounded RAG Assistant")
    parser.add_argument("pdf", help="Path to research paper PDF")
    parser.add_argument("-q", "--question", help="Specific question to answer")
    parser.add_argument(
        "-p", "--preset",
        choices=["objective", "methodology", "datasets", "findings", "limitations", "all"],
        help="Run standard academic analysis preset"
    )
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai", "groq", "local"], help="LLM Provider")
    parser.add_argument("--api-key", help="API key for selected provider")
    parser.add_argument("-k", "--top-k", type=int, default=4, help="Number of chunks to retrieve")

    args = parser.parse_args()
    run_rag_cli(
        pdf_path=args.pdf,
        question=args.question,
        preset=args.preset,
        provider=args.provider,
        api_key=args.api_key,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
