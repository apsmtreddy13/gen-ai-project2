"""
Layout-aware PDF ingestion and section extraction for academic papers.
"""

from __future__ import annotations
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None


# Academic standard section patterns
ACADEMIC_SECTION_PATTERNS = [
    (r"^(?:abstract)\b", "Abstract"),
    (r"^(?:\d+[\.\s]+)?(?:introduction)\b", "Introduction"),
    (r"^(?:\d+[\.\s]+)?(?:related\s+work|background|prior\s+work)\b", "Related Work"),
    (r"^(?:\d+[\.\s]+)?(?:methodology|method|proposed\s+method|architecture|model|framework|approach)\b", "Methodology"),
    (r"^(?:\d+[\.\s]+)?(?:experiments?|experimental\s+setup|experimental\s+results|evaluation|datasets?)\b", "Experiments & Datasets"),
    (r"^(?:\d+[\.\s]+)?(?:results?|major\s+findings?|findings?|empirical\s+results)\b", "Results & Findings"),
    (r"^(?:\d+[\.\s]+)?(?:limitations?|threats\s+to\s+validity|discussion)\b", "Discussion & Limitations"),
    (r"^(?:\d+[\.\s]+)?(?:conclusion|concluding\s+remarks|future\s+work)\b", "Conclusion"),
    (r"^(?:\d+[\.\s]+)?(?:references?|bibliography)\b", "References"),
    (r"^(?:\d+[\.\s]+)?(?:appendix|supplementary\s+material)\b", "Appendix"),
]


@dataclass
class PaperSection:
    name: str
    canonical_type: str  # e.g., 'Methodology', 'Datasets', 'Limitations', etc.
    start_page: int
    content: str = ""


@dataclass
class DocumentPage:
    page_number: int  # 1-indexed
    text: str
    detected_sections: List[str] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    filename: str
    title: str
    total_pages: int
    pages: List[DocumentPage]
    sections: List[PaperSection]
    full_text: str

    def get_section_by_type(self, canonical_type: str) -> List[PaperSection]:
        return [s for s in self.sections if s.canonical_type.lower() == canonical_type.lower()]


class PaperLoader:
    """Ingests academic PDF files and performs column-aware layout parsing and section detection."""

    def __init__(self):
        if fitz is None:
            raise ImportError(
                "PyMuPDF ('fitz') is required for PDF ingestion. Please run: pip install pymupdf"
            )

    def load_pdf(
        self,
        source: Union[str, Path, bytes, BinaryIO],
        filename: Optional[str] = None
    ) -> ExtractedDocument:
        """Load and parse an academic paper from a file path, bytes, or file-like object."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            doc_name = filename or path.name
            doc = fitz.open(str(path))
        elif isinstance(source, bytes):
            doc_name = filename or "uploaded_paper.pdf"
            doc = fitz.open(stream=source, filetype="pdf")
        elif hasattr(source, "read"):
            doc_name = filename or getattr(source, "name", "uploaded_paper.pdf")
            bytes_data = source.read()
            doc = fitz.open(stream=bytes_data, filetype="pdf")
        else:
            raise ValueError(f"Unsupported input type for source: {type(source)}")

        total_pages = len(doc)
        pages: List[DocumentPage] = []
        all_sections: List[PaperSection] = []
        current_section: Optional[PaperSection] = None
        doc_title = ""

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # Extract layout-aware blocks (handles 2-column papers)
            page_text, detected_headings = self._extract_page_blocks(page, page_num)

            # Guess paper title from first page if not already determined
            if page_idx == 0:
                doc_title = self._detect_title(page, page_text, doc_name)

            page_obj = DocumentPage(
                page_number=page_num,
                text=page_text,
                detected_sections=[h[0] for h in detected_headings]
            )
            pages.append(page_obj)

            # Associate text to sections
            for heading_text, canon_type in detected_headings:
                if current_section is not None:
                    all_sections.append(current_section)
                current_section = PaperSection(
                    name=heading_text,
                    canonical_type=canon_type,
                    start_page=page_num,
                    content=""
                )

            if current_section is not None:
                current_section.content += f"\n\n[Page {page_num}]\n" + page_text
            else:
                # Default preliminary section (e.g. Title & Abstract header)
                current_section = PaperSection(
                    name="Header & Overview",
                    canonical_type="Overview",
                    start_page=page_num,
                    content=page_text
                )

        if current_section is not None and current_section not in all_sections:
            all_sections.append(current_section)

        full_text = "\n\n".join([f"--- Page {p.page_number} ---\n{p.text}" for p in pages])

        doc.close()

        return ExtractedDocument(
            filename=doc_name,
            title=doc_title,
            total_pages=total_pages,
            pages=pages,
            sections=all_sections,
            full_text=full_text
        )

    def _extract_page_blocks(
        self, page: fitz.Page, page_num: int
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Extract text from page with column awareness.
        Standard academic papers often use two columns.
        Sorting blocks by column ensures text isn't read horizontally across columns.
        """
        page_rect = page.rect
        page_width = page_rect.width
        mid_x = page_width / 2.0

        # get_text("blocks") returns tuples: (x0, y0, x1, y1, text, block_no, block_type)
        raw_blocks = page.get_text("blocks")
        text_blocks = [b for b in raw_blocks if b[6] == 0 and b[4].strip()]

        if not text_blocks:
            return "", []

        # Check if page looks like a two-column layout
        left_blocks = [b for b in text_blocks if b[2] <= mid_x + 35]
        right_blocks = [b for b in text_blocks if b[0] >= mid_x - 35]
        spanning_blocks = [b for b in text_blocks if b[0] < mid_x - 35 and b[2] > mid_x + 35]

        is_two_column = len(left_blocks) >= 2 and len(right_blocks) >= 2 and len(spanning_blocks) < len(text_blocks) * 0.4

        if is_two_column:
            # Sort spanning blocks by y0 (e.g. title / abstract spans top)
            top_spanning = sorted([b for b in spanning_blocks if b[1] < page_rect.height * 0.4], key=lambda b: b[1])
            bottom_spanning = sorted([b for b in spanning_blocks if b[1] >= page_rect.height * 0.4], key=lambda b: b[1])
            left_col = sorted(left_blocks, key=lambda b: (b[1], b[0]))
            right_col = sorted(right_blocks, key=lambda b: (b[1], b[0]))
            ordered_blocks = top_spanning + left_col + right_col + bottom_spanning
        else:
            # Sort top to bottom, left to right
            ordered_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))

        extracted_text_parts = []
        detected_headings: List[Tuple[str, str]] = []

        for block in ordered_blocks:
            raw_content = block[4].strip()
            cleaned_content = self._clean_text(raw_content)
            if not cleaned_content:
                continue

            # Check if this block represents a section heading
            heading_info = self._check_heading(cleaned_content)
            if heading_info:
                detected_headings.append(heading_info)

            extracted_text_parts.append(cleaned_content)

        page_text = "\n\n".join(extracted_text_parts)
        return page_text, detected_headings

    def _clean_text(self, text: str) -> str:
        """Clean hyphenated line breaks, dangling spaces, and control characters."""
        # Join words broken by hyphenation at line breaks: "meth-\nodology" -> "methodology"
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
        # Normalize excessive whitespace within lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        # Remove empty lines and rejoin
        cleaned = "\n".join([line for line in lines if line])
        return cleaned

    def _check_heading(self, block_text: str) -> Optional[Tuple[str, str]]:
        """Identify if a text block corresponds to an academic section heading."""
        first_line = block_text.splitlines()[0].strip()
        # Headings are typically short
        if len(first_line) > 85:
            return None

        # Ignore obvious references or equations
        clean_first = first_line.lower()

        for pattern, canon_type in ACADEMIC_SECTION_PATTERNS:
            if re.search(pattern, clean_first, re.IGNORECASE):
                return (first_line, canon_type)

        return None

    def _detect_title(self, page: fitz.Page, first_page_text: str, fallback: str) -> str:
        """Infer paper title from PDF metadata or prominent top text block on page 1."""
        meta_title = page.parent.metadata.get("title") if page.parent else None
        if meta_title and len(meta_title.strip()) > 5 and not meta_title.lower().endswith(".pdf"):
            return meta_title.strip()

        # Check largest font size blocks on page 1
        try:
            blocks = page.get_text("dict").get("blocks", [])
            max_size = 0.0
            candidate_text = ""
            for b in blocks:
                if b.get("type") == 0:  # text block
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            span_size = span.get("size", 0.0)
                            span_text = span.get("text", "").strip()
                            if span_size > max_size and len(span_text) > 6:
                                max_size = span_size
                                candidate_text = span_text
                            elif span_size == max_size and max_size > 14:
                                candidate_text += " " + span_text

            if candidate_text and len(candidate_text.strip()) > 8:
                return re.sub(r"\s+", " ", candidate_text).strip()
        except Exception:
            pass

        # Fallback to first non-empty lines of page 1 or document filename
        lines = [line.strip() for line in first_page_text.splitlines() if len(line.strip()) > 8]
        if lines:
            return lines[0][:120]

        return Path(fallback).stem
