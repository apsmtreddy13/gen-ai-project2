"""
Section-aware semantic chunker for research papers.
Preserves page numbers, section headers, and semantic boundaries.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional
from .loader import ExtractedDocument, DocumentPage


@dataclass
class TextChunk:
    chunk_id: str
    paper_title: str
    page_number: int
    section: str
    canonical_section: str
    content: str
    char_count: int

    @property
    def citation_label(self) -> str:
        """Returns standard citation label, e.g. [Page 3, Methodology]."""
        return f"[Page {self.page_number}, {self.section}]"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "paper_title": self.paper_title,
            "page_number": self.page_number,
            "section": self.section,
            "canonical_section": self.canonical_section,
            "content": self.content,
            "char_count": self.char_count,
        }


class SectionAwareChunker:
    """Chunks research papers while preserving section context and page metadata."""

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, doc: ExtractedDocument) -> List[TextChunk]:
        """Convert an ExtractedDocument into an indexed list of TextChunks."""
        chunks: List[TextChunk] = []
        global_chunk_idx = 0

        # Build mapping of page to active sections
        current_section = "Overview"
        current_canonical = "Overview"

        for page in doc.pages:
            # If the page started with new detected sections, update tracking
            page_text = page.text.strip()
            if not page_text:
                continue

            # Split page text into semantic paragraphs
            paragraphs = re.split(r"\n\s*\n", page_text)

            # Detect section transitions inside paragraphs
            current_buffer = ""

            for p in paragraphs:
                p_clean = p.strip()
                if not p_clean:
                    continue

                # Check if paragraph is a section header
                new_sec = self._detect_section_in_paragraph(p_clean)
                if new_sec:
                    # Flush current buffer if non-empty
                    if current_buffer:
                        sub_chunks = self._split_text_into_chunks(
                            current_buffer,
                            doc.title,
                            page.page_number,
                            current_section,
                            current_canonical,
                            global_chunk_idx,
                        )
                        chunks.extend(sub_chunks)
                        global_chunk_idx += len(sub_chunks)
                        current_buffer = ""

                    current_section, current_canonical = new_sec

                # Append paragraph to buffer
                if current_buffer:
                    current_buffer += "\n\n" + p_clean
                else:
                    current_buffer = p_clean

                # If buffer exceeds chunk size, split and flush
                if len(current_buffer) >= self.chunk_size:
                    sub_chunks = self._split_text_into_chunks(
                        current_buffer,
                        doc.title,
                        page.page_number,
                        current_section,
                        current_canonical,
                        global_chunk_idx,
                    )
                    chunks.extend(sub_chunks)
                    global_chunk_idx += len(sub_chunks)
                    # Retain overlap from end of buffer
                    overlap_chars = current_buffer[-self.chunk_overlap:]
                    current_buffer = overlap_chars.strip()

            # Flush any remaining buffer at end of page
            if len(current_buffer) >= self.min_chunk_size:
                sub_chunks = self._split_text_into_chunks(
                    current_buffer,
                    doc.title,
                    page.page_number,
                    current_section,
                    current_canonical,
                    global_chunk_idx,
                )
                chunks.extend(sub_chunks)
                global_chunk_idx += len(sub_chunks)

        return chunks

    def _detect_section_in_paragraph(self, text: str) -> Optional[tuple[str, str]]:
        """Determine if a short paragraph represents a section title."""
        first_line = text.splitlines()[0].strip()
        if len(first_line) > 75:
            return None

        clean = first_line.lower()
        from .loader import ACADEMIC_SECTION_PATTERNS
        for pattern, canon_type in ACADEMIC_SECTION_PATTERNS:
            if re.search(pattern, clean, re.IGNORECASE):
                return (first_line, canon_type)
        return None

    def _split_text_into_chunks(
        self,
        text: str,
        title: str,
        page_num: int,
        section: str,
        canonical_section: str,
        start_idx: int,
    ) -> List[TextChunk]:
        """Split a text block into chunks with boundary awareness."""
        if len(text) <= self.chunk_size:
            return [
                TextChunk(
                    chunk_id=f"p{page_num}_c{start_idx}",
                    paper_title=title,
                    page_number=page_num,
                    section=section,
                    canonical_section=canonical_section,
                    content=text.strip(),
                    char_count=len(text.strip()),
                )
            ]

        results: List[TextChunk] = []
        sentences = re.split(r"(?<=[.?!])\s+", text)
        current_chunk = ""
        chunk_counter = start_idx

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            if not current_chunk:
                current_chunk = sent_clean
            elif len(current_chunk) + len(sent_clean) + 1 <= self.chunk_size:
                current_chunk += " " + sent_clean
            else:
                # Flush current chunk
                results.append(
                    TextChunk(
                        chunk_id=f"p{page_num}_c{chunk_counter}",
                        paper_title=title,
                        page_number=page_num,
                        section=section,
                        canonical_section=canonical_section,
                        content=current_chunk.strip(),
                        char_count=len(current_chunk.strip()),
                    )
                )
                chunk_counter += 1

                # Carry overlap
                overlap = current_chunk[-self.chunk_overlap:]
                current_chunk = overlap + " " + sent_clean

        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            results.append(
                TextChunk(
                    chunk_id=f"p{page_num}_c{chunk_counter}",
                    paper_title=title,
                    page_number=page_num,
                    section=section,
                    canonical_section=canonical_section,
                    content=current_chunk.strip(),
                    char_count=len(current_chunk.strip()),
                )
            )

        return results
