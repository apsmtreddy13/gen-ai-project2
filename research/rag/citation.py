"""
Citation parser and source attribution engine for research paper Q&A.
Maps cited claims back to physical PDF pages, section headers, and exact excerpt snippets.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional
from .chunker import TextChunk


@dataclass
class Citation:
    page_number: int
    section: str
    raw_citation: str
    matched_chunk: Optional[TextChunk] = None
    excerpt_snippet: str = ""

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "section": self.section,
            "raw_citation": self.raw_citation,
            "chunk_id": self.matched_chunk.chunk_id if self.matched_chunk else None,
            "excerpt_snippet": self.excerpt_snippet,
        }


class CitationParser:
    """Parses citation tags from LLM responses and resolves them against retrieved context chunks."""

    # Matches [Page X] or [Page X, Section Name]
    CITATION_REGEX = re.compile(r"\[Page\s+(\d+)(?:,\s*([^\]]+))?\]", re.IGNORECASE)

    @classmethod
    def extract_citations(cls, answer_text: str, retrieved_chunks: List[TextChunk]) -> List[Citation]:
        """Extract all citation brackets from answer and bind them to matching chunks."""
        matches = cls.CITATION_REGEX.finditer(answer_text)
        citations: List[Citation] = []
        seen_keys = set()

        for match in matches:
            page_num = int(match.group(1))
            section_hint = (match.group(2) or "").strip()
            raw = match.group(0)

            key = (page_num, section_hint.lower())
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Find matching chunk
            matched = cls._find_matching_chunk(page_num, section_hint, retrieved_chunks)
            snippet = ""
            if matched:
                snippet = cls._extract_best_snippet(matched.content)

            citations.append(
                Citation(
                    page_number=page_num,
                    section=section_hint or (matched.section if matched else "General"),
                    raw_citation=raw,
                    matched_chunk=matched,
                    excerpt_snippet=snippet,
                )
            )

        return citations

    @classmethod
    def _find_matching_chunk(
        cls,
        page_num: int,
        section_hint: str,
        retrieved_chunks: List[TextChunk]
    ) -> Optional[TextChunk]:
        """Find the most specific chunk matching the citation page and section."""
        # First priority: match both page and section
        if section_hint:
            sec_lower = section_hint.lower()
            for c in retrieved_chunks:
                if c.page_number == page_num and (sec_lower in c.section.lower() or sec_lower in c.canonical_section.lower()):
                    return c

        # Second priority: match page number
        for c in retrieved_chunks:
            if c.page_number == page_num:
                return c

        # Third priority: if section matches regardless of page
        if section_hint:
            sec_lower = section_hint.lower()
            for c in retrieved_chunks:
                if sec_lower in c.section.lower():
                    return c

        return None

    @classmethod
    def _extract_best_snippet(cls, content: str, max_chars: int = 220) -> str:
        """Get a concise preview snippet from chunk content."""
        clean = " ".join(content.split())
        if len(clean) <= max_chars:
            return clean
        # Find nearest word boundary
        cutoff = clean[:max_chars].rfind(" ")
        if cutoff > 100:
            return clean[:cutoff] + "..."
        return clean[:max_chars] + "..."
