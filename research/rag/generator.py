"""
Context-grounded prompt engine and multi-provider LLM generator.
Supports Google Gemini, OpenAI, Groq, and a local extractive fallback with strict source citations.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Optional
from .vector_store import SearchResult
from .citation import CitationParser, Citation


GROUNDED_SYSTEM_PROMPT = """You are an expert academic research assistant analyzing a scientific paper.
Your task is to answer user queries with absolute factual precision, strictly grounded in the provided Context Passages.

Rules for your response:
1. STRICT GROUNDING: Answer ONLY based on the facts explicitly mentioned in the context. Do NOT extrapolate, hallucinate, or rely on external assumptions.
2. ABSENCE OF EVIDENCE: If the context does not provide information to answer a question (e.g., if limitations or datasets are omitted in the paper), state explicitly: "The provided sections of the paper do not mention [topic]."
3. SOURCE CITATIONS: You MUST cite the source page and section for EVERY substantive statement or finding using the exact format: `[Page X, Section Name]` or `[Page X]`. Example: "The authors evaluate the model on the SQuAD 2.0 dataset [Page 4, Experiments]."
4. ACADEMIC CLARITY: Use structured markdown with bullet points, bold key terms, and concise synthesis.
"""


@dataclass
class GenerationResult:
    question: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[SearchResult]
    provider: str
    model_name: str


class GroundedGenerator:
    """Generates context-grounded answers with citations from retrieved paper chunks."""

    def __init__(
        self,
        provider: str = "gemini",  # 'gemini', 'openai', 'groq', or 'local'
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")

        if not model_name:
            if self.provider == "gemini":
                self.model_name = "gemini-1.5-flash"
            elif self.provider == "openai":
                self.model_name = "gpt-4o-mini"
            elif self.provider == "groq":
                self.model_name = "llama-3.3-70b-versatile"
            else:
                self.model_name = "extractive-local"
        else:
            self.model_name = model_name

    def generate(
        self,
        question: str,
        retrieved_chunks: List[SearchResult],
        conversation_history: Optional[List[dict]] = None
    ) -> GenerationResult:
        """Generate a grounded answer for the user query using retrieved chunks."""
        if not retrieved_chunks:
            return GenerationResult(
                question=question,
                answer="No relevant sections were found in the paper for this question.",
                citations=[],
                retrieved_chunks=[],
                provider=self.provider,
                model_name=self.model_name,
            )

        # Build context block with provenance labels
        context_parts = []
        for idx, res in enumerate(retrieved_chunks, 1):
            c = res.chunk
            header = f"--- PASSAGE {idx} [Page {c.page_number} | Section: {c.section}] ---"
            context_parts.append(f"{header}\n{c.content}")

        context_text = "\n\n".join(context_parts)

        # Check if provider can execute
        raw_answer = None

        if self.provider == "gemini" and self.api_key:
            raw_answer = self._call_gemini(question, context_text, conversation_history)
        elif self.provider == "openai" and self.api_key:
            raw_answer = self._call_openai(question, context_text, conversation_history, base_url=None)
        elif self.provider == "groq" and self.api_key:
            raw_answer = self._call_openai(
                question, context_text, conversation_history, base_url="https://api.groq.com/openai/v1"
            )

        # Fallback to local extractive generator if no API key or provider call failed
        if raw_answer is None:
            raw_answer = self._local_extractive_answer(question, retrieved_chunks)
            actual_provider = "local-extractive"
            actual_model = "heuristic-summarizer"
        else:
            actual_provider = self.provider
            actual_model = self.model_name

        # Parse citations
        chunks_only = [r.chunk for r in retrieved_chunks]
        citations = CitationParser.extract_citations(raw_answer, chunks_only)

        # If model failed to add brackets but we have retrieved chunks, attach primary citation
        if not citations and retrieved_chunks:
            top_chunk = retrieved_chunks[0].chunk
            citations.append(
                Citation(
                    page_number=top_chunk.page_number,
                    section=top_chunk.section,
                    raw_citation=top_chunk.citation_label,
                    matched_chunk=top_chunk,
                    excerpt_snippet=CitationParser._extract_best_snippet(top_chunk.content),
                )
            )

        return GenerationResult(
            question=question,
            answer=raw_answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            provider=actual_provider,
            model_name=actual_model,
        )

    def _call_gemini(
        self, question: str, context: str, history: Optional[List[dict]]
    ) -> Optional[str]:
        """Invoke Gemini API."""
        try:
            # Try google-genai SDK first
            from google import genai
            client = genai.Client(api_key=self.api_key)
            prompt = f"{GROUNDED_SYSTEM_PROMPT}\n\nContext Passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text
        except Exception:
            # Try google.generativeai fallback
            try:
                import google.generativeai as gai
                gai.configure(api_key=self.api_key)
                model = gai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=GROUNDED_SYSTEM_PROMPT
                )
                prompt = f"Context Passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini API call failed: {e}")
                return None

    def _call_openai(
        self,
        question: str,
        context: str,
        history: Optional[List[dict]],
        base_url: Optional[str] = None
    ) -> Optional[str]:
        """Invoke OpenAI / Groq API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=base_url)
            messages = [
                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context Passages:\n{context}\n\nQuestion: {question}\n\nPlease provide a grounded answer with [Page X, Section Y] citations:"
                }
            ]
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"OpenAI/Groq API call failed: {e}")
            return None

    def _local_extractive_answer(self, question: str, retrieved: List[SearchResult]) -> str:
        """
        High-precision rule-based extractive synthesis for offline / zero-API-key operation.
        Extracts key sentences matching question intent and properly attaches citations.
        """
        q_lower = question.lower()
        extracted_points = []

        # Analyze question intent
        is_objective = any(w in q_lower for w in ["objective", "goal", "purpose", "aim"])
        is_method = any(w in q_lower for w in ["methodology", "method", "architecture", "approach", "how"])
        is_dataset = any(w in q_lower for w in ["dataset", "data", "benchmark", "corpus"])
        is_findings = any(w in q_lower for w in ["finding", "result", "performance", "achieve", "outperform"])
        is_limitations = any(w in q_lower for w in ["limitation", "weakness", "drawback", "future work", "threat"])

        for item in retrieved:
            chunk = item.chunk
            content = chunk.content
            # Split into clean sentences
            sentences = re.split(r"(?<=[.?!])\s+", content)

            for sent in sentences:
                s_clean = sent.strip()
                if len(s_clean) < 30:
                    continue
                s_lower = s_clean.lower()

                matches_intent = False
                if is_objective and any(k in s_lower for k in ["propose", "aim", "goal", "in this paper", "we introduce", "present", "objective"]):
                    matches_intent = True
                elif is_method and any(k in s_lower for k in ["method", "architecture", "algorithm", "we train", "pipeline", "framework", "layer", "module"]):
                    matches_intent = True
                elif is_dataset and any(k in s_lower for k in ["dataset", "evaluated on", "benchmark", "samples", "corpus", "data", "collected"]):
                    matches_intent = True
                elif is_findings and any(k in s_lower for k in ["result", "achieve", "improve", "outperform", "accuracy", "gain", "score", "finding", "show"]):
                    matches_intent = True
                elif is_limitations and any(k in s_lower for k in ["limit", "however", "challenge", "future work", "failure", "drawback", "restrict", "threat"]):
                    matches_intent = True
                elif not (is_objective or is_method or is_dataset or is_findings or is_limitations):
                    # General query
                    if any(word in s_lower for word in q_lower.split() if len(word) > 4):
                        matches_intent = True

                if matches_intent:
                    point = f"- {s_clean} [Page {chunk.page_number}, {chunk.section}]"
                    if point not in extracted_points:
                        extracted_points.append(point)
                    if len(extracted_points) >= 4:
                        break
            if len(extracted_points) >= 4:
                break

        if extracted_points:
            summary = "\n".join(extracted_points)
            return (
                f"### Analysis based on Retrieved Passages\n\n"
                f"{summary}\n\n"
                f"> *Note: Generated using local context extractor. Connect an API key (Gemini/OpenAI/Groq) in the sidebar for full conversational LLM generation.*"
            )
        else:
            # Fallback to top chunk excerpt
            top = retrieved[0].chunk
            return (
                f"### Relevant Passage Found\n\n"
                f"{top.content}\n\n"
                f"Source: [Page {top.page_number}, {top.section}]\n\n"
                f"> *Note: Connect an API key in the sidebar for conversational synthesis.*"
            )
