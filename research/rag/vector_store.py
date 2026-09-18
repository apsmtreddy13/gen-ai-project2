"""
Dense semantic vector indexing and retrieval for academic paper chunks.
Supports ChromaDB with FastEmbed / ONNX embeddings and pure-Python cosine fallback.
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .chunker import TextChunk

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None


@dataclass
class SearchResult:
    chunk: TextChunk
    score: float  # Normalized similarity score (0.0 to 1.0, higher is better)

    @property
    def citation_str(self) -> str:
        return f"[Page {self.chunk.page_number}, {self.chunk.section}]"


class SimpleBM25Fallback:
    """Lightweight in-memory BM25 + term-vector retriever for offline or zero-dependency scenarios."""

    def __init__(self):
        self.chunks: List[TextChunk] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freq: Dict[str, int] = {}
        self.avg_dl: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())

    def index(self, chunks: List[TextChunk]):
        self.chunks = chunks
        self.doc_tokens = [self._tokenize(c.content + " " + c.section) for c in chunks]
        self.doc_freq = {}
        total_len = 0
        for tokens in self.doc_tokens:
            total_len += len(tokens)
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        self.avg_dl = total_len / max(1, len(chunks))

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        q_tokens = self._tokenize(query)
        if not q_tokens or not self.chunks:
            return []

        k1 = 1.5
        b = 0.75
        N = len(self.chunks)
        scores: List[float] = []

        for idx, tokens in enumerate(self.doc_tokens):
            doc_len = len(tokens)
            score = 0.0
            token_counts: Dict[str, int] = {}
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            for qt in q_tokens:
                if qt in token_counts:
                    freq = token_counts[qt]
                    n_q = self.doc_freq.get(qt, 1)
                    idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
                    numerator = freq * (k1 + 1)
                    denominator = freq + k1 * (1 - b + b * (doc_len / max(1.0, self.avg_dl)))
                    score += idf * (numerator / denominator)

            scores.append(score)

        max_score = max(scores) if scores and max(scores) > 0 else 1.0
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in ranked_indices:
            norm_score = min(1.0, round(scores[i] / max_score, 4))
            results.append(SearchResult(chunk=self.chunks[i], score=norm_score))
        return results


class VectorIndex:
    """Vector database manager supporting ChromaDB, FastEmbed, and in-memory search."""

    def __init__(
        self,
        collection_name: str = "research_paper_rag",
        persist_dir: Optional[str] = None,
        use_fastembed: bool = True,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.chunks: List[TextChunk] = []
        self.chunk_map: Dict[str, TextChunk] = {}
        self.fallback = SimpleBM25Fallback()
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None

        # Try initializing FastEmbed
        if use_fastembed and TextEmbedding is not None:
            try:
                self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            except Exception:
                self.embedding_model = None

        # Try initializing ChromaDB
        if chromadb is not None:
            try:
                if persist_dir:
                    self.chroma_client = chromadb.PersistentClient(path=persist_dir)
                else:
                    self.chroma_client = chromadb.Client()
                
                # Delete existing collection if re-indexing
                try:
                    self.chroma_client.delete_collection(name=self.collection_name)
                except Exception:
                    pass

                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                self.collection = None

    def index_chunks(self, chunks: List[TextChunk]) -> int:
        """Index a list of chunks into the vector store."""
        if not chunks:
            return 0

        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.fallback.index(chunks)

        if self.collection is not None:
            ids = [c.chunk_id for c in chunks]
            # Context-rich text for embedding: prepend section and page header
            doc_texts = [
                f"[Section: {c.section} | Page {c.page_number}]\n{c.content}"
                for c in chunks
            ]
            metadatas = [
                {
                    "chunk_id": c.chunk_id,
                    "paper_title": c.paper_title,
                    "page_number": c.page_number,
                    "section": c.section,
                    "canonical_section": c.canonical_section,
                }
                for c in chunks
            ]

            if self.embedding_model is not None:
                # Use FastEmbed dense vectors
                embeddings = list(self.embedding_model.embed(doc_texts))
                embeddings_list = [emb.tolist() for emb in embeddings]
                self.collection.add(
                    ids=ids,
                    documents=doc_texts,
                    metadatas=metadatas,
                    embeddings=embeddings_list
                )
            else:
                # Use Chroma's default embedding function
                self.collection.add(
                    ids=ids,
                    documents=doc_texts,
                    metadatas=metadatas
                )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_section: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform semantic retrieval for a user question."""
        if not self.chunks:
            return []

        # Target section keyword boosts for academic queries
        canonical_boost_map = {
            "objective": ["overview", "abstract", "introduction"],
            "methodology": ["methodology", "architecture"],
            "dataset": ["experiments & datasets"],
            "finding": ["results & findings", "experiments & datasets"],
            "limitation": ["discussion & limitations", "conclusion"],
        }

        query_lower = query.lower()
        boost_targets = []
        for kw, targets in canonical_boost_map.items():
            if kw in query_lower:
                boost_targets.extend(targets)

        # 1. ChromaDB retrieval if available
        if self.collection is not None and len(self.chunks) > 0:
            try:
                where_clause = None
                if filter_section:
                    where_clause = {"canonical_section": filter_section}

                fetch_k = min(len(self.chunks), top_k * 2)

                if self.embedding_model is not None:
                    query_embedding = list(self.embedding_model.embed([query]))[0].tolist()
                    res = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=fetch_k,
                        where=where_clause
                    )
                else:
                    res = self.collection.query(
                        query_texts=[query],
                        n_results=fetch_k,
                        where=where_clause
                    )

                results: List[SearchResult] = []
                retrieved_ids = res["ids"][0] if res["ids"] else []
                retrieved_distances = res["distances"][0] if res.get("distances") else [0.5] * len(retrieved_ids)

                for chunk_id, dist in zip(retrieved_ids, retrieved_distances):
                    chunk = self.chunk_map.get(chunk_id)
                    if not chunk:
                        continue
                    # Cosine distance to similarity: similarity = 1 - distance
                    sim = max(0.0, min(1.0, 1.0 - dist))
                    # Apply semantic section boost
                    if any(t in chunk.canonical_section.lower() for t in boost_targets):
                        sim = min(1.0, sim * 1.15)
                    results.append(SearchResult(chunk=chunk, score=round(sim, 4)))

                # Sort by score descending and take top_k
                results.sort(key=lambda x: x.score, reverse=True)
                return results[:top_k]
            except Exception:
                pass

        # 2. Fallback to BM25 retriever
        fallback_results = self.fallback.search(query, top_k=top_k)
        for r in fallback_results:
            if any(t in r.chunk.canonical_section.lower() for t in boost_targets):
                r.score = min(1.0, round(r.score * 1.15, 4))
        fallback_results.sort(key=lambda x: x.score, reverse=True)
        return fallback_results[:top_k]
