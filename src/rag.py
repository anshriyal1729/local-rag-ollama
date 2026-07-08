"""Fully offline/on-prem RAG pipeline: TF-IDF retrieval + a locally-running
Ollama model for generation (falls back to extractive answers if Ollama
isn't running, so the pipeline never hard-crashes on a missing model)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ollama_client import OllamaClient, OllamaUnavailableError


@dataclass
class Chunk:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_text(doc_id: str, text: str, chunk_size: int = 400, overlap: int = 50) -> List[Chunk]:
    words = text.split()
    chunks: List[Chunk] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(Chunk(doc_id=doc_id, text=" ".join(words[start:end])))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def _extractive_fallback(query: str, chunks: List[Chunk]) -> str:
    if not chunks:
        return "I couldn't find anything relevant in the local knowledge base."
    query_terms = set(re.findall(r"\w+", query.lower()))
    best_sentence, best_score = "", -1
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
            score = len(query_terms & set(re.findall(r"\w+", sentence.lower())))
            if score > best_score:
                best_score, best_sentence = score, sentence
    return best_sentence.strip() or chunks[0].text[:200]


class LocalRAGPipeline:
    """RAG pipeline designed for fully offline/on-prem deployment: TF-IDF
    retrieval (no external embedding API) + a locally-running Ollama model
    for generation. No document or query ever leaves the machine."""

    def __init__(self, ollama_model: str = "llama3", chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Chunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        self.ollama = OllamaClient(model=ollama_model)

    def ingest(self, documents: dict) -> None:
        for doc_id, text in documents.items():
            self.chunks.extend(chunk_text(doc_id, text, self.chunk_size, self.overlap))
        if self.chunks:
            self._matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 3) -> List[Chunk]:
        if self._matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]

    def answer(self, query: str, top_k: int = 3) -> dict:
        chunks = self.retrieve(query, top_k=top_k)
        context = "\n\n".join(c.text for c in chunks)

        used_ollama = False
        if chunks and self.ollama.is_available():
            prompt = (
                f"Answer the question using only the context below. "
                f"If the context doesn't contain the answer, say so.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
            try:
                answer_text = self.ollama.generate(prompt)
                used_ollama = True
            except OllamaUnavailableError:
                answer_text = _extractive_fallback(query, chunks)
        else:
            answer_text = _extractive_fallback(query, chunks)

        return {
            "query": query,
            "answer": answer_text,
            "generated_by": "ollama" if used_ollama else "extractive_fallback",
            "sources": [{"doc_id": c.doc_id, "text": c.text[:200]} for c in chunks],
        }
