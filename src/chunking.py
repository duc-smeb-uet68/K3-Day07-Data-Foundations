from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        """Split ``text`` into groups of complete sentences.

        Sentence boundaries are detected after ``.``, ``!`` or ``?`` when
        followed by whitespace, and at newline boundaries.  Punctuation is
        kept in the resulting chunks so that the chunks remain readable.
        Empty or whitespace-only input produces no chunks.
        """
        if not text or not text.strip():
            return []

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
            if sentence.strip()
        ]

        return [
            " ".join(sentences[index : index + self.max_sentences_per_chunk])
            for index in range(0, len(sentences), self.max_sentences_per_chunk)
        ]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = max(1, chunk_size)

    def chunk(self, text: str) -> list[str]:
        """Return non-empty chunks while respecting the configured size when possible."""
        if not text or not text.strip():
            return []
        return [piece for piece in self._split(text.strip(), list(self.separators)) if piece]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        """Recursively split a string using the next available separator."""
        current_text = current_text.strip()
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            return self._hard_split(current_text)

        separator = remaining_separators[0]
        if not separator:
            return self._hard_split(current_text)
        if separator not in current_text:
            return self._split(current_text, remaining_separators[1:])

        parts = [part.strip() for part in current_text.split(separator) if part.strip()]
        if not parts:
            return self._split(current_text, remaining_separators[1:])

        chunks: list[str] = []
        buffer = ""
        next_separators = remaining_separators[1:]

        for part in parts:
            if len(part) > self.chunk_size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._split(part, next_separators))
                continue

            candidate = part if not buffer else f"{buffer}{separator}{part}"
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = part

        if buffer:
            chunks.append(buffer)
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Split by character boundaries as the final fallback."""
        return [
            text[start : start + self.chunk_size].strip()
            for start in range(0, len(text), self.chunk_size)
            if text[start : start + self.chunk_size].strip()
        ]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(sum(value * value for value in vec_a))
    norm_b = math.sqrt(sum(value * value for value in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            total_length = sum(len(piece) for piece in chunks)
            comparison[name] = {
                "count": len(chunks),
                "avg_length": total_length / len(chunks) if chunks else 0.0,
                "chunks": chunks,
            }
        return comparison
