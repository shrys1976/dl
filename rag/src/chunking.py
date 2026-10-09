import re

import numpy as np
from sentence_transformers import SentenceTransformer


def apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for chunk in chunks[1:]:
        previous = overlapped[-1]
        prefix = previous[-overlap:] if len(previous) > overlap else previous
        overlapped.append(f"{prefix} {chunk}".strip())

    return overlapped


def fixed_split(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def recursive_split(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
    separators: list[str] | None = None,
) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    def split_with_separators(content: str, seps: list[str]) -> list[str]:
        if not content:
            return []

        if len(content) <= chunk_size:
            return [content]

        if not seps:
            return [
                content[index:index + chunk_size].strip()
                for index in range(0, len(content), chunk_size)
                if content[index:index + chunk_size].strip()
            ]

        separator = seps[0]
        remaining_separators = seps[1:]

        if separator == "":
            return split_with_separators(content, remaining_separators)

        if separator not in content:
            return split_with_separators(content, remaining_separators)

        parts = [part.strip() for part in content.split(separator) if part.strip()]
        merged: list[str] = []
        current = ""

        for part in parts:
            candidate = part if not current else f"{current}{separator}{part}"

            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                merged.append(current)

            if len(part) > chunk_size:
                merged.extend(split_with_separators(part, remaining_separators))
                current = ""
            else:
                current = part

        if current:
            merged.append(current)

        return merged

    chunks = split_with_separators(text, separators)
    return apply_overlap(chunks, overlap)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def semantic_split(
    text: str,
    embed_model: SentenceTransformer,
    *,
    max_chunk_size: int = 1000,
    similarity_threshold: float = 0.72,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = split_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    embeddings = embed_model.encode(
        sentences,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    chunks: list[str] = []
    current_sentences = [sentences[0]]
    current_vectors = [embeddings[0]]

    for index in range(1, len(sentences)):
        mean_vector = np.mean(current_vectors, axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm

        similarity = float(embeddings[index] @ mean_vector)
        candidate = " ".join(current_sentences + [sentences[index]])

        if similarity >= similarity_threshold and len(candidate) <= max_chunk_size:
            current_sentences.append(sentences[index])
            current_vectors.append(embeddings[index])
            continue

        chunks.append(" ".join(current_sentences))

        if len(sentences[index]) > max_chunk_size:
            chunks.extend(
                fixed_split(sentences[index], chunk_size=max_chunk_size, overlap=0)
            )
            current_sentences = []
            current_vectors = []
        else:
            current_sentences = [sentences[index]]
            current_vectors = [embeddings[index]]

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
