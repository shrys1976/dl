import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import ChunkingStrategy, load_all_chunks
from retrieve import retrieve

GroundTruthMethod = Literal["keyword", "semantic"]
RECALL_KS = (1, 5, 10)

EVAL_QUERIES = [
    {
        "query": "Who is eligible for the iDEX Open Challenge?",
        "keywords": ["eligibility", "startups", "msmes"],
        "source": "doc1.pdf",
    },
    {
        "query": "What is the agentic software framework for DPDP?",
        "keywords": ["agentic", "data governance", "dpdp"],
        "source": "doc2.pdf",
    },
    {
        "query": "What tech stack is used for the DPDP compliance platform?",
        "keywords": ["next.js", "tailwind", "shadcn"],
        "source": "doc3.pdf",
    },
    {
        "query": "What is Privra's fund utilization plan?",
        "keywords": ["fund utilization", "genesis"],
        "source": "doc4.pdf",
    },
    {
        "query": "What is Privra?",
        "keywords": ["ai-native", "dpdp compliance", "vit"],
        "source": "doc5.pdf",
    },
]


def recall_at_k(
    retrieved: list[int],
    relevant: set[int],
    k: int,
) -> float:
    if not relevant:
        return 0.0

    return len(set(retrieved[:k]) & relevant) / len(relevant)


def find_relevant_indices_keyword(
    chunks: list[dict],
    keywords: list[str],
    *,
    source: str | None = None,
    match: str = "any",
) -> set[int]:
    relevant = set()
    lowered_keywords = [keyword.lower() for keyword in keywords]

    for index, chunk in enumerate(chunks):
        if source and chunk["source"] != source:
            continue

        text = chunk["text"].lower()
        matches = [keyword in text for keyword in lowered_keywords]

        if match == "all" and all(matches):
            relevant.add(index)
        elif match == "any" and any(matches):
            relevant.add(index)

    return relevant


def find_relevant_indices_semantic(
    query: str,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    *,
    source: str | None = None,
    similarity_threshold: float = 0.45,
    max_relevant: int = 5,
) -> set[int]:
    candidate_indices = [
        index
        for index, chunk in enumerate(chunks)
        if source is None or chunk["source"] == source
    ]

    if not candidate_indices:
        return set()

    candidate_texts = [chunks[index]["text"] for index in candidate_indices]
    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True,
    )
    chunk_embeddings = embed_model.encode(
        candidate_texts,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(query_embedding, dtype=np.float32)[0]
    chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)

    scores = chunk_embeddings @ query_embedding
    ranked = sorted(
        zip(candidate_indices, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    relevant = {
        index
        for index, score in ranked[:max_relevant]
        if score >= similarity_threshold
    }

    if not relevant and ranked:
        relevant = {ranked[0][0]}

    return relevant


def resolve_relevant_indices(
    item: dict,
    chunks: list[dict],
    *,
    ground_truth: GroundTruthMethod,
    embed_model: SentenceTransformer | None = None,
) -> set[int]:
    explicit = item.get("relevant_chunks")
    if explicit is not None:
        return set(explicit)

    if ground_truth == "keyword":
        return find_relevant_indices_keyword(
            chunks,
            item["keywords"],
            source=item.get("source"),
        )

    if embed_model is None:
        raise ValueError("Semantic ground truth requires an embedding model.")

    return find_relevant_indices_semantic(
        item["query"],
        chunks,
        embed_model,
        source=item.get("source"),
    )


def evaluate_retriever(
    eval_queries: list[dict],
    retrieve_fn: Callable[[str, int], list[dict]],
    chunks: list[dict],
    k: int = 5,
    *,
    ground_truth: GroundTruthMethod = "keyword",
    embed_model: SentenceTransformer | None = None,
) -> float:
    recalls = []

    for item in eval_queries:
        relevant = resolve_relevant_indices(
            item,
            chunks,
            ground_truth=ground_truth,
            embed_model=embed_model,
        )
        results = retrieve_fn(item["query"], k)
        retrieved = [result["index"] for result in results]
        recalls.append(recall_at_k(retrieved, relevant, k))

    return sum(recalls) / len(recalls)


def evaluate_retriever_at_ks(
    eval_queries: list[dict],
    retrieve_fn: Callable[[str, int], list[dict]],
    chunks: list[dict],
    ks: tuple[int, ...] = RECALL_KS,
    *,
    ground_truth: GroundTruthMethod = "keyword",
    embed_model: SentenceTransformer | None = None,
) -> dict[int, float]:
    max_k = max(ks)
    per_k_recalls = {k: [] for k in ks}

    for item in eval_queries:
        relevant = resolve_relevant_indices(
            item,
            chunks,
            ground_truth=ground_truth,
            embed_model=embed_model,
        )
        results = retrieve_fn(item["query"], max_k)
        retrieved = [result["index"] for result in results]

        for k in ks:
            per_k_recalls[k].append(recall_at_k(retrieved, relevant, k))

    return {
        k: sum(values) / len(values)
        for k, values in per_k_recalls.items()
    }


def measure_query_latency(
    retrieve_fn: Callable[[str, int], list[dict]],
    query: str,
    *,
    k: int = 5,
    runs: int = 5,
) -> float:
    for _ in range(runs):
        retrieve_fn(query, k)

    start = time.perf_counter()
    retrieve_fn(query, k)
    elapsed = time.perf_counter() - start

    return elapsed * 1000


def build_retrieve_fn(
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict],
) -> Callable[[str, int], list[dict]]:
    def retrieve_fn(query: str, k: int = 5) -> list[dict]:
        return retrieve(query, model, embeddings, chunks, k=k)

    return retrieve_fn


def build_embeddings(
    model: SentenceTransformer,
    chunks: list[dict],
) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    MODEL_NAME = "all-MiniLM-L6-v2"
    TOP_K = 5

    print("Loading PDFs...")
    all_chunks = load_all_chunks(DATA_DIR)
    print(f"Loaded {len(all_chunks)} chunks")

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = build_embeddings(model, all_chunks)
    retrieve_fn = build_retrieve_fn(model, embeddings, all_chunks)

    print(f"\nEvaluating retriever (Recall@{TOP_K})...")
    recall = evaluate_retriever(
        EVAL_QUERIES,
        retrieve_fn,
        all_chunks,
        k=TOP_K,
    )
    print(f"Recall@{TOP_K}: {recall:.2%}")

    sample_query = EVAL_QUERIES[0]["query"]
    latency_ms = measure_query_latency(
        retrieve_fn,
        sample_query,
        k=TOP_K,
    )
    print(f"Query latency: {latency_ms:.2f} ms")
