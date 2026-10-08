import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import load_all_chunks
from retrieve import retrieve

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


def find_relevant_indices(
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


def evaluate_retriever(
    eval_queries: list[dict],
    retrieve_fn: Callable[[str, int], list[dict]],
    chunks: list[dict],
    k: int = 5,
) -> float:
    recalls = []

    for item in eval_queries:
        relevant = item.get("relevant_chunks")
        if relevant is None:
            relevant = find_relevant_indices(
                chunks,
                item["keywords"],
                source=item.get("source"),
            )

        results = retrieve_fn(item["query"], k)
        retrieved = [result["index"] for result in results]
        recalls.append(recall_at_k(retrieved, set(relevant), k))

    return sum(recalls) / len(recalls)


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


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    MODEL_NAME = "all-MiniLM-L6-v2"
    TOP_K = 5

    print("Loading PDFs...")
    all_chunks = load_all_chunks(DATA_DIR)
    print(f"Loaded {len(all_chunks)} chunks")

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Building embeddings...")
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

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

    print("\nPer-query breakdown:")
    for item in EVAL_QUERIES:
        relevant = find_relevant_indices(
            all_chunks,
            item["keywords"],
            source=item.get("source"),
        )
        results = retrieve_fn(item["query"], TOP_K)
        retrieved = [result["index"] for result in results]
        query_recall = recall_at_k(retrieved, relevant, TOP_K)

        print(f"\nQuery: {item['query']}")
        print(f"  Relevant chunks: {sorted(relevant)}")
        print(f"  Retrieved chunks: {retrieved}")
        print(f"  Recall@{TOP_K}: {query_recall:.2%}")
