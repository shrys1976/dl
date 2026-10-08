import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from eval import EVAL_QUERIES, evaluate_retriever
from faiss_retrieve import build_faiss_index, build_faiss_retrieve_fn
from ingest import load_all_chunks
from retrieve import retrieve

INDEX_TYPES = ("flat", "hnsw", "ivf")
TOP_K = 5
WARMUP_RUNS = 3


@dataclass
class BenchmarkResult:
    name: str
    build_time_ms: float
    recall_at_5: float
    search_latency_ms: float
    params: dict


def build_numpy_retrieve_fn(
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict],
) -> Callable[[str, int], list[dict]]:
    def retrieve_fn(query: str, k: int = 5) -> list[dict]:
        return retrieve(query, model, embeddings, chunks, k=k)

    return retrieve_fn


def measure_search_latency(
    retrieve_fn: Callable[[str, int], list[dict]],
    queries: list[str],
    *,
    k: int = 5,
    warmup_runs: int = WARMUP_RUNS,
) -> float:
    for _ in range(warmup_runs):
        for query in queries:
            retrieve_fn(query, k)

    start = time.perf_counter()
    for query in queries:
        retrieve_fn(query, k)
    elapsed = time.perf_counter() - start

    return (elapsed / len(queries)) * 1000


def benchmark_numpy(
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict],
    queries: list[str],
    k: int,
) -> BenchmarkResult:
    start = time.perf_counter()
    retrieve_fn = build_numpy_retrieve_fn(model, embeddings, chunks)
    build_time_ms = (time.perf_counter() - start) * 1000

    recall = evaluate_retriever(EVAL_QUERIES, retrieve_fn, chunks, k=k)
    latency_ms = measure_search_latency(retrieve_fn, queries, k=k)

    return BenchmarkResult(
        name="numpy",
        build_time_ms=build_time_ms,
        recall_at_5=recall,
        search_latency_ms=latency_ms,
        params={},
    )


def benchmark_faiss_index(
    index_type: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict],
    queries: list[str],
    k: int,
) -> BenchmarkResult:
    index, build_time_ms, params = build_faiss_index(index_type, embeddings)
    retrieve_fn = build_faiss_retrieve_fn(model, index, chunks)

    recall = evaluate_retriever(EVAL_QUERIES, retrieve_fn, chunks, k=k)
    latency_ms = measure_search_latency(retrieve_fn, queries, k=k)

    return BenchmarkResult(
        name=f"faiss_{index_type}",
        build_time_ms=build_time_ms,
        recall_at_5=recall,
        search_latency_ms=latency_ms,
        params=params,
    )


def print_results(results: list[BenchmarkResult], k: int) -> None:
    headers = ("Index", f"Recall@{k}", "Search Latency (ms)", "Build Time (ms)")
    rows = [
        (
            result.name,
            f"{result.recall_at_5:.2%}",
            f"{result.search_latency_ms:.2f}",
            f"{result.build_time_ms:.2f}",
        )
        for result in results
    ]

    widths = [
        max(len(header), *(len(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ]

    header_line = " | ".join(
        header.ljust(widths[i]) for i, header in enumerate(headers)
    )
    separator = "-+-".join("-" * width for width in widths)

    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))

    print("\nIndex parameters:")
    for result in results:
        if result.params:
            print(f"  {result.name}: {result.params}")


def run_benchmarks(
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict],
    *,
    k: int = TOP_K,
) -> list[BenchmarkResult]:
    queries = [item["query"] for item in EVAL_QUERIES]
    results = [benchmark_numpy(model, embeddings, chunks, queries, k)]

    for index_type in INDEX_TYPES:
        results.append(
            benchmark_faiss_index(
                index_type,
                model,
                embeddings,
                chunks,
                queries,
                k,
            )
        )

    return results


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    MODEL_NAME = "all-MiniLM-L6-v2"

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

    print(f"\nBenchmarking retrievers (Recall@{TOP_K})...\n")
    results = run_benchmarks(model, embeddings, all_chunks, k=TOP_K)
    print_results(results, TOP_K)
