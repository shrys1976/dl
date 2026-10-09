import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from eval import (
    EVAL_QUERIES,
    build_embeddings,
    build_retrieve_fn,
    recall_at_k,
    resolve_relevant_indices,
)
from hyde import create_hyde_llm, retrieve_hyde
from ingest import load_all_chunks

MODEL_NAME = "all-MiniLM-L6-v2"
RECALL_KS = (5, 10)
WARMUP_RUNS = 2


@dataclass
class HydeBenchmarkResult:
    method: str
    recall_at_5: float
    recall_at_10: float
    query_latency_ms: float
    llm_generation_cost_usd: float
    relevant_passages_retrieved: float


def avg_relevant_passages(
    eval_queries: list[dict],
    retrieve_fn: Callable[[str, int], list[dict]],
    chunks: list[dict],
    embed_model: SentenceTransformer,
    *,
    k: int = 10,
) -> float:
    hits = []

    for item in eval_queries:
        relevant = resolve_relevant_indices(
            item,
            chunks,
            ground_truth="keyword",
            embed_model=embed_model,
        )
        results = retrieve_fn(item["query"], k)
        retrieved = [result["index"] for result in results]
        hits.append(len(set(retrieved[:k]) & relevant))

    return sum(hits) / len(hits)


def evaluate_recalls(
    eval_queries: list[dict],
    retrieve_fn: Callable[[str, int], list[dict]],
    chunks: list[dict],
    embed_model: SentenceTransformer,
) -> tuple[float, float]:
    recalls_at_5 = []
    recalls_at_10 = []

    for item in eval_queries:
        relevant = resolve_relevant_indices(
            item,
            chunks,
            ground_truth="keyword",
            embed_model=embed_model,
        )
        results = retrieve_fn(item["query"], max(RECALL_KS))
        retrieved = [result["index"] for result in results]

        recalls_at_5.append(recall_at_k(retrieved, relevant, 5))
        recalls_at_10.append(recall_at_k(retrieved, relevant, 10))

    return (
        sum(recalls_at_5) / len(recalls_at_5),
        sum(recalls_at_10) / len(recalls_at_10),
    )


def measure_average_query_latency(
    queries: list[str],
    retrieve_fn: Callable[[str, int], list[dict]],
    *,
    k: int = 10,
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


def benchmark_baseline(
    embed_model: SentenceTransformer,
    embeddings,
    chunks: list[dict],
    queries: list[str],
) -> HydeBenchmarkResult:
    retrieve_fn = build_retrieve_fn(embed_model, embeddings, chunks)

    recall_at_5, recall_at_10 = evaluate_recalls(
        EVAL_QUERIES,
        retrieve_fn,
        chunks,
        embed_model,
    )
    latency_ms = measure_average_query_latency(queries, retrieve_fn, k=10)
    relevant_passages = avg_relevant_passages(
        EVAL_QUERIES,
        retrieve_fn,
        chunks,
        embed_model,
        k=10,
    )

    return HydeBenchmarkResult(
        method="baseline",
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        query_latency_ms=latency_ms,
        llm_generation_cost_usd=0.0,
        relevant_passages_retrieved=relevant_passages,
    )


def benchmark_hyde(
    embed_model: SentenceTransformer,
    embeddings,
    chunks: list[dict],
    queries: list[str],
) -> HydeBenchmarkResult:
    llm = create_hyde_llm()
    total_cost_usd = 0.0
    recalls_at_5 = []
    recalls_at_10 = []
    relevant_hits = []

    for item in EVAL_QUERIES:
        relevant = resolve_relevant_indices(
            item,
            chunks,
            ground_truth="keyword",
            embed_model=embed_model,
        )
        results, usage = retrieve_hyde(
            item["query"],
            embed_model,
            embeddings,
            chunks,
            llm,
            k=max(RECALL_KS),
        )
        total_cost_usd += usage["cost_usd"]
        retrieved = [result["index"] for result in results]

        recalls_at_5.append(recall_at_k(retrieved, relevant, 5))
        recalls_at_10.append(recall_at_k(retrieved, relevant, 10))
        relevant_hits.append(len(set(retrieved[:10]) & relevant))

    def retrieve_fn(query: str, k: int = 10) -> list[dict]:
        results, _ = retrieve_hyde(
            query,
            embed_model,
            embeddings,
            chunks,
            llm,
            k=k,
        )
        return results

    latency_ms = measure_average_query_latency(queries, retrieve_fn, k=10)

    return HydeBenchmarkResult(
        method="hyde",
        recall_at_5=sum(recalls_at_5) / len(recalls_at_5),
        recall_at_10=sum(recalls_at_10) / len(recalls_at_10),
        query_latency_ms=latency_ms,
        llm_generation_cost_usd=total_cost_usd,
        relevant_passages_retrieved=sum(relevant_hits) / len(relevant_hits),
    )


def print_results(results: list[HydeBenchmarkResult]) -> None:
    headers = (
        "Method",
        "Recall@5",
        "Recall@10",
        "Query Latency (ms)",
        "LLM Generation Cost (USD)",
        "Relevant Passages Retrieved",
    )
    rows = [
        (
            result.method,
            f"{result.recall_at_5:.2%}",
            f"{result.recall_at_10:.2%}",
            f"{result.query_latency_ms:.2f}",
            f"{result.llm_generation_cost_usd:.6f}",
            f"{result.relevant_passages_retrieved:.2f}",
        )
        for result in results
    ]

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    header_line = " | ".join(
        headers[index].ljust(widths[index]) for index in range(len(headers))
    )
    separator = "-+-".join("-" * width for width in widths)

    print(header_line)
    print(separator)
    for row in rows:
        print(
            " | ".join(
                row[index].ljust(widths[index]) for index in range(len(headers))
            )
        )


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"
    queries = [item["query"] for item in EVAL_QUERIES]

    print("Loading chunks and embeddings...")
    chunks = load_all_chunks(data_dir)
    embed_model = SentenceTransformer(MODEL_NAME)
    embeddings = build_embeddings(embed_model, chunks)

    print("\nBenchmarking baseline retrieval...")
    baseline = benchmark_baseline(embed_model, embeddings, chunks, queries)

    print("Benchmarking HyDE retrieval (calls Groq)...")
    hyde = benchmark_hyde(embed_model, embeddings, chunks, queries)

    print("\nHyDE vs baseline:\n")
    print_results([baseline, hyde])
