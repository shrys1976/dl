from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer

from eval import (
    EVAL_QUERIES,
    RECALL_KS,
    GroundTruthMethod,
    build_embeddings,
    build_retrieve_fn,
    evaluate_retriever_at_ks,
    measure_query_latency,
)
from ingest import ChunkingStrategy, load_all_chunks

STRATEGIES: tuple[ChunkingStrategy, ...] = ("fixed", "recursive", "semantic")
GROUND_TRUTHS: tuple[GroundTruthMethod, ...] = ("keyword", "semantic")
MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class ChunkBenchmarkResult:
    strategy: ChunkingStrategy
    ground_truth: GroundTruthMethod
    num_chunks: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    search_latency_ms: float


def benchmark_strategy(
    strategy: ChunkingStrategy,
    ground_truth: GroundTruthMethod,
    *,
    data_dir: Path,
    model: SentenceTransformer,
) -> ChunkBenchmarkResult:
    chunks = load_all_chunks(
        data_dir,
        strategy=strategy,
        embed_model=model,
    )
    embeddings = build_embeddings(model, chunks)
    retrieve_fn = build_retrieve_fn(model, embeddings, chunks)

    recalls = evaluate_retriever_at_ks(
        EVAL_QUERIES,
        retrieve_fn,
        chunks,
        ks=RECALL_KS,
        ground_truth=ground_truth,
        embed_model=model,
    )

    queries = [item["query"] for item in EVAL_QUERIES]
    latency_ms = measure_query_latency(
        retrieve_fn,
        queries[0],
        k=max(RECALL_KS),
    )

    return ChunkBenchmarkResult(
        strategy=strategy,
        ground_truth=ground_truth,
        num_chunks=len(chunks),
        recall_at_1=recalls[1],
        recall_at_5=recalls[5],
        recall_at_10=recalls[10],
        search_latency_ms=latency_ms,
    )


def print_results(results: list[ChunkBenchmarkResult]) -> None:
    headers = (
        "Strategy",
        "Ground Truth",
        "Chunks",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Latency (ms)",
    )
    rows = [
        (
            result.strategy,
            result.ground_truth,
            str(result.num_chunks),
            f"{result.recall_at_1:.2%}",
            f"{result.recall_at_5:.2%}",
            f"{result.recall_at_10:.2%}",
            f"{result.search_latency_ms:.2f}",
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


def run_chunk_benchmarks(
    data_dir: Path,
    model: SentenceTransformer,
) -> list[ChunkBenchmarkResult]:
    results = []

    for strategy in STRATEGIES:
        for ground_truth in GROUND_TRUTHS:
            print(f"Benchmarking strategy={strategy}, ground_truth={ground_truth}...")
            results.append(
                benchmark_strategy(
                    strategy,
                    ground_truth,
                    data_dir=data_dir,
                    model=model,
                )
            )

    return results


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\nComparing chunking strategies and ground-truth methods...\n")
    results = run_chunk_benchmarks(data_dir, model)
    print_results(results)
