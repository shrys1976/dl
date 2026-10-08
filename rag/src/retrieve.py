from sentence_transformers import SentenceTransformer
import numpy as np


def retrieve(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict],
    k: int = 5,
) -> list[dict]:
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    scores = (query_embedding @ embeddings.T)[0]
    top_indices = np.argsort(-scores)[:k]

    results = []
    for idx in top_indices:
        results.append({
            "index": int(idx),
            "score": float(scores[idx]),
            **chunks[idx],
        })

    return results


if __name__ == "__main__":
    from pathlib import Path

    from ingest import load_all_chunks

    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    MODEL_NAME = "all-MiniLM-L6-v2"
    TOP_K = 5

    print("Loading PDFs...")
    all_chunks = load_all_chunks(DATA_DIR)
    pdf_count = len(list(DATA_DIR.glob("*.pdf")))
    print(f"Loaded {len(all_chunks)} chunks from {pdf_count} PDFs")

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

    print("\nReady. Ask questions about the PDFs.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Question: ").strip()
        if not query or query.lower() in {"quit", "exit", "q"}:
            break

        results = retrieve(query, model, embeddings, all_chunks, k=TOP_K)

        for i, result in enumerate(results, start=1):
            print(f"\n--- Result {i} (score: {result['score']:.4f}) ---")
            print(
                f"{result['source']} | page {result['page']} "
                f"| chunk {result['chunk_id']}"
            )
            preview = result["text"][:500]
            print(preview)
            if len(result["text"]) > 500:
                print("...")

        print("\n" + "=" * 80 + "\n")
