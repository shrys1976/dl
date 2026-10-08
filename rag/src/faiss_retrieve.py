import time
from collections.abc import Callable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def ensure_float32_contiguous(embeddings: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def build_faiss_index(
    index_type: str,
    embeddings: np.ndarray,
    *,
    nlist: int | None = None,
    nprobe: int | None = None,
    hnsw_m: int = 32,
    hnsw_ef_search: int = 64,
    hnsw_ef_construction: int = 64,
) -> tuple[faiss.Index, float, dict]:
    vectors = ensure_float32_contiguous(embeddings)
    n_vectors, dim = vectors.shape
    start = time.perf_counter()
    params: dict = {}
    index_type = index_type.lower()

    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = hnsw_ef_construction
        index.hnsw.efSearch = hnsw_ef_search
        index.add(vectors)
        params = {
            "M": hnsw_m,
            "efConstruction": hnsw_ef_construction,
            "efSearch": hnsw_ef_search,
        }
    elif index_type == "ivf":
        if nlist is None:
            # Keep nlist small for tiny corpora to avoid under-trained clusters.
            nlist = min(4, max(1, n_vectors // 7))
        nlist = min(nlist, n_vectors)

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(
            quantizer,
            dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(vectors)
        index.add(vectors)

        if nprobe is None:
            nprobe = min(4, nlist)
        index.nprobe = nprobe
        params = {"nlist": nlist, "nprobe": nprobe}
    else:
        raise ValueError(
            f"Unknown index type: {index_type}. Use 'flat', 'hnsw', or 'ivf'."
        )

    build_time_ms = (time.perf_counter() - start) * 1000
    return index, build_time_ms, params


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )
    return ensure_float32_contiguous(query_embedding)


def search_faiss_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores, indices = index.search(query_embedding, k)
    return scores[0], indices[0]


def format_results(
    scores: np.ndarray,
    indices: np.ndarray,
    chunks: list[dict],
) -> list[dict]:
    results = []

    for score, idx in zip(scores, indices, strict=True):
        if idx < 0:
            continue

        results.append({
            "index": int(idx),
            "score": float(score),
            **chunks[int(idx)],
        })

    return results


def build_faiss_retrieve_fn(
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: list[dict],
) -> Callable[[str, int], list[dict]]:
    def retrieve_fn(query: str, k: int = 5) -> list[dict]:
        query_embedding = encode_query(model, query)
        scores, indices = search_faiss_index(index, query_embedding, k)
        return format_results(scores, indices, chunks)

    return retrieve_fn
