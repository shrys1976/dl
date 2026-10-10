from typing import Literal

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

RerankerType = Literal["cross_encoder", "bi_encoder"]

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_BI_ENCODER = "all-MiniLM-L6-v2"


def create_cross_encoder(
    model_name: str = DEFAULT_CROSS_ENCODER,
) -> CrossEncoder:
    return CrossEncoder(model_name)


def create_bi_encoder(
    model_name: str = DEFAULT_BI_ENCODER,
) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    cross_encoder: CrossEncoder,
    k: int = 5,
) -> list[dict]:
    if not candidates:
        return []

    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:k]

    results = []
    for candidate, score in ranked:
        results.append({
            **candidate,
            "score": float(score),
            "rerank_score": float(score),
            "retrieval_score": candidate.get("score"),
            "reranker": "cross_encoder",
        })

    return results


def rerank_bi_encoder(
    query: str,
    candidates: list[dict],
    bi_encoder: SentenceTransformer,
    k: int = 5,
) -> list[dict]:
    if not candidates:
        return []

    query_embedding = bi_encoder.encode(
        [query],
        normalize_embeddings=True,
    )
    document_embeddings = bi_encoder.encode(
        [candidate["text"] for candidate in candidates],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(query_embedding, dtype=np.float32)[0]
    document_embeddings = np.asarray(document_embeddings, dtype=np.float32)
    scores = document_embeddings @ query_embedding

    top_indices = np.argsort(-scores)[:k]
    results = []

    for idx in top_indices:
        candidate = candidates[int(idx)]
        score = float(scores[idx])
        results.append({
            **candidate,
            "score": score,
            "rerank_score": score,
            "retrieval_score": candidate.get("score"),
            "reranker": "bi_encoder",
        })

    return results


def rerank(
    query: str,
    candidates: list[dict],
    *,
    reranker_type: RerankerType = "cross_encoder",
    k: int = 5,
    cross_encoder: CrossEncoder | None = None,
    bi_encoder: SentenceTransformer | None = None,
) -> list[dict]:
    if reranker_type == "cross_encoder":
        if cross_encoder is None:
            cross_encoder = create_cross_encoder()
        return rerank_cross_encoder(query, candidates, cross_encoder, k=k)

    if reranker_type == "bi_encoder":
        if bi_encoder is None:
            bi_encoder = create_bi_encoder()
        return rerank_bi_encoder(query, candidates, bi_encoder, k=k)

    raise ValueError(f"Unknown reranker type: {reranker_type}")


def retrieve_rerank(
    query: str,
    bi_encoder: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict],
    *,
    reranker_type: RerankerType = "cross_encoder",
    k: int = 5,
    candidate_k: int = 20,
    cross_encoder: CrossEncoder | None = None,
    rerank_bi_encoder_model: SentenceTransformer | None = None,
) -> list[dict]:
    from retrieve import retrieve

    candidate_k = max(candidate_k, k)
    candidates = retrieve(query, bi_encoder, embeddings, chunks, k=candidate_k)

    return rerank(
        query,
        candidates,
        reranker_type=reranker_type,
        k=k,
        cross_encoder=cross_encoder,
        bi_encoder=rerank_bi_encoder_model or bi_encoder,
    )
