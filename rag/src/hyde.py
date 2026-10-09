import os

import numpy as np
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

HYDE_PROMPT = """Write a short, technically detailed passage that could answer
the following question.

Focus on the likely concepts, terminology, and explanation.
Do not mention that the passage is hypothetical.

Question: {query}

Passage:
"""

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_INPUT_USD_PER_M = 0.59
GROQ_OUTPUT_USD_PER_M = 0.79


def create_hyde_llm(
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> ChatGroq:
    return ChatGroq(
        model=model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        temperature=temperature,
    )


def estimate_groq_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * GROQ_INPUT_USD_PER_M
        + (output_tokens / 1_000_000) * GROQ_OUTPUT_USD_PER_M
    )


def extract_token_usage(response) -> dict:
    metadata = getattr(response, "response_metadata", {}) or {}
    token_usage = metadata.get("token_usage", {})

    input_tokens = int(
        token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or 0
    )
    output_tokens = int(
        token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or 0
    )

    if input_tokens == 0 and output_tokens == 0:
        prompt_text = HYDE_PROMPT
        output_text = getattr(response, "content", "") or ""
        input_tokens = max(1, len(prompt_text) // 4)
        output_tokens = max(1, len(output_text) // 4)

    cost_usd = estimate_groq_cost(input_tokens, output_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }


def gen_hyde(query: str, llm: ChatGroq) -> tuple[str, dict]:
    response = llm.invoke(HYDE_PROMPT.format(query=query))
    passage = (response.content or "").strip()
    usage = extract_token_usage(response)
    return passage, usage


def retrieve_hyde(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict],
    llm: ChatGroq,
    k: int = 5,
) -> tuple[list[dict], dict]:
    hyp_doc, usage = gen_hyde(query, llm)

    hyde_embedding = model.encode(
        [hyp_doc],
        normalize_embeddings=True,
    )
    hyde_embedding = np.asarray(hyde_embedding, dtype=np.float32)

    scores = (hyde_embedding @ embeddings.T)[0]
    top_indices = np.argsort(-scores)[:k]

    results = []
    for idx in top_indices:
        results.append({
            "index": int(idx),
            "score": float(scores[idx]),
            "hyde_passage": hyp_doc,
            **chunks[idx],
        })

    return results, usage
