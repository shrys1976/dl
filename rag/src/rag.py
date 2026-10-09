import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import ChunkingStrategy, load_all_chunks
from retrieve import retrieve

RAG_PROMPT = """You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def format_context(results: list[dict]) -> str:
    blocks = []

    for index, result in enumerate(results, start=1):
        blocks.append(
            f"[{index}] source={result['source']}, page={result['page']}\n"
            f"{result['text']}"
        )

    return "\n\n".join(blocks)


def build_prompt(query: str, results: list[dict]) -> str:
    return RAG_PROMPT.format(
        context=format_context(results),
        question=query,
    )


def generate_answer(
    prompt: str,
    *,
    llm_model: str | None = None,
    api_key: str | None = None,
) -> str:
    model = llm_model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    api_key = api_key or os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export your Groq API key, e.g. "
            "`export GROQ_API_KEY='your-key-here'`"
        )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode("utf-8")

    request = urllib.request.Request(
        GROQ_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Groq API request failed ({exc.code}): {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Groq API. Details: {exc}"
        ) from exc

    try:
        answer = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Groq API response: {body}") from exc

    if not answer:
        raise RuntimeError("LLM returned an empty response.")

    return answer


def rag_answer(
    query: str,
    embed_model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict],
    *,
    k: int = 5,
    llm_model: str | None = None,
    api_key: str | None = None,
) -> dict:
    results = retrieve(query, embed_model, embeddings, chunks, k=k)
    prompt = build_prompt(query, results)
    answer = generate_answer(
        prompt,
        llm_model=llm_model,
        api_key=api_key,
    )

    return {
        "query": query,
        "prompt": prompt,
        "answer": answer,
        "sources": results,
    }


def load_rag_resources(
    data_dir: Path | None = None,
    embed_model_name: str = EMBED_MODEL_NAME,
    chunk_strategy: ChunkingStrategy | None = None,
) -> tuple[SentenceTransformer, np.ndarray, list[dict]]:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"

    if chunk_strategy is None:
        env_strategy = os.environ.get("CHUNK_STRATEGY", "fixed")
        chunk_strategy = (
            env_strategy
            if env_strategy in {"fixed", "recursive", "semantic"}
            else "fixed"
        )

    embed_model = SentenceTransformer(embed_model_name)
    chunks = load_all_chunks(
        data_dir,
        strategy=chunk_strategy,
        embed_model=embed_model,
    )

    from eval import build_embeddings

    embeddings = build_embeddings(embed_model, chunks)
    return embed_model, embeddings, chunks


if __name__ == "__main__":
    TOP_K = 5

    print("Loading PDFs and building embeddings...")
    embed_model, embeddings, chunks = load_rag_resources()
    print(f"Loaded {len(chunks)} chunks")

    llm_model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    print(f"\nReady. Using Groq model: {llm_model}")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Question: ").strip()
        if not query or query.lower() in {"quit", "exit", "q"}:
            break

        response = rag_answer(
            query,
            embed_model,
            embeddings,
            chunks,
            k=TOP_K,
            llm_model=llm_model,
        )

        print("\nAnswer:")
        print(response["answer"])

        print("\nSources:")
        for index, source in enumerate(response["sources"], start=1):
            print(
                f"  [{index}] {source['source']} | page {source['page']} "
                f"| score {source['score']:.4f}"
            )

        print("\n" + "=" * 80 + "\n")
