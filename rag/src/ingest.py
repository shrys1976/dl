from pathlib import Path
from typing import Literal

import pymupdf
from sentence_transformers import SentenceTransformer

from chunking import fixed_split, recursive_split, semantic_split

ChunkingStrategy = Literal["fixed", "recursive", "semantic"]


def extract_pdf(path: str | Path) -> list[dict]:
    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc):
            text = page.get_text("text").strip()

            pages.append({
                "source": pdf_path.name,
                "page": page_number + 1,
                "text": text,
            })

    return pages


def split_page_text(
    text: str,
    strategy: ChunkingStrategy,
    embed_model: SentenceTransformer | None = None,
) -> list[str]:
    if strategy == "fixed":
        return fixed_split(text)

    if strategy == "recursive":
        return recursive_split(text)

    if strategy == "semantic":
        if embed_model is None:
            raise ValueError("Semantic chunking requires an embedding model.")
        return semantic_split(text, embed_model)

    raise ValueError(f"Unknown chunking strategy: {strategy}")


def build_chunks(
    pages: list[dict],
    strategy: ChunkingStrategy = "fixed",
    embed_model: SentenceTransformer | None = None,
) -> list[dict]:
    chunks = []

    for page in pages:
        page_chunks = split_page_text(
            page["text"],
            strategy=strategy,
            embed_model=embed_model,
        )

        for chunk_id, chunk in enumerate(page_chunks):
            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "chunk_id": chunk_id,
                "chunk_strategy": strategy,
                "text": chunk,
            })

    return chunks


def load_all_chunks(
    data_dir: Path | str | None = None,
    *,
    strategy: ChunkingStrategy = "fixed",
    embed_model: SentenceTransformer | None = None,
) -> list[dict]:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"
    else:
        data_dir = Path(data_dir)

    if strategy == "semantic" and embed_model is None:
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        pages = extract_pdf(pdf_path)
        all_chunks.extend(
            build_chunks(pages, strategy=strategy, embed_model=embed_model)
        )

    return all_chunks


if __name__ == "__main__":
    chunks = load_all_chunks(strategy="fixed")
    print(len(chunks))
    print(chunks[0])

    for chunk in chunks[:3]:
        print("=" * 80)
        print("SOURCE:", chunk["source"])
        print("PAGE:", chunk["page"])
        print("CHUNK:", chunk["chunk_id"])
        print(chunk["text"][:500])
