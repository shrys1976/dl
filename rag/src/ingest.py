from pathlib import Path
import pymupdf

def extract_pdf(path: str) -> list[dict]:
    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc):
            text = page.get_text("text").strip()

            pages.append({

                "source":pdf_path.name,
                "page":page_number+1,
                "text":text,
            })   

    return pages        



pages = extract_pdf("rag/data/doc1.pdf")
print(len(pages))
print(pages[0])
