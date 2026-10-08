from pathlib import Path
import pymupdf

# each page is stripped of its text and stored with
# additional data such as source, page, the actual text
# each page is stored as a seperate dict in the list
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



def chunk_text(
    text:str,
    chunk_size:int =1000,
    overlap:int =150
    ) -> list[str]:

    if overlap>=chunk_size:
        raise ValueError("Overlap must be less than chunk size")

    words = text.split() # splits words by spaces
    chunks = []
    start = 0

    while start<len(words):
        end = start+chunk_size

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        start = end-overlap

    return chunks



def build_chunks(pages:list[dict])->list[dict]:
    chunks = []

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for chunk_id, chunk in enumerate(page_chunks):
            chunks.append({
                "source":page["source"],
                "page":page["page"],
                "chunk_id": chunk_id,
                "text": chunk,
            })                    
   
    return chunks


def load_all_chunks(data_dir: Path | str | None = None) -> list[dict]:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"
    else:
        data_dir = Path(data_dir)

    all_chunks = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        pages = extract_pdf(pdf_path)
        all_chunks.extend(build_chunks(pages))

    return all_chunks


if __name__ == "__main__":
    chunks = load_all_chunks()
    print(len(chunks))
    print(chunks[0])

    for chunk in chunks[:3]:
        print("=" * 80)
        print("SOURCE:", chunk["source"])
        print("PAGE:", chunk["page"])
        print("CHUNK:", chunk["chunk_id"])
        print(chunk["text"][:500])