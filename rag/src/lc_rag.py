import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ingest import load_all_chunks

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
all_chunks = load_all_chunks(DATA_DIR)

lc_docs = [
    Document(
        page_content=chunk["text"],
        metadata={
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
        },
    )
    for chunk in all_chunks
    if chunk["text"].strip()
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.from_documents(
    lc_docs,
    embedding_model,
    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5},
)


def format_docs(docs: list[Document]) -> str:
    sections = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        sections.append(
            f"[Source: {source}, page {page}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(sections)


prompt = ChatPromptTemplate.from_template("""
You are a question-answering assistant for the provided documents.

Answer using only the supplied context.
If the context does not contain enough information, say so.
Cite supporting material using the source and page labels shown
in the context. Never invent a citation.

Context:
{context}

Question:
{question}

Answer:
""")


if __name__ == "__main__":
    llm = ChatGroq(
        model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0.2,
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    question = input("Question: ").strip()
    if not question:
        raise SystemExit("Question cannot be empty.")

    answer = rag_chain.invoke(question)

    print(answer)

    retrieved_docs = retriever.invoke(question)

    for doc in retrieved_docs:
        print(doc.metadata)
        print(doc.page_content[:500])
        print("-" * 60)
