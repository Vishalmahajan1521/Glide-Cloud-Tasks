import os
from fastapi import APIRouter
from ingestion.pdf_loader import load_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from embeddings.ollama_embedder import embed
from vectorstore.qdrant_store import QdrantStore
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

router = APIRouter()


@router.post("/ingest")
def ingest_documents():
    vectors = []
    texts = []

    for file in os.listdir("data/raw_docs"):
        if file.endswith(".pdf"):
            raw = load_pdf(f"data/raw_docs/{file}")
            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)

            for chunk in chunks:
                vectors.append(embed(chunk))
                texts.append((chunk, file))

    store = QdrantStore(vector_size=len(vectors[0]))

    for vector, (chunk, source) in zip(vectors, texts):
        store.add_vector(
            vector=vector,
            text=chunk,
            metadata={"type": "document", "source": source}
        )

    return {"message": "Documents ingested successfully"}
