import os
from fastapi import APIRouter

from ingestion.pdf_loader import load_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text

from embeddings.ollama_embedder import embed
from vectorstore.qdrant_store import QdrantStore

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.chunk_exporter import export_chunks_to_csv

router = APIRouter()


@router.post("/ingest")
def ingest_documents():

    all_vectors = []
    all_texts = []

    raw_docs_path = "data/raw_docs"

    for file_name in os.listdir(raw_docs_path):
        if not file_name.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(raw_docs_path, file_name)

        # 1. Load PDF
        raw_text = load_pdf(file_path)

        # 2. Clean text
        cleaned_text = clean_text(raw_text)

        # 3. Chunk text
        chunks = chunk_text(cleaned_text, CHUNK_SIZE, CHUNK_OVERLAP)

        # 4. Export chunks to CSV (NEW REQUIREMENT)
        export_chunks_to_csv(chunks, source_file=file_name)

        # 5. Generate embeddings
        for chunk in chunks:
            vector = embed(chunk)
            all_vectors.append(vector)
            all_texts.append((chunk, file_name))

    # Safety check
    if not all_vectors:
        return {"message": "No documents found to ingest."}

    # 6. Store in Qdrant
    store = QdrantStore(vector_size=len(all_vectors[0]))

    for vector, (chunk, source) in zip(all_vectors, all_texts):
        store.add_vector(
            vector=vector,
            text=chunk,
            metadata={
                "type": "document",
                "source": source
            }
        )

    return {
        "message": "Documents ingested successfully",
        "total_documents": len(set(source for _, source in all_texts)),
        "total_chunks": len(all_texts)
    }
