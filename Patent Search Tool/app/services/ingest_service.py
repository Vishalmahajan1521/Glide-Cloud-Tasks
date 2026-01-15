from app.ml.chunking import split_into_sections, create_chunks
from app.ml.embeddings import embedding_model
from app.retrieval.qdrant_store import QdrantStore
from app.core.exceptions import IngestionError


class IngestService:

    def __init__(self):
        self.vector_store = QdrantStore()

    def ingest_patent(
        self,
        pdf_path: str,
        metadata: dict,
        topic: str = None
    ) -> dict:
        try:
            # 1. Extract text
            text = extract_text_from_pdf(pdf_path)

            # 2. Split into sections
            sections = split_into_sections(text)

            # 3. Create chunks
            chunks = create_chunks(sections, text)

            if not chunks:
                raise IngestionError(
                    "PDF text could not be extracted. Possibly scanned or empty."
                )

            # 4. Generate embeddings
            texts = [c["text"] for c in chunks]
            embeddings = embedding_model.embed_documents(texts)

            # 5. Store in Qdrant
            metadata["topic"] = topic
            self.vector_store.upsert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )

            return {
                "status": "success",
                "patent_id": metadata.get("patent_id"),
                "chunks_created": len(chunks)
            }

        except Exception as e:
            raise IngestionError(str(e))


    def ingest_from_api(self, patent_id: str, topic: str = None) -> dict:
        from app.utils.patent_api_client import fetch_patent_data  # Import here to avoid circular
        try:
            # 1. Fetch from API
            patent_data = fetch_patent_data(patent_id)

            if not patent_data['text']:
                raise IngestionError("No text extracted from API.")

            # 2. Split into sections
            sections = split_into_sections(patent_data['text'])
            chunks = create_chunks(sections, patent_data['text'])

            # 3. Generate embeddings
            texts = [c["text"] for c in chunks]
            embeddings = embedding_model.embed_documents(texts)

            # 4. Store in Qdrant
            metadata = patent_data['metadata']
            metadata["topic"] = topic
            self.vector_store.upsert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )

            return {
                "status": "success",
                "patent_id": patent_id,
                "chunks_created": len(chunks)
            }
        except Exception as e:
            raise IngestionError(str(e))


    def ingest_from_text(self, text: str, metadata: dict, topic: str = None) -> dict:
        try:
            # 2. Split into sections
            sections = split_into_sections(text)
            chunks = create_chunks(sections, text)

            # Override chunk_type to "abstract" since we're only using title + abstract
            for chunk in chunks:
                chunk["chunk_type"] = "abstract"

            if not chunks:
                return {
                    "status": "skipped",
                    "patent_id": metadata.get("patent_id"),
                    "reason": "no chunks created"
                }

            # 3. Generate embeddings
            texts = [c["text"] for c in chunks]
            embeddings = embedding_model.embed_documents(texts)

            # 4. Store in Qdrant
            metadata["topic"] = topic
            self.vector_store.upsert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )

            return {
                "status": "success",
                "patent_id": metadata.get("patent_id"),
                "chunks_created": len(chunks)
            }
        except Exception as e:
            raise IngestionError(str(e))


ingest_service = IngestService()
