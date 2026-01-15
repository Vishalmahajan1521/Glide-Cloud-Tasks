from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PayloadSchemaType
)
from app.core.config import settings
from uuid import uuid4
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, Range

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = "patent_chunks"

    def create_collection(self):
        collection_name = "patent_chunks"

        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if collection_name in existing:
            return

        self.client.recreate_collection(
            collection_name="patent_chunks",
            vectors_config=VectorParams(
                size=768,   # 🔴 MUST MATCH OLLAMA
                distance=Distance.COSINE
            )
        )

        # Payload indexes (VERY IMPORTANT)
        self.client.create_payload_index(
            collection_name,
            field_name="jurisdiction",
            field_schema=PayloadSchemaType.KEYWORD
        )

        self.client.create_payload_index(
            collection_name,
            field_name="assignee",
            field_schema=PayloadSchemaType.KEYWORD
        )

        self.client.create_payload_index(
            collection_name,
            field_name="filing_year",
            field_schema=PayloadSchemaType.INTEGER
        )

        self.client.create_payload_index(
            collection_name,
            field_name="patent_class",
            field_schema=PayloadSchemaType.KEYWORD
        )

        self.client.create_payload_index(
            collection_name,
            field_name="chunk_type",
            field_schema=PayloadSchemaType.KEYWORD
        )

        self.client.create_payload_index(
            collection_name,
            field_name="topic",
            field_schema=PayloadSchemaType.KEYWORD
        )
    
    def delete_collection(self):
        collection_name = "patent_chunks"
        try:
            self.client.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def upsert_chunks(
        self,
        chunks: list,
        embeddings: list,
        metadata: dict
    ):
        points = []

        for chunk, vector in zip(chunks, embeddings):
            payload = {
                **metadata,
                "text": chunk.get("text"),
                "chunk_type": chunk.get("chunk_type"),
                "section_priority": chunk.get("section_priority"),
                "claim_number": chunk.get("claim_number"),
                "chunk_index": chunk.get("chunk_index"),
            }

            points.append({
                "id": str(uuid4()),
                "vector": vector,
                "payload": payload
            })

        self.client.upsert(
            collection_name="patent_chunks",
            points=points
        )
    
    def search(self, query_vector, top_k, filters=None):
        qdrant_filter = None

        if filters:
            conditions = []

            if filters.jurisdiction:
                conditions.append(
                    FieldCondition(
                        key="jurisdiction",
                        match=MatchAny(any=filters.jurisdiction)
                    )
                )

            if filters.assignee:
                conditions.append(
                    FieldCondition(
                        key="assignee",
                        match=MatchAny(any=filters.assignee)
                    )
                )

            if filters.patent_class:
                conditions.append(
                    FieldCondition(
                        key="patent_class",
                        match=MatchAny(any=filters.patent_class)
                    )
                )

            if filters.filing_year_from or filters.filing_year_to:
                conditions.append(
                    FieldCondition(
                        key="filing_year",
                        range={
                            "gte": filters.filing_year_from or None,
                            "lte": filters.filing_year_to or None,
                        }
                    )
                )

            if filters.topic:
                conditions.append(
                    FieldCondition(
                        key="topic",
                        match=MatchValue(value=filters.topic)
                    )
                )

            if conditions:
                qdrant_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=0.0,
            query_filter=qdrant_filter
        )

        return results