import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue
)
from app.config import QDRANT_URL, COLLECTION_NAME


class QdrantStore:
    def __init__(self, vector_size: int):
        self.client = QdrantClient(url=QDRANT_URL)

        collections = self.client.get_collections().collections
        if COLLECTION_NAME not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    def add_vector(self, vector, text, metadata):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": text,
                **metadata
            }
        )
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

    def search(self, query_vector, top_k: int):
        result = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchValue(value="document")
                    )
                ]
            )
        )
        return result.points
