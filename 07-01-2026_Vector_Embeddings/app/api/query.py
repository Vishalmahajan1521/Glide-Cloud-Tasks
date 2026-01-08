from fastapi import APIRouter
from embeddings.ollama_embedder import embed
from vectorstore.qdrant_store import QdrantStore
from app.config import TOP_K

router = APIRouter()


@router.post("/query")
def semantic_query(query: str):
    query_vector = embed(query)
    store = QdrantStore(vector_size=len(query_vector))

    results = store.search(query_vector, TOP_K)

    response = []
    for hit in results:
        response.append({
            "score": hit.score,
            "source": hit.payload.get("source"),
            "text": hit.payload.get("text")[:300],
            "vector_preview": hit.vector[:10]
        })

    return {
        "query_vector_preview": query_vector[:10],
        "dimension": len(query_vector),
        "results": response
    }
