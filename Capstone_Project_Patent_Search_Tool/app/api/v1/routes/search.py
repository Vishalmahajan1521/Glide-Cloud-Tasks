# app/api/v1/routes/search.py

from fastapi import APIRouter
from app.models.schemas.search import SearchRequest
from app.services.search_service import SearchService
from app.retrieval.qdrant_store import QdrantStore
from app.ml.embeddings import embedding_model  # ✅ import singleton

router = APIRouter()

vector_store = QdrantStore()

search_service = SearchService(
    vector_store=vector_store,
    embedder=embedding_model  # ✅ same instance everywhere
)

@router.post("/search")
def search_patents(request: SearchRequest):
    return search_service.search(request)
