from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1.routes import ingest, search, health
from app.retrieval.qdrant_store import QdrantStore

setup_logging()

app = FastAPI(title=settings.app_name)

qdrant = QdrantStore()
qdrant.create_collection()

app.include_router(ingest.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
