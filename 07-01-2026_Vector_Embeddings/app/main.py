from fastapi import FastAPI
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router
from app.api.health import router as health_router

app = FastAPI(title="Vector Embedding & Retrieval API")

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)
