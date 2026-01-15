from fastapi import APIRouter
from app.services.ingest_service import ingest_service
import json
from pydantic import BaseModel

router = APIRouter()





from pydantic import BaseModel

class IngestTextRequest(BaseModel):
    text: str
    metadata: str
    topic: str = None

@router.post("/ingest/from-text")
def ingest_from_text(request: IngestTextRequest):
    try:
        metadata_dict = json.loads(request.metadata)
        result = ingest_service.ingest_from_text(request.text, metadata_dict, request.topic)
        return result
    except Exception as e:
        return {"error": str(e)}
