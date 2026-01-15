# app/ml/embeddings.py

import requests
import logging
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self):
        self.base_url = settings.ollama_url
        self.model = settings.ollama_model

        logger.info(
            f"Using Ollama embedding model '{self.model}' at {self.base_url}"
        )

    def _embed_single(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)


# ✅ SINGLE global instance
embedding_model = EmbeddingModel()
