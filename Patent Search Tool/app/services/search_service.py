from collections import defaultdict
from app.ml.embeddings import embedding_model
from app.retrieval.qdrant_store import QdrantStore
from app.models.schemas.search import SearchRequest
from app.core.exceptions import SearchError
from app.models.schemas.search import SearchFilters

SECTION_WEIGHTS = {
    "claim": 1.0,
    "abstract": 0.7,
    "description": 0.4
}

class SearchService:

    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def search(self, request):
        try:
            query_embedding = self.embedder.embed_query(request.query)

            filters = None
            if request.filters:
                filters = request.filters

            results = self.vector_store.search(
                query_vector=query_embedding,
                top_k=request.top_k,
                filters=filters
            )

            return [
                {
                    "score": hit.score,
                    "text": hit.payload.get("text"),
                    "patent_id": hit.payload.get("patent_id"),
                    "title": hit.payload.get("title"),
                    "assignee": hit.payload.get("assignee"),
                    "jurisdiction": hit.payload.get("jurisdiction"),
                    "filing_year": hit.payload.get("filing_year"),
                    "patent_class": hit.payload.get("patent_class"),
                    "chunk_type": hit.payload.get("chunk_type"),
                }
                for hit in results.points
            ]

        except Exception as e:
            raise SearchError(str(e))

    def _build_explanation(self, hit) -> str:
        """
        Build human-readable explanation for legal trust.
        """
        chunk_type = hit.payload.get("chunk_type", "description")
        similarity = round(hit.score, 4)

        if chunk_type == "claim":
            return f"High semantic similarity ({similarity}) with a patent claim."
        elif chunk_type == "abstract":
            return f"Matched the patent abstract with similarity {similarity}."
        else:
            return f"Matched the detailed description with similarity {similarity}."


# search_service = SearchService()
