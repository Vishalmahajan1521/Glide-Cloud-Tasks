from pydantic import BaseModel
from typing import List, Optional, Dict


class SearchFilters(BaseModel):
    jurisdiction: Optional[List[str]] = None
    assignee: Optional[List[str]] = None
    filing_year_from: Optional[int] = None
    filing_year_to: Optional[int] = None
    patent_class: Optional[List[str]] = None
    topic: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    filters: Optional[SearchFilters] = None


class SearchResult(BaseModel):
    patent_id: str
    score: float
    title: Optional[str] = None
    assignee: Optional[str] = None
    jurisdiction: Optional[str] = None

    matched_chunk_type: str
    explanation: str
