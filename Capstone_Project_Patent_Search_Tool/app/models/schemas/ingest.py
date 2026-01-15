from pydantic import BaseModel
from typing import List


class PatentMetadata(BaseModel):
    patent_id: str
    title: str
    assignee: str
    jurisdiction: str
    filing_year: int
    patent_class: List[str]
