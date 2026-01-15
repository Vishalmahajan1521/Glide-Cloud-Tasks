from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Patent:
    patent_id: str
    title: str
    assignee: str
    jurisdiction: str
    filing_year: int
    patent_class: List[str]


@dataclass
class PatentChunk:
    patent_id: str
    text: str
    chunk_type: str
    section_priority: float
    claim_number: Optional[int] = None
    chunk_index: Optional[int] = None
