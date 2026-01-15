import re
from typing import List, Dict

# ---------- Section Split ----------

SECTION_PATTERNS = {
    "abstract": r"\babstract\b",
    "claim": r"\bclaim[s]?\b",
    "description": r"\bdescription\b|\bdetailed description\b"
}


def split_into_sections(text: str) -> Dict[str, str]:
    text_lower = text.lower()
    sections = {}

    for section, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text_lower)
        if match:
            sections[section] = text[match.start():]

    return sections


# ---------- Chunk Creation ----------

def create_chunks(sections: Dict[str, str], full_text: str = "") -> List[Dict]:
    chunks = []

    # Section-aware chunking
    for section, content in sections.items():
        text_chunks = sliding_window_chunk(content)

        for idx, chunk in enumerate(text_chunks):
            chunks.append({
                "text": chunk,
                "chunk_type": section,
                "section_priority": section_weight(section),
                "chunk_index": idx
            })

    # 🔴 FALLBACK: no sections detected
    if not chunks:
        all_text = " ".join(sections.values()) if sections else full_text
        fallback_chunks = sliding_window_chunk(all_text)

        for idx, chunk in enumerate(fallback_chunks):
            chunks.append({
                "text": chunk,
                "chunk_type": "description",
                "section_priority": 0.3,
                "chunk_index": idx
            })

    return chunks


# ---------- Sliding Window ----------

def sliding_window_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    if not text.strip():
        return []

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap

    return chunks


def section_weight(section: str) -> float:
    return {
        "claim": 1.0,
        "abstract": 0.7,
        "description": 0.4
    }.get(section, 0.4)
