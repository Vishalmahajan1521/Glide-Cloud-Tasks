import csv
import json
import re
import sys
import time

import requests

# Increase CSV field size limit to handle large patent text fields (Claims, Description, etc.)
# Default is 131072 (128KB), we'll set it to 10MB
csv.field_size_limit(min(2147483647, sys.maxsize))

API_URL = "http://127.0.0.1:8000/api/v1/ingest/from-text"
MAX_ROWS = 1000  # Only embed the first 200 rows
MAX_TEXT_LENGTH = 3000  # Overall text length cap per patent (chars) - reduced for faster embedding
MAX_RETRIES = 0  # No retries - fail fast to move on quickly
RETRY_DELAY = 2  # Seconds to wait before retry (if retries enabled)
HTTP_TIMEOUT = 20  # HTTP request timeout in seconds


def _get_field(row: dict, *candidates, default: str = "") -> str:
    """
    Helper to read a value from a CSV row in a case-insensitive way,
    trying multiple possible column names.
    """
    lower_map = {str(k).lower(): v for k, v in row.items()}
    for name in candidates:
        key = name.lower()
        if key in lower_map and lower_map[key] not in (None, ""):
            return lower_map[key]
    return default


def _parse_year(value: str, fallback: int = 2020) -> int:
    """
    Try to extract a 4‑digit year from an arbitrary string.
    """
    if not value:
        return fallback
    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return fallback
    return fallback


def _truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """
    Truncate text if it exceeds max_length, preserving word boundaries.
    """
    if not text or len(text) <= max_length:
        return text
    # Truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.9:  # If we found a space near the end
        return truncated[:last_space] + "..."
    return truncated + "..."


def _is_valid_text(text: str) -> bool:
    """
    Check if text is valid (not None, not empty, not just whitespace).
    """
    if not text:
        return False
    if isinstance(text, str):
        return text.strip() != ""
    return True


def ingest_single_patent(row, row_idx, topic=None):
    """
    Ingest a single patent with retry logic and better error handling.
    """
    try:
        # --- Build section‑aware text from ML + healthcare dataset ---
        title = _get_field(row, "title", "Title")
        abstract = _get_field(row, "abstract", "Abstract")
        claims = _get_field(row, "claims", "Claims", "claim_text")
        description = _get_field(
            row,
            "description",
            "Description",
            "detailed_description",
            "specification",
        )

        # Pre-check: Skip rows where raw text is extremely large (likely to timeout)
        total_raw_length = (
            len(str(abstract) if abstract else "") +
            len(str(description) if description else "") +
            len(str(claims) if claims else "")
        )
        if total_raw_length > 50000:  # If raw text is > 50k chars, likely too heavy
            print(f"[SKIP] Row {row_idx}: raw text too large ({total_raw_length} chars), likely to timeout")
            return None

        # Validate and truncate text fields.
        # Use mostly abstract; only very small snippets of description/claims to keep requests fast.
        sections = []
        if _is_valid_text(abstract):
            truncated_abstract = _truncate_text(str(abstract), max_length=2000)
            sections.append(f"Abstract\n{truncated_abstract}")
        if _is_valid_text(description):
            truncated_description = _truncate_text(str(description), max_length=500)
            sections.append(f"Description\n{truncated_description}")
        if _is_valid_text(claims):
            truncated_claims = _truncate_text(str(claims), max_length=500)
            sections.append(f"Claims\n{truncated_claims}")

        text = "\n\n".join(sections).strip()

        if not text or len(text) < 50:  # Minimum text length
            print(f"[SKIP] Row {row_idx}: text too short or empty (length: {len(text) if text else 0})")
            return None

        # Final safeguard: cap whole payload text
        text = _truncate_text(text, MAX_TEXT_LENGTH)
        
        # Log text size for debugging
        if len(text) > MAX_TEXT_LENGTH * 0.9:
            print(f"[WARN] Row {row_idx}: text near limit ({len(text)} chars)")

        # --- Metadata mapping (robust to different column names) ---
        raw_id = _get_field(
            row,
            "patent_id",
            "publication_number",
            "publication_id",
            "application_number",
            "id",
        )
        patent_id = (raw_id or title or f"row_{row_idx}").replace(" ", "").strip()
        
        if not patent_id or patent_id == f"row_{row_idx}":
            print(f"[SKIP] Row {row_idx}: no valid patent ID")
            return None

        assignee = _get_field(
            row,
            "assignee",
            "applicant",
            "applicant_name",
            "owner",
            "owners",
        )

        jurisdiction = _get_field(
            row,
            "jurisdiction",
            "country",
            "publication_country",
            default="US",
        )

        year_str = _get_field(
            row,
            "filing_year",
            "application_year",
            "application_date",
            "publication_year",
            "publication_date",
        )
        filing_year = _parse_year(year_str, fallback=2020)

        raw_classes = _get_field(
            row,
            "cpc",
            "cpc_codes",
            "ipc",
            "ipc_codes",
            "patent_class",
            "us_classifications",
        )
        patent_class = []
        if raw_classes:
            # Split on common delimiters ; or , and strip spaces
            parts = re.split(r"[;,]", str(raw_classes))
            patent_class = [p.strip() for p in parts if p.strip()]

        metadata = {
            "patent_id": patent_id,
            "title": title[:200] if title else "",  # Limit title length
            "assignee": assignee[:200] if assignee else "",  # Limit assignee length
            "jurisdiction": jurisdiction,
            "filing_year": filing_year,
            "patent_class": patent_class,
        }

        payload = {
            "text": text,
            "metadata": json.dumps(metadata),
        }
        if topic:
            payload["topic"] = topic

        # Retry logic
        for attempt in range(MAX_RETRIES + 1):
            try:
                # Short timeout so we fail fast on very heavy rows
                response = requests.post(API_URL, json=payload, timeout=HTTP_TIMEOUT)
                if response.status_code == 200:
                    result = response.json()
                    print(f"[OK] {patent_id[:50]} → {result.get('chunks_created', 0)} chunks")
                    return result
                else:
                    error_msg = response.text[:200]  # Limit error message length
                    if attempt < MAX_RETRIES:
                        print(f"[RETRY {attempt + 1}/{MAX_RETRIES}] {patent_id[:50]} → {response.status_code}: {error_msg}")
                        time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                    else:
                        print(f"[ERROR] {patent_id[:50]} → {response.status_code}: {error_msg}")
                        return None
            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES:
                    print(f"[TIMEOUT] {patent_id[:50]} → Retrying ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    print(f"[TIMEOUT] {patent_id[:50]} → Max retries exceeded, skipping")
                    return None
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES:
                    print(f"[ERROR] {patent_id[:50]} → {str(e)[:100]}, retrying...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    print(f"[ERROR] {patent_id[:50]} → {str(e)[:100]}")
                    return None
        
        return None
        
    except Exception as e:
        patent_id = row.get("patent_id") or row.get("id") or f"row_{row_idx}"
        print(f"[EXCEPTION] {str(patent_id)[:50]} → {str(e)[:200]}")
        return None


def ingest_patent_batch(rows, topic=None):
    """
    Process rows sequentially to avoid overwhelming the embedding service.
    """
    results = []
    for idx, row in enumerate(rows, 1):
        result = ingest_single_patent(row, idx, topic)
        if result:
            results.append(result)
        # Small delay between requests to avoid overwhelming the server
        # Reduced delay since we're already processing sequentially
        time.sleep(0.2)
    return results


def main():
    # Load from patent_analysis_data.csv (ML + healthcare dataset)
    patents = []
    topic = "ml_healthcare"  # Topic label for this dataset

    print(f"Loading up to {MAX_ROWS} rows from patent_analysis_data.csv...")
    
    with open(
        "data/patent_analysis_data.csv", newline="", encoding="utf-8"
    ) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i >= MAX_ROWS:
                break
            patents.append(row)

    print(f"Loaded {len(patents)} rows from patent_analysis_data.csv.")
    print(f"Processing sequentially to avoid timeouts...")
    print(f"Text limits: Abstract=2000, Description=500, Claims=500, Total={MAX_TEXT_LENGTH} chars")
    print(f"HTTP timeout: {HTTP_TIMEOUT}s, Retries: {MAX_RETRIES}")
    print("-" * 60)

    # Process sequentially to avoid overwhelming the embedding service
    results = ingest_patent_batch(patents, topic)
    
    successful = len([r for r in results if r])
    print("-" * 60)
    print(f"Completed: {successful}/{len(patents)} patents processed successfully")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total time: {time.time() - start_time:.2f} seconds")

