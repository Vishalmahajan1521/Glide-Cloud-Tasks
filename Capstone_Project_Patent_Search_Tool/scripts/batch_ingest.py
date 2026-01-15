import csv
import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

API_URL = "http://127.0.0.1:8000/api/v1/ingest/from-text"
MAX_WORKERS = 4
BATCH_SIZE = 10

def ingest_patent_batch(rows, topic=None):
    results = []
    for row in rows:
        try:
            # Construct text and metadata from CSV (using Lens.org columns)
            text = f"{row.get('Title', '')} {row.get('Abstract', '')}".strip()
            if not text:
                print(f"[SKIP] No text for {row['Display Key']}")
                continue  # Skip patents with no text
            
            metadata = {
                "patent_id": row["Display Key"].replace(" ", ""),
                "title": row.get("Title", ""),
                "assignee": row.get("Applicants", "") or row.get("Owners", ""),
                "jurisdiction": row.get("Jurisdiction", "US"),
                "filing_year": int(row.get("Publication Year", 2020)),
                "patent_class": [],  # Can parse from US Classifications if needed
            }
            
            payload = {
                "text": text,
                "metadata": json.dumps(metadata)
            }
            if topic:
                payload["topic"] = topic
            
            response = requests.post(API_URL, json=payload, timeout=300)
            if response.status_code == 200:
                result = response.json()
                print(f"[OK] {metadata['patent_id']} → {result['chunks_created']} chunks")
                results.append(result)
            else:
                print(f"[ERROR] {metadata['patent_id']} → {response.text}")
        except Exception as e:
            patent_id = row.get("Display Key", "unknown").replace(" ", "")
            print(f"[ERROR] {patent_id} → {str(e)}")
    return results

def main():
    # Load from Patents.csv
    patents = []
    topic = "thermal_management"  # Set topic
    with open("data/Patents.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            patents.append(row)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(0, len(patents), BATCH_SIZE):
            batch = patents[i:i + BATCH_SIZE]
            futures.append(executor.submit(ingest_patent_batch, batch, topic))
        
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total time: {time.time() - start_time:.2f} seconds")
