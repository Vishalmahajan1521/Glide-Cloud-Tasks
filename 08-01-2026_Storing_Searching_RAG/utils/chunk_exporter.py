import csv
import os
from datetime import datetime

def export_chunks_to_csv(chunks, source_file, output_dir="data/chunks"):
    """
    Exports text chunks to a CSV file for manual inspection.
    """

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir,
        f"chunks_{source_file}_{timestamp}.csv"
    )

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "source_file", "chunk_index", "chunk_text"])

        for idx, chunk in enumerate(chunks):
            writer.writerow([idx + 1, source_file, idx, chunk])

    return output_file
