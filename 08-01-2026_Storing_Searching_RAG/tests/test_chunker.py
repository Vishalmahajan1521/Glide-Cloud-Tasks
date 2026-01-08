from ingestion.chunker import chunk_text

def test_chunking_basic():
    text = "This is a simple test text " * 50
    chunks = chunk_text(text, size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
