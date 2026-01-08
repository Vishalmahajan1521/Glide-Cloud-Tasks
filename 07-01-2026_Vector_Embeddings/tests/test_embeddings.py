from embeddings.ollama_embedder import embed

def test_embedding_generation():
    text = "Vector embeddings represent semantic meaning"
    vector = embed(text)

    assert isinstance(vector, list)
    assert len(vector) > 0
