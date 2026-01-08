from embeddings.ollama_embedder import embed


def semantic_search(query: str, store, top_k: int):
    query_vector = embed(query)

    # Store query vector
    store.add_vector(
        vector=query_vector,
        text=query,
        metadata={"type": "query"}
    )

    results = store.search(query_vector, top_k)
    return query_vector, results
