from app.retrieval.qdrant_store import QdrantStore

if __name__ == "__main__":
    store = QdrantStore()
    store.delete_collection()
    store.create_collection()
    print("Database cleared and recreated.")
