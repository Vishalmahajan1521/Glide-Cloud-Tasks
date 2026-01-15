from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "Patent Search Backend"
    env: str = "development"

    # Qdrant
    qdrant_host: str
    qdrant_port: int

    # Ollama (Embeddings)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"

    class Config:
        env_file = ".env"


settings = Settings()
