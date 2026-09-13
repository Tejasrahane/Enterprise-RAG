import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class ProjectSettings(BaseSettings):
    # Open-Source Local LLM Configuration (Ollama)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2:latest"
    
    # Open-Source Local Embeddings (SentenceTransformers / HuggingFace)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Free Open-Source Search Configuration
    SEARCH_ENGINE: str = "duckduckgo"
    
    # Optional Legacy/Cloud API Keys (Defaults to empty/mock fallback)
    OPENAI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    
    # Vector Database configurations 
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    DB_PERSIST_DIR: str = "./data/storage/chroma_db"
    DB_COLLECTION_NAME: str = "enterprise_rag_index"
    
    # Engine parameters
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 2.0
    RECURSION_LIMIT: int = 25
    MAX_REWRITE_LOOPS: int = 3
    
    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = ProjectSettings()
