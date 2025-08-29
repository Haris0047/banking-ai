"""
Configuration settings for Vanna.AI custom implementation.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Main configuration settings."""
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4.1-nano", env="OPENAI_MODEL")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[str] = Field(None, env="COHERE_API_KEY")
    
    # Embedding Configuration
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(1536, env="EMBEDDING_DIMENSIONS")
    
    # Vector Database Configuration (Qdrant)
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    qdrant_collection_name: str = Field("vanna_training_data", env="QDRANT_COLLECTION_NAME")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    
    # Database Configuration
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_reload: bool = Field(True, env="API_RELOAD")
    
    # Streamlit Configuration
    streamlit_port: int = Field(8501, env="STREAMLIT_PORT")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("./logs/vanna.log", env="LOG_FILE")
    
    # Training Configuration
    max_training_examples: int = Field(1000, env="MAX_TRAINING_EXAMPLES")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")
    context_window_size: int = Field(4000, env="CONTEXT_WINDOW_SIZE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields to be ignored


# Global settings instance
settings = Settings()


def ensure_directories():
    """Ensure required directories exist."""
    directories = [
        os.path.dirname(settings.log_file),
        "./data",
        "./logs"
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True) 