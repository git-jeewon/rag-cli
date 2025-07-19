"""Configuration management for RAG CLI."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "rag_cli")
    username: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "")
    
    @property
    def url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql+psycopg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))


@dataclass
class YouTubeConfig:
    """YouTube API configuration."""
    api_key: str = os.getenv("YOUTUBE_API_KEY", "")


@dataclass
class TwitterConfig:
    """Twitter API configuration."""
    bearer_token: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    api_key: str = os.getenv("TWITTER_API_KEY", "")
    api_secret: str = os.getenv("TWITTER_API_SECRET", "")
    access_token: str = os.getenv("TWITTER_ACCESS_TOKEN", "")
    access_token_secret: str = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")


@dataclass
class RedditConfig:
    """Reddit API configuration."""
    client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent: str = os.getenv("REDDIT_USER_AGENT", "RAG-CLI/1.0")


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    provider: str = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")  # or "openai"
    model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    max_length: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))


@dataclass
class FAISSConfig:
    """FAISS indexing configuration."""
    index_type: str = os.getenv("FAISS_INDEX_TYPE", "flat")  # flat, ivf, hnsw
    dimension: int = int(os.getenv("FAISS_DIMENSION", "384"))
    index_path: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")


@dataclass
class Config:
    """Main configuration class."""
    database: DatabaseConfig
    openai: OpenAIConfig
    youtube: YouTubeConfig
    twitter: TwitterConfig
    reddit: RedditConfig
    embedding: EmbeddingConfig
    faiss: FAISSConfig
    
    # General settings
    data_dir: str = os.getenv("DATA_DIR", "./data")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))


# Global configuration instance
config = Config(
    database=DatabaseConfig(),
    openai=OpenAIConfig(),
    youtube=YouTubeConfig(),
    twitter=TwitterConfig(),
    reddit=RedditConfig(),
    embedding=EmbeddingConfig(),
    faiss=FAISSConfig()
)


def validate_config() -> list[str]:
    """Validate configuration and return list of missing required settings."""
    errors = []
    
    # Check database connection
    if not config.database.password:
        errors.append("DB_PASSWORD is required")
    
    # Check embedding provider
    if config.embedding.provider == "openai" and not config.openai.api_key:
        errors.append("OPENAI_API_KEY is required when using OpenAI embeddings")
    
    return errors 