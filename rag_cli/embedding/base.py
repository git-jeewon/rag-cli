"""Base classes for text embedding."""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)


class EmbeddingProvider(Enum):
    """Available embedding providers."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    
    text: str
    vector: np.ndarray
    model_name: str
    provider: EmbeddingProvider
    metadata: Dict[str, Any]
    
    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return len(self.vector) if self.vector is not None else 0
    
    def to_list(self) -> List[float]:
        """Convert vector to list for storage."""
        return self.vector.tolist() if self.vector is not None else []


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding operation."""
    
    texts: List[str]
    vectors: np.ndarray  # Shape: (num_texts, embedding_dim)
    model_name: str
    provider: EmbeddingProvider
    metadata: Dict[str, Any]
    successful_count: int
    failed_count: int
    errors: List[str]
    
    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self.vectors.shape[1] if self.vectors is not None and len(self.vectors.shape) > 1 else 0
    
    @property
    def total_count(self) -> int:
        """Get total number of items processed."""
        return self.successful_count + self.failed_count
    
    def get_embedding(self, index: int) -> Optional[EmbeddingResult]:
        """Get individual embedding result by index."""
        if index >= self.successful_count or self.vectors is None:
            return None
        
        return EmbeddingResult(
            text=self.texts[index],
            vector=self.vectors[index],
            model_name=self.model_name,
            provider=self.provider,
            metadata=self.metadata
        )


class BaseEmbedder(ABC):
    """Base class for all text embedding implementations."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the embedder."""
        self.model_name = model_name
        self.config = kwargs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def get_provider(self) -> EmbeddingProvider:
        """Return the embedding provider type."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with vector and metadata
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> BatchEmbeddingResult:
        """
        Embed multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            batch_size: Optional batch size for processing
            
        Returns:
            BatchEmbeddingResult with vectors and metadata
        """
        pass
    
    def validate_text(self, text: str) -> str:
        """Validate and clean text before embedding."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Clean up text
        text = text.strip()
        
        # Handle very long texts (truncate if needed)
        max_length = self.config.get('max_length', 8000)
        if len(text) > max_length:
            self.logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
        
        return text
    
    def prepare_texts(self, texts: List[str]) -> List[str]:
        """Prepare and validate a batch of texts."""
        cleaned_texts = []
        for i, text in enumerate(texts):
            try:
                cleaned_text = self.validate_text(text)
                cleaned_texts.append(cleaned_text)
            except ValueError as e:
                self.logger.warning(f"Skipping text {i}: {e}")
                cleaned_texts.append("")  # Empty string as placeholder
        return cleaned_texts


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, EmbeddingResult] = {}
        self.max_size = max_size
    
    def _make_key(self, text: str, model_name: str) -> str:
        """Create cache key from text and model."""
        # Use hash to handle long texts
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{model_name}:{text_hash}"
    
    def get(self, text: str, model_name: str) -> Optional[EmbeddingResult]:
        """Get cached embedding."""
        key = self._make_key(text, model_name)
        return self.cache.get(key)
    
    def set(self, text: str, result: EmbeddingResult):
        """Cache embedding result."""
        key = self._make_key(text, result.model_name)
        
        # Simple LRU: remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = result
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return np.linalg.norm(vec1 - vec2)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm 