"""SentenceTransformers embedding implementation."""

import logging
import numpy as np
from typing import List, Optional, Dict, Any
import time

from rag_cli.embedding.base import (
    BaseEmbedder, EmbeddingResult, BatchEmbeddingResult, 
    EmbeddingProvider, EmbeddingCache
)

logger = logging.getLogger(__name__)


class SentenceTransformersEmbedder(BaseEmbedder):
    """SentenceTransformers-based text embedder."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2", 
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None,
                 use_cache: bool = True,
                 **kwargs):
        """
        Initialize SentenceTransformers embedder.
        
        Args:
            model_name: Name of the SentenceTransformers model
            device: Device to use ('cpu', 'cuda', 'mps' for Apple Silicon)
            cache_folder: Folder to cache models
            use_cache: Whether to use in-memory caching
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        
        self.device = device
        self.cache_folder = cache_folder
        self.model = None
        self._dimension = None
        
        # Set up caching
        self.use_cache = use_cache
        if use_cache:
            cache_size = kwargs.get('cache_size', 1000)
            self.cache = EmbeddingCache(max_size=cache_size)
        else:
            self.cache = None
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "SentenceTransformers not installed. Run: pip install sentence-transformers"
            )
        
        try:
            self.logger.info(f"Loading SentenceTransformers model: {self.model_name}")
            start_time = time.time()
            
            # Auto-detect device if not specified
            if self.device is None:
                self.device = self._auto_detect_device()
            
            # Load model
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_folder
            )
            
            # Get embedding dimension
            self._dimension = self.model.get_sentence_embedding_dimension()
            
            load_time = time.time() - start_time
            self.logger.info(
                f"Model loaded successfully in {load_time:.2f}s "
                f"(device: {self.device}, dimension: {self._dimension})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformers model {self.model_name}: {e}")
            raise
    
    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device."""
        try:
            import torch
            
            # Check for Apple Silicon MPS
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            
            # Check for CUDA
            if torch.cuda.is_available():
                return 'cuda'
            
            # Fall back to CPU
            return 'cpu'
            
        except ImportError:
            self.logger.warning("PyTorch not available, using CPU")
            return 'cpu'
    
    def get_provider(self) -> EmbeddingProvider:
        """Return the embedding provider type."""
        return EmbeddingProvider.SENTENCE_TRANSFORMERS
    
    def get_dimension(self) -> int:
        """Return the dimension of embeddings."""
        if self._dimension is None:
            if self.model is not None:
                self._dimension = self.model.get_sentence_embedding_dimension()
            else:
                # Default dimensions for common models
                model_dimensions = {
                    'all-MiniLM-L6-v2': 384,
                    'all-MiniLM-L12-v2': 384,
                    'all-mpnet-base-v2': 768,
                    'all-distilroberta-v1': 768,
                }
                self._dimension = model_dimensions.get(self.model_name, 384)
        
        return self._dimension
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text string."""
        # Validate and clean text
        text = self.validate_text(text)
        
        # Check cache
        if self.cache is not None:
            cached_result = self.cache.get(text, self.model_name)
            if cached_result is not None:
                self.logger.debug("Using cached embedding")
                return cached_result
        
        # Generate embedding
        start_time = time.time()
        vector = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        embedding_time = time.time() - start_time
        
        # Create result
        result = EmbeddingResult(
            text=text,
            vector=vector,
            model_name=self.model_name,
            provider=self.get_provider(),
            metadata={
                'device': self.device,
                'embedding_time': embedding_time,
                'text_length': len(text),
                'dimension': len(vector)
            }
        )
        
        # Cache result
        if self.cache is not None:
            self.cache.set(text, result)
        
        self.logger.debug(f"Generated embedding in {embedding_time:.3f}s")
        return result
    
    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> BatchEmbeddingResult:
        """Embed multiple texts in batch."""
        if not texts:
            return BatchEmbeddingResult(
                texts=[],
                vectors=np.array([]),
                model_name=self.model_name,
                provider=self.get_provider(),
                metadata={},
                successful_count=0,
                failed_count=0,
                errors=[]
            )
        
        # Prepare texts
        cleaned_texts = self.prepare_texts(texts)
        batch_size = batch_size or self.config.get('batch_size', 32)
        
        # Check cache for existing embeddings
        cached_results = []
        texts_to_embed = []
        indices_to_embed = []
        
        if self.cache is not None:
            for i, text in enumerate(cleaned_texts):
                if text:  # Skip empty texts
                    cached_result = self.cache.get(text, self.model_name)
                    if cached_result is not None:
                        cached_results.append((i, cached_result.vector))
                    else:
                        texts_to_embed.append(text)
                        indices_to_embed.append(i)
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        else:
            texts_to_embed = [t for t in cleaned_texts if t]
            indices_to_embed = [i for i, t in enumerate(cleaned_texts) if t]
        
        # Generate embeddings for non-cached texts
        all_vectors = {}
        errors = []
        successful_count = 0
        
        # Add cached results
        for idx, vector in cached_results:
            all_vectors[idx] = vector
            successful_count += 1
        
        if texts_to_embed:
            try:
                start_time = time.time()
                
                # Process in batches
                for i in range(0, len(texts_to_embed), batch_size):
                    batch_texts = texts_to_embed[i:i + batch_size]
                    batch_indices = indices_to_embed[i:i + batch_size]
                    
                    try:
                        batch_vectors = self.model.encode(
                            batch_texts,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            batch_size=len(batch_texts)
                        )
                        
                        # Store results and cache
                        for j, (text, idx, vector) in enumerate(zip(batch_texts, batch_indices, batch_vectors)):
                            all_vectors[idx] = vector
                            successful_count += 1
                            
                            # Cache individual results
                            if self.cache is not None and text:
                                result = EmbeddingResult(
                                    text=text,
                                    vector=vector,
                                    model_name=self.model_name,
                                    provider=self.get_provider(),
                                    metadata={}
                                )
                                self.cache.set(text, result)
                    
                    except Exception as e:
                        error_msg = f"Batch {i//batch_size + 1} failed: {e}"
                        errors.append(error_msg)
                        self.logger.warning(error_msg)
                
                embedding_time = time.time() - start_time
                self.logger.info(f"Generated {successful_count} embeddings in {embedding_time:.2f}s")
                
            except Exception as e:
                error_msg = f"Batch embedding failed: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        # Organize results in original order
        final_vectors = []
        for i in range(len(cleaned_texts)):
            if i in all_vectors:
                final_vectors.append(all_vectors[i])
            else:
                # Create zero vector for failed embeddings
                final_vectors.append(np.zeros(self.get_dimension()))
        
        vectors_array = np.array(final_vectors) if final_vectors else np.array([])
        failed_count = len(cleaned_texts) - successful_count
        
        return BatchEmbeddingResult(
            texts=cleaned_texts,
            vectors=vectors_array,
            model_name=self.model_name,
            provider=self.get_provider(),
            metadata={
                'device': self.device,
                'batch_size': batch_size,
                'cached_count': len(cached_results),
                'embedding_time': time.time() - start_time if 'start_time' in locals() else 0
            },
            successful_count=successful_count,
            failed_count=failed_count,
            errors=errors
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'provider': self.get_provider().value,
            'dimension': self.get_dimension(),
            'device': self.device,
            'cache_size': self.cache.size() if self.cache else 0,
            'model_loaded': self.model is not None
        }


# Commonly used models with their properties
RECOMMENDED_MODELS = {
    'all-MiniLM-L6-v2': {
        'dimension': 384,
        'description': 'Fast and lightweight, good for most use cases',
        'size': '80MB',
        'speed': 'Very Fast'
    },
    'all-MiniLM-L12-v2': {
        'dimension': 384,
        'description': 'Slightly better quality than L6, still fast',
        'size': '120MB',
        'speed': 'Fast'
    },
    'all-mpnet-base-v2': {
        'dimension': 768,
        'description': 'High quality embeddings, slower but better',
        'size': '420MB',
        'speed': 'Medium'
    },
    'all-distilroberta-v1': {
        'dimension': 768,
        'description': 'RoBERTa-based, good quality',
        'size': '290MB',
        'speed': 'Medium'
    }
}


def create_sentence_transformers_embedder(
    model_name: str = "all-MiniLM-L6-v2",
    **kwargs
) -> SentenceTransformersEmbedder:
    """
    Factory function to create a SentenceTransformers embedder.
    
    Args:
        model_name: Name of the model to use
        **kwargs: Additional configuration options
        
    Returns:
        Configured SentenceTransformersEmbedder
    """
    return SentenceTransformersEmbedder(model_name=model_name, **kwargs)


def list_recommended_models() -> Dict[str, Dict[str, Any]]:
    """Get list of recommended SentenceTransformers models."""
    return RECOMMENDED_MODELS.copy() 