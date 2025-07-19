"""Test script for the embedding system."""

import logging
import sys
import time
from typing import Dict, Any

from rag_cli.embedding.service import embedding_service
from rag_cli.embedding.sentence_transformers import list_recommended_models
from rag_cli.storage.database import check_database_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_database_connection():
    """Test that database connection is working."""
    logger.info("Testing database connection...")
    
    if not check_database_connection():
        logger.error("Database connection failed!")
        return False
    
    logger.info("Database connection successful!")
    return True


def test_model_loading():
    """Test loading the SentenceTransformers model."""
    logger.info("Testing SentenceTransformers model loading...")
    
    try:
        # Get embedder info
        info = embedding_service.get_embedder_info()
        logger.info(f"Embedder loaded: {info}")
        
        # Test basic functionality
        if info.get('model_loaded', False):
            logger.info("✅ Model loaded successfully!")
            return True
        else:
            logger.error("❌ Model failed to load")
            return False
            
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False


def test_single_embedding():
    """Test embedding a single piece of text."""
    logger.info("Testing single text embedding...")
    
    try:
        test_text = "This is a test sentence for embedding. Machine learning is fascinating!"
        
        start_time = time.time()
        result = embedding_service.embedder.embed_text(test_text)
        embedding_time = time.time() - start_time
        
        logger.info(f"✅ Single embedding successful!")
        logger.info(f"   Text: '{test_text[:50]}...'")
        logger.info(f"   Dimension: {result.dimension}")
        logger.info(f"   Time: {embedding_time:.3f}s")
        logger.info(f"   Device: {result.metadata.get('device', 'unknown')}")
        
        # Show first few vector values
        vector_sample = result.vector[:5] if len(result.vector) > 5 else result.vector
        logger.info(f"   Vector sample: {vector_sample}")
        
        return True
        
    except Exception as e:
        logger.error(f"Single embedding failed: {e}")
        return False


def test_batch_embedding():
    """Test embedding multiple texts in batch."""
    logger.info("Testing batch embedding...")
    
    try:
        test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand text.",
            "Deep learning uses neural networks with multiple layers.",
            "Python is a popular programming language for data science.",
            "Embeddings convert text into numerical vectors."
        ]
        
        start_time = time.time()
        batch_result = embedding_service.embedder.embed_batch(test_texts)
        embedding_time = time.time() - start_time
        
        logger.info(f"✅ Batch embedding successful!")
        logger.info(f"   Texts processed: {len(test_texts)}")
        logger.info(f"   Successful: {batch_result.successful_count}")
        logger.info(f"   Failed: {batch_result.failed_count}")
        logger.info(f"   Total time: {embedding_time:.3f}s")
        logger.info(f"   Time per text: {embedding_time/len(test_texts):.3f}s")
        logger.info(f"   Vector shape: {batch_result.vectors.shape}")
        
        return batch_result.successful_count == len(test_texts)
        
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        return False


def test_document_processing():
    """Test processing documents from the database."""
    logger.info("Testing document processing...")
    
    try:
        # Get initial stats
        initial_stats = embedding_service.get_embedding_stats()
        logger.info(f"Initial embedding stats: {initial_stats}")
        
        # Process unembedded documents
        result = embedding_service.process_unembedded_documents()
        
        logger.info(f"Document processing results:")
        logger.info(f"   Processed: {result['processed']}")
        logger.info(f"   Successful: {result['successful']}")
        logger.info(f"   Failed: {result['failed']}")
        
        if result['errors']:
            logger.warning("Errors encountered:")
            for error in result['errors'][:3]:  # Show first 3 errors
                logger.warning(f"   - {error}")
        
        # Get final stats
        final_stats = embedding_service.get_embedding_stats()
        logger.info(f"Final embedding stats: {final_stats}")
        
        return result['successful'] > 0 or result['processed'] == 0
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        return False


def test_similarity_search():
    """Test similarity search functionality."""
    logger.info("Testing similarity search...")
    
    try:
        # Test queries related to our sample data
        test_queries = [
            "artificial intelligence",
            "social media post",
            "online discussion",
            "video content"
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 Searching for: '{query}'")
            
            start_time = time.time()
            similar_docs = embedding_service.search_similar_content(
                query_text=query,
                limit=5,
                similarity_threshold=0.1  # Low threshold for testing
            )
            search_time = time.time() - start_time
            
            logger.info(f"   Found {len(similar_docs)} similar documents in {search_time:.3f}s")
            
            for i, doc in enumerate(similar_docs):
                logger.info(f"   {i+1}. [{doc['content_type']}] {doc.get('title', 'No title')}")
                logger.info(f"      Score: {doc['similarity_score']:.3f}")
                logger.info(f"      Content: {doc['content'][:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return False


def show_model_info():
    """Display information about available models."""
    logger.info("\n📊 SentenceTransformers Model Information:")
    
    models = list_recommended_models()
    for model_name, info in models.items():
        logger.info(f"\n🤖 {model_name}:")
        logger.info(f"   Description: {info['description']}")
        logger.info(f"   Dimension: {info['dimension']}")
        logger.info(f"   Size: {info['size']}")
        logger.info(f"   Speed: {info['speed']}")


def show_embedding_stats():
    """Show detailed embedding statistics."""
    logger.info("\n📈 Detailed Embedding Statistics:")
    
    try:
        stats = embedding_service.get_embedding_stats()
        
        logger.info(f"📄 Documents:")
        logger.info(f"   Total: {stats.get('total_documents', 0)}")
        logger.info(f"   Embedded: {stats.get('embedded_documents', 0)}")
        logger.info(f"   Pending: {stats.get('pending_documents', 0)}")
        logger.info(f"   Coverage: {stats.get('embedding_coverage', '0%')}")
        
        logger.info(f"\n🧠 Embeddings:")
        logger.info(f"   Total: {stats.get('total_embeddings', 0)}")
        
        models = stats.get('models', [])
        if models:
            logger.info(f"\n🤖 Models Used:")
            for model in models:
                logger.info(f"   {model['model_name']} ({model['provider']}):")
                logger.info(f"     Count: {model['count']}")
                logger.info(f"     Dimension: {model['avg_dimension']}")
        
    except Exception as e:
        logger.error(f"Failed to get detailed stats: {e}")


def main():
    """Run all embedding tests."""
    logger.info("🚀 Starting embedding system tests...")
    
    # Show model information first
    show_model_info()
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Model Loading", test_model_loading),
        ("Single Embedding", test_single_embedding),
        ("Batch Embedding", test_batch_embedding),
        ("Document Processing", test_document_processing),
        ("Similarity Search", test_similarity_search)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running: {test_name} ---")
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    # Show final statistics
    show_embedding_stats()
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All embedding tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 