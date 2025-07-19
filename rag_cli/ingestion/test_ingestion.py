"""Test script for the ingestion system."""

import logging
import sys
from datetime import datetime, timedelta

from rag_cli.ingestion.manager import ingestion_manager
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


def test_ingestion_manager():
    """Test the ingestion manager setup."""
    logger.info("Testing ingestion manager...")
    
    # Check available sources
    sources = ingestion_manager.get_available_sources()
    logger.info(f"Available sources: {sources}")
    
    if not sources:
        logger.warning("No ingestion sources available!")
        return False
    
    # Test configuration validation
    for source in sources:
        errors = ingestion_manager.validate_source_config(source)
        if errors:
            logger.warning(f"Configuration errors for {source}: {errors}")
        else:
            logger.info(f"Configuration valid for {source}")
    
    return True


def test_youtube_ingestion():
    """Test YouTube ingestion with a public channel."""
    logger.info("Testing YouTube ingestion...")
    
    # Test with a well-known YouTube channel (TED)
    # This channel should have videos with transcripts
    test_channel_id = "UCAuUUnT6oDeKwE6v1NGQxug"  # TED channel
    
    try:
        # Test ingestion with a small limit
        result = ingestion_manager.ingest_youtube_channel(
            channel_id=test_channel_id,
            channel_name="TED",
            limit=2  # Only fetch 2 videos for testing
        )
        
        logger.info(f"Ingestion result: {result.success}")
        logger.info(f"Videos processed: {result.total_fetched}")
        logger.info(f"New content: {result.total_new}")
        logger.info(f"Updated content: {result.total_updated}")
        logger.info(f"Errors: {len(result.errors)}")
        
        if result.errors:
            for error in result.errors[:3]:  # Show first 3 errors
                logger.warning(f"Error: {error}")
        
        if result.content_items:
            # Show info about first content item
            first_item = result.content_items[0]
            logger.info(f"Sample content: {first_item.title[:50]}...")
            logger.info(f"Content length: {len(first_item.content)} characters")
            logger.info(f"Published: {first_item.published_at}")
        
        return result.success
        
    except Exception as e:
        logger.error(f"YouTube ingestion test failed: {e}")
        return False


def test_ingestion_stats():
    """Test getting ingestion statistics."""
    logger.info("Testing ingestion statistics...")
    
    try:
        stats = ingestion_manager.get_ingestion_stats()
        
        logger.info("Ingestion Statistics:")
        logger.info(f"Overall: {stats.get('overall', {})}")
        logger.info(f"Available sources: {stats.get('available_sources', [])}")
        
        by_source = stats.get('by_source', {})
        for source, source_stats in by_source.items():
            logger.info(f"{source}: {source_stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Stats test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting ingestion system tests...")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Ingestion Manager", test_ingestion_manager),
        ("YouTube Ingestion", test_youtube_ingestion),
        ("Ingestion Stats", test_ingestion_stats)
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
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 