"""Database initialization and management utilities."""

import logging
import sys
from typing import Optional

from rag_cli.storage.database import init_database, check_database_connection, get_db_session
from rag_cli.storage.models import Source, Document, Embedding
from config.config import config, validate_config

logger = logging.getLogger(__name__)


def setup_database() -> bool:
    """Set up the database and create all tables."""
    try:
        # Validate configuration
        config_errors = validate_config()
        if config_errors:
            logger.error("Configuration errors:")
            for error in config_errors:
                logger.error(f"  - {error}")
            return False
        
        # Initialize database
        logger.info("Initializing database...")
        init_database()
        
        # Test connection
        if not check_database_connection():
            logger.error("Database connection test failed")
            return False
        
        logger.info("Database setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def reset_database() -> bool:
    """Drop all tables and recreate them (DESTRUCTIVE!)."""
    try:
        from rag_cli.storage.database import Base, get_engine
        
        logger.warning("Resetting database - ALL DATA WILL BE LOST!")
        
        engine = get_engine()
        
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        logger.info("Dropped all tables")
        
        # Recreate all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Recreated all tables")
        
        return True
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        return False


def get_database_stats() -> dict:
    """Get statistics about the database contents."""
    try:
        with get_db_session() as session:
            stats = {
                "sources": session.query(Source).count(),
                "documents": session.query(Document).count(),
                "embeddings": session.query(Embedding).count(),
                "processed_documents": session.query(Document).filter(Document.is_processed == True).count(),
                "embedded_documents": session.query(Document).filter(Document.is_embedded == True).count(),
            }
            
            # Get source type breakdown
            from sqlalchemy import func
            source_types = {}
            for source_type, count in session.query(Source.source_type, func.count(Source.id)).group_by(Source.source_type).all():
                source_types[source_type] = count
            stats["source_types"] = source_types
            
            return stats
            
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}


def create_sample_data() -> bool:
    """Create sample data for testing (only if database is empty)."""
    try:
        with get_db_session() as session:
            # Check if data already exists
            if session.query(Source).count() > 0:
                logger.info("Database already contains data, skipping sample data creation")
                return True
            
            # Create sample sources
            youtube_source = Source(
                name="Sample YouTube Channel",
                source_type="youtube",
                source_id="UC_sample_channel_id",
                url="https://youtube.com/channel/UC_sample_channel_id",
                description="A sample YouTube channel for testing"
            )
            
            twitter_source = Source(
                name="Sample Twitter User",
                source_type="twitter",
                source_id="sample_user",
                url="https://twitter.com/sample_user",
                description="A sample Twitter user for testing"
            )
            
            reddit_source = Source(
                name="Sample Subreddit",
                source_type="reddit",
                source_id="samplesubreddit",
                url="https://reddit.com/r/samplesubreddit",
                description="A sample subreddit for testing"
            )
            
            session.add_all([youtube_source, twitter_source, reddit_source])
            session.flush()  # Get IDs
            
            # Create sample documents
            youtube_doc = Document(
                source_id=youtube_source.id,
                external_id="sample_video_id",
                title="Sample YouTube Video",
                content="This is a sample YouTube video transcript for testing the RAG system.",
                content_type="video_transcript",
                url="https://youtube.com/watch?v=sample_video_id",
                author="Sample Creator",
                word_count=14,
                char_count=79,
                is_processed=True
            )
            
            twitter_doc = Document(
                source_id=twitter_source.id,
                external_id="sample_tweet_id",
                title=None,
                content="This is a sample tweet for testing the RAG system. #testing",
                content_type="tweet",
                url="https://twitter.com/sample_user/status/sample_tweet_id",
                author="sample_user",
                word_count=12,
                char_count=65,
                is_processed=True
            )
            
            reddit_doc = Document(
                source_id=reddit_source.id,
                external_id="sample_post_id",
                title="Sample Reddit Post",
                content="This is a sample Reddit post for testing the RAG system functionality.",
                content_type="reddit_post",
                url="https://reddit.com/r/samplesubreddit/comments/sample_post_id",
                author="sample_redditor",
                word_count=13,
                char_count=76,
                is_processed=True
            )
            
            session.add_all([youtube_doc, twitter_doc, reddit_doc])
            session.commit()
            
            logger.info("Sample data created successfully")
            return True
            
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            success = setup_database()
            sys.exit(0 if success else 1)
            
        elif command == "reset":
            print("WARNING: This will delete all data in the database!")
            confirm = input("Type 'yes' to confirm: ")
            if confirm.lower() == "yes":
                success = reset_database()
                sys.exit(0 if success else 1)
            else:
                print("Reset cancelled")
                sys.exit(0)
                
        elif command == "stats":
            stats = get_database_stats()
            print("Database Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        elif command == "sample":
            success = create_sample_data()
            sys.exit(0 if success else 1)
            
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    else:
        print("Usage: python -m rag_cli.storage.init_db <command>")
        print("Commands: setup, reset, stats, sample")
        sys.exit(1) 