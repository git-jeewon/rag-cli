"""Database connection and session management."""

import logging
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator

from config.config import config

logger = logging.getLogger(__name__)

# Base class for all models
Base = declarative_base()

# Database engine
engine = None
SessionLocal = None


def init_database() -> None:
    """Initialize database connection and create tables."""
    global engine, SessionLocal
    
    try:
        # Create engine with connection pooling
        engine = create_engine(
            config.database.url,
            echo=(config.log_level == "DEBUG"),  # Log SQL queries in debug mode
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections every hour
        )
        
        # Note: PostgreSQL event listeners can be added here if needed
        
        # Create session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Import models to register them
        from rag_cli.storage.models import Source, Document, Embedding
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_engine():
    """Get database engine."""
    if engine is None:
        init_database()
    return engine


def get_session_factory():
    """Get session factory."""
    if SessionLocal is None:
        init_database()
    return SessionLocal


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get database session with automatic cleanup."""
    if SessionLocal is None:
        init_database()
        
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def check_database_connection() -> bool:
    """Check if database connection is working."""
    try:
        from sqlalchemy import text
        with get_db_session() as session:
            session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


 