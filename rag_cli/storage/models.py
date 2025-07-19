"""SQLAlchemy database models for RAG CLI."""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, 
    ForeignKey, Index, Boolean, JSON, LargeBinary
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid

from rag_cli.storage.database import Base


class Source(Base):
    """Represents a data source (YouTube channel, Twitter user, Reddit subreddit)."""
    
    __tablename__ = "sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    source_type = Column(String(50), nullable=False)  # youtube, twitter, reddit
    source_id = Column(String(255), nullable=False)  # channel_id, username, subreddit
    url = Column(String(500))
    description = Column(Text)
    source_metadata = Column(JSON)  # Additional source-specific metadata
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_synced = Column(DateTime)
    
    # Relationships
    documents = relationship("Document", back_populates="source", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_source_type_id", "source_type", "source_id"),
        Index("idx_source_active", "active"),
        Index("idx_source_last_synced", "last_synced"),
    )
    
    def __repr__(self):
        return f"<Source(id={self.id}, type={self.source_type}, name={self.name})>"


class Document(Base):
    """Represents a piece of content from a source."""
    
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("sources.id"), nullable=False)
    external_id = Column(String(255))  # Original ID from the source platform
    title = Column(String(500))
    content = Column(Text, nullable=False)
    content_type = Column(String(50))  # tweet, video_transcript, reddit_post, reddit_comment
    url = Column(String(1000))
    author = Column(String(255))
    published_at = Column(DateTime)
    document_metadata = Column(JSON)  # Platform-specific metadata (likes, retweets, etc.)
    
    # Content processing
    word_count = Column(Integer)
    char_count = Column(Integer)
    language = Column(String(10))
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    is_embedded = Column(Boolean, default=False)
    processing_error = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source = relationship("Source", back_populates="documents")
    embeddings = relationship("Embedding", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_document_source_id", "source_id"),
        Index("idx_document_external_id", "external_id"),
        Index("idx_document_published_at", "published_at"),
        Index("idx_document_content_type", "content_type"),
        Index("idx_document_is_processed", "is_processed"),
        Index("idx_document_is_embedded", "is_embedded"),
        Index("idx_document_author", "author"),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, type={self.content_type}, title={self.title[:50]}...)>"


class Embedding(Base):
    """Represents vector embeddings for documents."""
    
    __tablename__ = "embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Embedding data
    vector = Column(LargeBinary)  # Serialized numpy array or use pgvector if available
    model_name = Column(String(255), nullable=False)  # e.g., "all-MiniLM-L6-v2"
    embedding_provider = Column(String(50), nullable=False)  # openai, sentence_transformers
    dimension = Column(Integer, nullable=False)
    
    # Chunking information (for large documents)
    chunk_index = Column(Integer, default=0)  # 0 for full document, >0 for chunks
    chunk_text = Column(Text)  # The actual text that was embedded
    chunk_start = Column(Integer)  # Start position in original document
    chunk_end = Column(Integer)    # End position in original document
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="embeddings")
    
    # Indexes
    __table_args__ = (
        Index("idx_embedding_document_id", "document_id"),
        Index("idx_embedding_model", "model_name"),
        Index("idx_embedding_provider", "embedding_provider"),
        Index("idx_embedding_chunk", "document_id", "chunk_index"),
    )
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, document_id={self.document_id}, model={self.model_name})>"


# Additional utility functions for models

def get_documents_by_source_type(session, source_type: str, limit: int = 100):
    """Get documents filtered by source type."""
    return (
        session.query(Document)
        .join(Source)
        .filter(Source.source_type == source_type)
        .order_by(Document.published_at.desc())
        .limit(limit)
        .all()
    )


def get_documents_by_date_range(session, start_date: datetime, end_date: datetime):
    """Get documents within a date range."""
    return (
        session.query(Document)
        .filter(Document.published_at.between(start_date, end_date))
        .order_by(Document.published_at.desc())
        .all()
    )


def get_unprocessed_documents(session, limit: int = 100):
    """Get documents that haven't been processed yet."""
    return (
        session.query(Document)
        .filter(Document.is_processed == False)
        .order_by(Document.created_at.asc())
        .limit(limit)
        .all()
    )


def get_documents_without_embeddings(session, limit: int = 100):
    """Get documents that don't have embeddings yet."""
    return (
        session.query(Document)
        .filter(Document.is_embedded == False)
        .filter(Document.is_processed == True)
        .order_by(Document.created_at.asc())
        .limit(limit)
        .all()
    ) 