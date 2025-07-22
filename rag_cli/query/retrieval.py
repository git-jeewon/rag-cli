"""Context retrieval system for RAG queries."""

import logging
import pickle
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from rag_cli.embedding.service import embedding_service
from rag_cli.embedding.base import cosine_similarity
from rag_cli.storage.database import get_db_session
from rag_cli.storage.models import Document, Embedding, Source

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A document retrieved for context."""
    
    document_id: str
    title: Optional[str]
    content: str
    content_type: str
    author: Optional[str]
    url: Optional[str]
    published_at: Optional[datetime]
    source_name: Optional[str]
    similarity_score: float
    chunk_text: Optional[str] = None
    
    def to_context_string(self, include_metadata: bool = True) -> str:
        """Convert to formatted context string for LLM."""
        context_parts = []
        
        if include_metadata:
            # Add metadata header
            metadata_parts = []
            if self.content_type:
                metadata_parts.append(f"Type: {self.content_type}")
            if self.source_name:
                metadata_parts.append(f"Source: {self.source_name}")
            if self.author:
                metadata_parts.append(f"Author: {self.author}")
            if self.published_at:
                metadata_parts.append(f"Date: {self.published_at.strftime('%Y-%m-%d')}")
            
            if metadata_parts:
                context_parts.append(f"[{', '.join(metadata_parts)}]")
        
        # Add title if available
        if self.title:
            context_parts.append(f"Title: {self.title}")
        
        # Add content
        context_parts.append(f"Content: {self.content}")
        
        return "\n".join(context_parts)


@dataclass
class RetrievalResult:
    """Result of context retrieval."""
    
    query: str
    documents: List[RetrievedDocument]
    total_found: int
    retrieval_time: float
    embedding_time: float
    search_metadata: Dict[str, Any]
    
    def get_context_string(self, 
                          max_documents: Optional[int] = None,
                          include_metadata: bool = True) -> str:
        """Get formatted context string for LLM."""
        docs_to_use = self.documents
        if max_documents:
            docs_to_use = docs_to_use[:max_documents]
        
        context_parts = []
        for i, doc in enumerate(docs_to_use, 1):
            context_parts.append(f"Document {i}:")
            context_parts.append(doc.to_context_string(include_metadata))
            context_parts.append("")  # Empty line between documents
        
        return "\n".join(context_parts)


class ContextRetriever:
    """Retrieves relevant context for RAG queries."""
    
    def __init__(self):
        """Initialize the context retriever."""
        self.logger = logging.getLogger(__name__)
        self.embedding_service = embedding_service
    
    def retrieve_context(self,
                        query: str,
                        limit: int = 5,
                        similarity_threshold: float = 0.1,
                        content_types: Optional[List[str]] = None,
                        date_range: Optional[Tuple[datetime, datetime]] = None,
                        sources: Optional[List[str]] = None) -> RetrievalResult:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: The search query
            limit: Maximum number of documents to retrieve
            similarity_threshold: Minimum similarity score
            content_types: Filter by content types (e.g., ['youtube_video', 'tweet'])
            date_range: Filter by date range (start_date, end_date)
            sources: Filter by source names
            
        Returns:
            RetrievalResult with relevant documents
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Retrieving context for query: '{query}'")
        
        # Generate embedding for query
        embedding_start = time.time()
        query_result = self.embedding_service.embedder.embed_text(query)
        query_vector = query_result.vector
        embedding_time = time.time() - embedding_start
        
        # Retrieve and rank documents
        relevant_docs = []
        
        with get_db_session() as session:
            # Get all embeddings (we'll add filtering later)
            embeddings_query = session.query(Embedding).filter(
                Embedding.model_name == self.embedding_service.embedder.model_name
            )
            
            embeddings = embeddings_query.all()
            
            # Calculate similarities
            similarities = []
            for embedding in embeddings:
                try:
                    # Deserialize vector
                    stored_vector = pickle.loads(embedding.vector)
                    
                    # Calculate similarity
                    similarity = cosine_similarity(query_vector, stored_vector)
                    
                    if similarity >= similarity_threshold:
                        similarities.append((embedding, similarity))
                
                except Exception as e:
                    self.logger.warning(f"Error processing embedding {embedding.id}: {e}")
                    continue
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get document details and apply filters
            for embedding, similarity in similarities[:limit * 2]:  # Get extra for filtering
                try:
                    doc = session.query(Document).filter(
                        Document.id == embedding.document_id
                    ).first()
                    
                    if not doc:
                        continue
                    
                    # Apply content type filter
                    if content_types and doc.content_type not in content_types:
                        continue
                    
                    # Apply date range filter
                    if date_range and doc.published_at:
                        start_date, end_date = date_range
                        if not (start_date <= doc.published_at <= end_date):
                            continue
                    
                    # Get source information
                    source = session.query(Source).filter(
                        Source.id == doc.source_id
                    ).first()
                    
                    # Apply source filter
                    if sources and source and source.name not in sources:
                        continue
                    
                    retrieved_doc = RetrievedDocument(
                        document_id=str(doc.id),
                        title=doc.title,
                        content=doc.content,
                        content_type=doc.content_type,
                        author=doc.author,
                        url=doc.url,
                        published_at=doc.published_at,
                        source_name=source.name if source else None,
                        similarity_score=similarity,
                        chunk_text=embedding.chunk_text
                    )
                    
                    relevant_docs.append(retrieved_doc)
                    
                    # Stop if we have enough
                    if len(relevant_docs) >= limit:
                        break
                
                except Exception as e:
                    self.logger.warning(f"Error processing document for embedding {embedding.id}: {e}")
                    continue
        
        retrieval_time = time.time() - start_time
        
        self.logger.info(
            f"Retrieved {len(relevant_docs)} documents in {retrieval_time:.3f}s "
            f"(embedding: {embedding_time:.3f}s)"
        )
        
        return RetrievalResult(
            query=query,
            documents=relevant_docs,
            total_found=len(relevant_docs),
            retrieval_time=retrieval_time,
            embedding_time=embedding_time,
            search_metadata={
                'similarity_threshold': similarity_threshold,
                'limit': limit,
                'content_types': content_types,
                'date_range': date_range,
                'sources': sources,
                'total_embeddings_checked': len(embeddings) if 'embeddings' in locals() else 0
            }
        )
    
    def retrieve_similar_to_document(self,
                                   document_id: str,
                                   limit: int = 5,
                                   similarity_threshold: float = 0.3) -> RetrievalResult:
        """
        Find documents similar to a specific document.
        
        Args:
            document_id: ID of the document to find similar content for
            limit: Maximum number of similar documents
            similarity_threshold: Minimum similarity score
            
        Returns:
            RetrievalResult with similar documents
        """
        with get_db_session() as session:
            # Get the target document
            target_doc = session.query(Document).filter(
                Document.id == document_id
            ).first()
            
            if not target_doc:
                raise ValueError(f"Document {document_id} not found")
            
            # Use document title + content as query
            query_parts = []
            if target_doc.title:
                query_parts.append(target_doc.title)
            if target_doc.content:
                query_parts.append(target_doc.content[:500])  # First 500 chars
            
            query = " ".join(query_parts)
            
            # Retrieve similar documents (excluding the original)
            result = self.retrieve_context(
                query=query,
                limit=limit + 1,  # Get extra to account for filtering
                similarity_threshold=similarity_threshold
            )
            
            # Filter out the original document
            filtered_docs = [
                doc for doc in result.documents 
                if doc.document_id != document_id
            ][:limit]
            
            return RetrievalResult(
                query=f"Similar to: {target_doc.title or 'Document'}",
                documents=filtered_docs,
                total_found=len(filtered_docs),
                retrieval_time=result.retrieval_time,
                embedding_time=result.embedding_time,
                search_metadata={
                    **result.search_metadata,
                    'source_document_id': document_id,
                    'source_document_title': target_doc.title
                }
            )
    
    def get_recent_context(self,
                          query: str,
                          days: int = 30,
                          limit: int = 5) -> RetrievalResult:
        """
        Retrieve context from recent documents.
        
        Args:
            query: Search query
            days: Number of recent days to search
            limit: Maximum documents to return
            
        Returns:
            RetrievalResult with recent relevant documents
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.retrieve_context(
            query=query,
            limit=limit,
            date_range=(start_date, end_date)
        )
    
    def get_context_by_source(self,
                             query: str,
                             source_type: str,
                             limit: int = 5) -> RetrievalResult:
        """
        Retrieve context from specific source type.
        
        Args:
            query: Search query
            source_type: Type of source (e.g., 'youtube', 'twitter', 'reddit')
            limit: Maximum documents to return
            
        Returns:
            RetrievalResult with documents from specified source
        """
        # Map source types to content types
        content_type_mapping = {
            'youtube': ['video_transcript'],
            'twitter': ['tweet'],
            'reddit': ['reddit_post'],
            'web': ['article', 'blog_post'],
        }
        
        content_types = content_type_mapping.get(source_type, [source_type])
        
        return self.retrieve_context(
            query=query,
            limit=limit,
            content_types=content_types
        )


# Global retriever instance
context_retriever = ContextRetriever() 