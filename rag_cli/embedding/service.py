"""Embedding service for processing documents and storing embeddings."""

import logging
import pickle
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from rag_cli.embedding.base import BaseEmbedder, EmbeddingResult
from rag_cli.embedding.sentence_transformers import SentenceTransformersEmbedder
from rag_cli.storage.database import get_db_session
from rag_cli.storage.models import Document, Embedding
from config.config import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for managing document embeddings."""
    
    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        """
        Initialize the embedding service.
        
        Args:
            embedder: Optional embedder instance. If None, will create default.
        """
        self.embedder = embedder or self._create_default_embedder()
        self.logger = logging.getLogger(__name__)
    
    def _create_default_embedder(self) -> BaseEmbedder:
        """Create default embedder based on configuration."""
        provider = config.embedding.provider.lower()
        model_name = config.embedding.model_name
        
        if provider == "sentence_transformers":
            return SentenceTransformersEmbedder(
                model_name=model_name,
                batch_size=config.embedding.batch_size,
                max_length=config.embedding.max_length,
                use_cache=True
            )
        elif provider == "openai":
            # TODO: Implement OpenAI embedder when needed
            raise NotImplementedError("OpenAI embeddings not yet implemented")
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
    
    def process_unembedded_documents(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all documents that don't have embeddings yet.
        
        Args:
            limit: Maximum number of documents to process
            
        Returns:
            Statistics about the processing
        """
        self.logger.info("Starting batch embedding processing...")
        
        # Get unembedded documents
        documents = self._get_unembedded_documents(limit)
        
        if not documents:
            self.logger.info("No documents need embedding")
            return {
                'processed': 0,
                'successful': 0,
                'failed': 0,
                'errors': []
            }
        
        self.logger.info(f"Processing {len(documents)} documents for embedding")
        
        # Process in batches
        batch_size = config.embedding.batch_size
        total_processed = 0
        total_successful = 0
        total_failed = 0
        all_errors = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            try:
                stats = self._process_document_batch(batch_docs)
                total_processed += stats['processed']
                total_successful += stats['successful']
                total_failed += stats['failed']
                all_errors.extend(stats['errors'])
                
            except Exception as e:
                error_msg = f"Batch processing failed: {e}"
                self.logger.error(error_msg)
                all_errors.append(error_msg)
                total_failed += len(batch_docs)
        
        self.logger.info(
            f"Embedding processing completed: {total_successful} successful, "
            f"{total_failed} failed out of {total_processed} total"
        )
        
        return {
            'processed': total_processed,
            'successful': total_successful,
            'failed': total_failed,
            'errors': all_errors
        }
    
    def _get_unembedded_documents(self, limit: Optional[int] = None) -> List[Document]:
        """Get documents that need embeddings."""
        with get_db_session() as session:
            query = session.query(Document).filter(
                Document.is_processed == True,
                Document.is_embedded == False
            ).order_by(Document.created_at.asc())
            
            if limit:
                query = query.limit(limit)
            
            # Detach from session to avoid session issues
            documents = query.all()
            for doc in documents:
                session.expunge(doc)
            return documents
    
    def _process_document_batch(self, documents: List[Document]) -> Dict[str, Any]:
        """Process a batch of documents for embedding."""
        if not documents:
            return {'processed': 0, 'successful': 0, 'failed': 0, 'errors': []}
        
        # Extract texts and document info
        texts = []
        doc_info = []
        
        for doc in documents:
            # Combine title and content for embedding
            text_parts = []
            if doc.title:
                text_parts.append(doc.title)
            if doc.content:
                text_parts.append(doc.content)
            
            full_text = " ".join(text_parts)
            texts.append(full_text)
            doc_info.append({
                'document': doc,
                'text': full_text
            })
        
        # Generate embeddings
        try:
            batch_result = self.embedder.embed_batch(texts)
            
            successful_count = 0
            failed_count = 0
            errors = []
            
            # Store embeddings in database
            with get_db_session() as session:
                for i, info in enumerate(doc_info):
                    doc = info['document']
                    
                    try:
                        if i < batch_result.successful_count and batch_result.vectors is not None:
                            # Get the embedding vector
                            vector = batch_result.vectors[i]
                            
                            # Serialize vector for storage
                            vector_data = pickle.dumps(vector)
                            
                            # Create embedding record
                            embedding = Embedding(
                                document_id=doc.id,
                                vector=vector_data,
                                model_name=self.embedder.model_name,
                                embedding_provider=self.embedder.get_provider().value,
                                dimension=len(vector),
                                chunk_index=0,
                                chunk_text=info['text'][:1000]  # Store first 1000 chars for reference
                            )
                            
                            session.add(embedding)
                            
                            # Update document status - get fresh copy from database
                            db_doc = session.query(Document).filter(Document.id == doc.id).first()
                            if db_doc:
                                db_doc.is_embedded = True
                            
                            successful_count += 1
                            self.logger.debug(f"Embedded document {doc.id}")
                        
                        else:
                            error_msg = f"No embedding generated for document {doc.id}"
                            errors.append(error_msg)
                            failed_count += 1
                    
                    except Exception as e:
                        error_msg = f"Failed to store embedding for document {doc.id}: {e}"
                        errors.append(error_msg)
                        failed_count += 1
                        self.logger.warning(error_msg)
                
                session.commit()
            
            return {
                'processed': len(documents),
                'successful': successful_count,
                'failed': failed_count,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Batch embedding generation failed: {e}"
            self.logger.error(error_msg)
            return {
                'processed': len(documents),
                'successful': 0,
                'failed': len(documents),
                'errors': [error_msg]
            }
    
    def embed_single_document(self, document_id: str) -> bool:
        """
        Embed a single document by ID.
        
        Args:
            document_id: UUID of the document to embed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with get_db_session() as session:
                doc = session.query(Document).filter(Document.id == document_id).first()
                
                if not doc:
                    self.logger.error(f"Document not found: {document_id}")
                    return False
                
                # Check if already embedded
                existing_embedding = session.query(Embedding).filter(
                    Embedding.document_id == document_id,
                    Embedding.model_name == self.embedder.model_name
                ).first()
                
                if existing_embedding:
                    self.logger.info(f"Document {document_id} already has embedding")
                    return True
                
                # Prepare text
                text_parts = []
                if doc.title:
                    text_parts.append(doc.title)
                if doc.content:
                    text_parts.append(doc.content)
                
                full_text = " ".join(text_parts)
                
                # Generate embedding
                result = self.embedder.embed_text(full_text)
                
                # Store in database
                vector_data = pickle.dumps(result.vector)
                
                embedding = Embedding(
                    document_id=doc.id,
                    vector=vector_data,
                    model_name=self.embedder.model_name,
                    embedding_provider=self.embedder.get_provider().value,
                    dimension=result.dimension,
                    chunk_index=0,
                    chunk_text=full_text[:1000]
                )
                
                session.add(embedding)
                
                # Update document
                doc.is_embedded = True
                session.merge(doc)
                
                session.commit()
                
                self.logger.info(f"Successfully embedded document {document_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to embed document {document_id}: {e}")
            return False
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings in the database."""
        try:
            with get_db_session() as session:
                # Count embeddings by model
                from sqlalchemy import func
                
                model_stats = session.query(
                    Embedding.model_name,
                    Embedding.embedding_provider,
                    func.count(Embedding.id).label('count'),
                    func.avg(Embedding.dimension).label('avg_dimension')
                ).group_by(Embedding.model_name, Embedding.embedding_provider).all()
                
                # Total counts
                total_embeddings = session.query(Embedding).count()
                total_documents = session.query(Document).count()
                embedded_documents = session.query(Document).filter(Document.is_embedded == True).count()
                pending_documents = session.query(Document).filter(
                    Document.is_processed == True,
                    Document.is_embedded == False
                ).count()
                
                return {
                    'total_embeddings': total_embeddings,
                    'total_documents': total_documents,
                    'embedded_documents': embedded_documents,
                    'pending_documents': pending_documents,
                    'embedding_coverage': f"{(embedded_documents/total_documents*100):.1f}%" if total_documents > 0 else "0%",
                    'models': [
                        {
                            'model_name': stat.model_name,
                            'provider': stat.embedding_provider,
                            'count': stat.count,
                            'avg_dimension': int(stat.avg_dimension) if stat.avg_dimension else 0
                        }
                        for stat in model_stats
                    ]
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get embedding stats: {e}")
            return {}
    
    def search_similar_content(self, 
                             query_text: str, 
                             limit: int = 10, 
                             similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find documents similar to the query text.
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Generate embedding for query
            query_result = self.embedder.embed_text(query_text)
            query_vector = query_result.vector
            
            # Get all embeddings from database
            similar_docs = []
            
            with get_db_session() as session:
                embeddings = session.query(Embedding).filter(
                    Embedding.model_name == self.embedder.model_name
                ).all()
                
                for embedding in embeddings:
                    try:
                        # Deserialize vector
                        stored_vector = pickle.loads(embedding.vector)
                        
                        # Calculate similarity
                        from rag_cli.embedding.base import cosine_similarity
                        similarity = cosine_similarity(query_vector, stored_vector)
                        
                        if similarity >= similarity_threshold:
                            # Get document info
                            doc = session.query(Document).filter(
                                Document.id == embedding.document_id
                            ).first()
                            
                            if doc:
                                similar_docs.append({
                                    'document_id': str(doc.id),
                                    'title': doc.title,
                                    'content': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                                    'content_type': doc.content_type,
                                    'author': doc.author,
                                    'similarity_score': float(similarity),
                                    'url': doc.url
                                })
                    
                    except Exception as e:
                        self.logger.warning(f"Error processing embedding {embedding.id}: {e}")
                        continue
            
            # Sort by similarity and limit results
            similar_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_docs[:limit]
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []
    
    def get_embedder_info(self) -> Dict[str, Any]:
        """Get information about the current embedder."""
        if hasattr(self.embedder, 'get_model_info'):
            return self.embedder.get_model_info()
        else:
            return {
                'model_name': self.embedder.model_name,
                'provider': self.embedder.get_provider().value,
                'dimension': self.embedder.get_dimension()
            }


# Global embedding service instance
embedding_service = EmbeddingService() 