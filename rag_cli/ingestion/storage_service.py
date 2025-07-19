"""Storage service for ingested content."""

import logging
from typing import List, Optional, Dict, Tuple
from datetime import datetime

from rag_cli.storage.database import get_db_session
from rag_cli.storage.models import Source, Document
from rag_cli.ingestion.base import IngestedContent, IngestionResult, count_words, detect_language

logger = logging.getLogger(__name__)


class IngestionStorageService:
    """Handles storage of ingested content to the database."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def ensure_source_exists(self, 
                           source_type: str, 
                           source_id: str, 
                           name: str,
                           url: Optional[str] = None,
                           description: Optional[str] = None) -> str:
        """
        Ensure a source exists in the database, create if it doesn't.
        
        Args:
            source_type: Type of source (youtube, twitter, reddit)
            source_id: External source identifier
            name: Human-readable name
            url: Source URL
            description: Source description
            
        Returns:
            UUID of the source record
        """
        with get_db_session() as session:
            # Try to find existing source
            source = session.query(Source).filter(
                Source.source_type == source_type,
                Source.source_id == source_id
            ).first()
            
            if source:
                # Update if needed
                if source.name != name:
                    source.name = name
                if url and source.url != url:
                    source.url = url
                if description and source.description != description:
                    source.description = description
                source.updated_at = datetime.utcnow()
                session.commit()
                self.logger.debug(f"Updated existing source: {source.id}")
            else:
                # Create new source
                source = Source(
                    name=name,
                    source_type=source_type,
                    source_id=source_id,
                    url=url,
                    description=description,
                    source_metadata={}
                )
                session.add(source)
                session.commit()
                self.logger.info(f"Created new source: {source.id} ({source_type}:{source_id})")
            
            return str(source.id)
    
    def content_exists(self, source_uuid: str, external_id: str) -> bool:
        """Check if content already exists in the database."""
        with get_db_session() as session:
            document = session.query(Document).filter(
                Document.source_id == source_uuid,
                Document.external_id == external_id
            ).first()
            return document is not None
    
    def save_content(self, content: IngestedContent, source_uuid: str) -> Tuple[bool, str]:
        """
        Save ingested content to the database.
        
        Args:
            content: The content to save
            source_uuid: UUID of the source record
            
        Returns:
            Tuple of (success, document_id or error_message)
        """
        try:
            with get_db_session() as session:
                # Check if document already exists
                existing = session.query(Document).filter(
                    Document.source_id == source_uuid,
                    Document.external_id == content.external_id
                ).first()
                
                if existing:
                    # Update existing document if content is different
                    if existing.content != content.content:
                        existing.content = content.content
                        existing.title = content.title
                        existing.author = content.author
                        existing.published_at = content.published_at
                        existing.document_metadata = content.metadata
                        existing.updated_at = datetime.utcnow()
                        existing.is_processed = False  # Re-process updated content
                        existing.is_embedded = False   # Re-embed updated content
                        
                        # Update text stats
                        existing.word_count = count_words(existing.content)
                        existing.char_count = len(existing.content)
                        existing.language = detect_language(existing.content)
                        
                        session.commit()
                        self.logger.info(f"Updated existing document: {existing.id}")
                        return True, str(existing.id)
                    else:
                        self.logger.debug(f"Document unchanged: {existing.id}")
                        return True, str(existing.id)
                
                # Create new document
                document = Document(
                    source_id=source_uuid,
                    external_id=content.external_id,
                    title=content.title,
                    content=content.content,
                    content_type=content.content_type.value,
                    url=content.url,
                    author=content.author,
                    published_at=content.published_at,
                    document_metadata=content.metadata,
                    word_count=count_words(content.content),
                    char_count=len(content.content),
                    language=detect_language(content.content),
                    is_processed=True,  # Mark as processed since we've cleaned it
                    is_embedded=False   # Not embedded yet
                )
                
                session.add(document)
                session.commit()
                self.logger.info(f"Created new document: {document.id} ({content.content_type.value})")
                return True, str(document.id)
                
        except Exception as e:
            self.logger.error(f"Failed to save content {content.external_id}: {e}")
            return False, str(e)
    
    def save_ingestion_results(self, 
                             results: IngestionResult, 
                             source_type: str,
                             source_identifier: str,
                             source_name: str,
                             source_url: Optional[str] = None) -> IngestionResult:
        """
        Save all content from ingestion results to the database.
        
        Args:
            results: Ingestion results to save
            source_type: Type of source
            source_identifier: External source identifier  
            source_name: Human-readable source name
            source_url: Source URL
            
        Returns:
            Updated IngestionResult with save statistics
        """
        try:
            # Ensure source exists
            source_uuid = self.ensure_source_exists(
                source_type=source_type,
                source_id=source_identifier,
                name=source_name,
                url=source_url
            )
            
            # Save each content item
            for content in results.content_items:
                success, result_id = self.save_content(content, source_uuid)
                
                if success:
                    # Check if this was a new or updated document
                    if self.content_exists(source_uuid, content.external_id):
                        results.total_updated += 1
                    else:
                        results.total_new += 1
                else:
                    results.add_error(f"Failed to save {content.external_id}: {result_id}")
            
            # Update source sync time
            self.update_source_sync_time(source_uuid)
            
            self.logger.info(
                f"Saved ingestion results: {results.total_new} new, "
                f"{results.total_updated} updated, {len(results.errors)} errors"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save ingestion results: {e}")
            results.add_error(f"Storage error: {e}")
            results.success = False
        
        return results
    
    def update_source_sync_time(self, source_uuid: str):
        """Update the last sync time for a source."""
        try:
            with get_db_session() as session:
                source = session.query(Source).filter(Source.id == source_uuid).first()
                if source:
                    source.last_synced = datetime.utcnow()
                    session.commit()
        except Exception as e:
            self.logger.error(f"Failed to update source sync time: {e}")
    
    def get_latest_content_date(self, source_type: str, source_id: str) -> Optional[datetime]:
        """Get the date of the most recent content for a source."""
        try:
            with get_db_session() as session:
                # Get the source
                source = session.query(Source).filter(
                    Source.source_type == source_type,
                    Source.source_id == source_id
                ).first()
                
                if not source:
                    return None
                
                # Get the latest document
                latest_doc = session.query(Document).filter(
                    Document.source_id == source.id
                ).order_by(Document.published_at.desc()).first()
                
                return latest_doc.published_at if latest_doc else None
                
        except Exception as e:
            self.logger.error(f"Failed to get latest content date: {e}")
            return None
    
    def get_content_stats(self, source_type: Optional[str] = None) -> Dict[str, int]:
        """Get statistics about stored content."""
        try:
            with get_db_session() as session:
                query = session.query(Document)
                
                if source_type:
                    query = query.join(Source).filter(Source.source_type == source_type)
                
                total = query.count()
                processed = query.filter(Document.is_processed == True).count()
                embedded = query.filter(Document.is_embedded == True).count()
                
                return {
                    'total_documents': total,
                    'processed_documents': processed,
                    'embedded_documents': embedded,
                    'pending_embedding': processed - embedded
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get content stats: {e}")
            return {}


# Global storage service instance
storage_service = IngestionStorageService() 