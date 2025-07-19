"""Ingestion manager for coordinating content ingestion from various sources."""

import logging
from typing import Dict, List, Optional, Type
from datetime import datetime

from rag_cli.ingestion.base import BaseIngester, IngestionResult
from rag_cli.ingestion.storage_service import storage_service
from rag_cli.ingestion.youtube import YouTubeIngester

logger = logging.getLogger(__name__)


class IngestionManager:
    """Manages content ingestion from multiple sources."""
    
    def __init__(self):
        self.ingesters: Dict[str, BaseIngester] = {}
        self._register_ingesters()
    
    def _register_ingesters(self):
        """Register all available ingesters."""
        # Register YouTube ingester
        try:
            youtube_ingester = YouTubeIngester()
            self.ingesters['youtube'] = youtube_ingester
            logger.info("Registered YouTube ingester")
        except Exception as e:
            logger.warning(f"Failed to register YouTube ingester: {e}")
        
        # TODO: Register other ingesters (Twitter, Reddit) when implemented
    
    def get_available_sources(self) -> List[str]:
        """Get list of available ingestion sources."""
        return list(self.ingesters.keys())
    
    def validate_source_config(self, source_type: str) -> List[str]:
        """Validate configuration for a specific source type."""
        if source_type not in self.ingesters:
            return [f"Unknown source type: {source_type}"]
        
        return self.ingesters[source_type].validate_config()
    
    def ingest_from_source(self, 
                          source_type: str,
                          source_identifier: str,
                          source_name: str,
                          limit: Optional[int] = None,
                          since: Optional[datetime] = None,
                          save_to_db: bool = True) -> IngestionResult:
        """
        Ingest content from a specific source.
        
        Args:
            source_type: Type of source ('youtube', 'twitter', 'reddit')
            source_identifier: Source identifier (channel ID, username, etc.)
            source_name: Human-readable name for the source
            limit: Maximum number of items to fetch
            since: Only fetch content newer than this date
            save_to_db: Whether to save results to database
            
        Returns:
            IngestionResult with fetched content and metadata
        """
        logger.info(f"Starting ingestion from {source_type}: {source_name}")
        
        # Validate source type
        if source_type not in self.ingesters:
            result = IngestionResult(
                success=False,
                content_items=[],
                errors=[f"Unknown source type: {source_type}"],
                total_fetched=0,
                total_new=0,
                total_updated=0
            )
            return result
        
        # Get the appropriate ingester
        ingester = self.ingesters[source_type]
        
        # Validate configuration
        config_errors = ingester.validate_config()
        if config_errors:
            result = IngestionResult(
                success=False,
                content_items=[],
                errors=config_errors,
                total_fetched=0,
                total_new=0,
                total_updated=0
            )
            return result
        
        try:
            # If since is not provided, get the latest content date from DB
            if since is None and save_to_db:
                since = storage_service.get_latest_content_date(source_type, source_identifier)
                if since:
                    logger.info(f"Fetching content since: {since}")
            
            # Fetch content
            result = ingester.fetch_content(source_identifier, limit, since)
            
            # Save to database if requested
            if save_to_db and result.success and result.content_items:
                # Determine source URL based on type
                source_url = self._get_source_url(source_type, source_identifier)
                
                result = storage_service.save_ingestion_results(
                    results=result,
                    source_type=source_type,
                    source_identifier=source_identifier,
                    source_name=source_name,
                    source_url=source_url
                )
            
            logger.info(
                f"Ingestion completed for {source_name}: "
                f"{result.total_fetched} fetched, {result.total_new} new, "
                f"{result.total_updated} updated, {len(result.errors)} errors"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed for {source_name}: {e}")
            result = IngestionResult(
                success=False,
                content_items=[],
                errors=[f"Ingestion failed: {e}"],
                total_fetched=0,
                total_new=0,
                total_updated=0
            )
            return result
    
    def _get_source_url(self, source_type: str, source_identifier: str) -> Optional[str]:
        """Generate source URL based on type and identifier."""
        if source_type == 'youtube':
            return f"https://www.youtube.com/channel/{source_identifier}"
        elif source_type == 'twitter':
            return f"https://twitter.com/{source_identifier}"
        elif source_type == 'reddit':
            return f"https://reddit.com/r/{source_identifier}"
        return None
    
    def ingest_youtube_channel(self, 
                              channel_id: str,
                              channel_name: Optional[str] = None,
                              limit: Optional[int] = None,
                              since: Optional[datetime] = None) -> IngestionResult:
        """
        Convenience method for ingesting YouTube channel content.
        
        Args:
            channel_id: YouTube channel ID
            channel_name: Optional human-readable name (will be fetched if not provided)
            limit: Maximum number of videos to process
            since: Only fetch videos newer than this date
            
        Returns:
            IngestionResult with fetched content
        """
        # Get channel name if not provided
        if not channel_name:
            try:
                from rag_cli.ingestion.youtube import get_channel_info
                from config.config import config
                
                channel_info = get_channel_info(channel_id, config.youtube.api_key)
                channel_name = channel_info['title'] if channel_info else f"Channel {channel_id}"
            except Exception:
                channel_name = f"Channel {channel_id}"
        
        return self.ingest_from_source(
            source_type='youtube',
            source_identifier=channel_id,
            source_name=channel_name,
            limit=limit,
            since=since
        )
    
    def get_ingestion_stats(self) -> Dict[str, any]:
        """Get statistics about ingested content."""
        try:
            stats = storage_service.get_content_stats()
            
            # Add per-source statistics
            source_stats = {}
            for source_type in self.get_available_sources():
                source_stats[source_type] = storage_service.get_content_stats(source_type)
            
            return {
                'overall': stats,
                'by_source': source_stats,
                'available_sources': self.get_available_sources()
            }
            
        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {e}")
            return {}


# Global ingestion manager instance
ingestion_manager = IngestionManager()


def ingest_content(source_type: str, 
                  source_identifier: str, 
                  source_name: str,
                  limit: Optional[int] = None) -> IngestionResult:
    """
    Convenience function for ingesting content.
    
    Args:
        source_type: Type of source ('youtube', 'twitter', 'reddit')
        source_identifier: Source identifier (channel ID, username, etc.)
        source_name: Human-readable name for the source
        limit: Maximum number of items to fetch
        
    Returns:
        IngestionResult with fetched content
    """
    return ingestion_manager.ingest_from_source(
        source_type=source_type,
        source_identifier=source_identifier,
        source_name=source_name,
        limit=limit
    ) 