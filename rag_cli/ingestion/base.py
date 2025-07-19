"""Base classes for data ingestion."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Generator
from enum import Enum

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content that can be ingested."""
    YOUTUBE_TRANSCRIPT = "video_transcript"
    TWITTER_TWEET = "tweet"
    REDDIT_POST = "reddit_post"
    REDDIT_COMMENT = "reddit_comment"


@dataclass
class IngestedContent:
    """Represents a piece of content ingested from a source."""
    
    # Core content
    external_id: str  # ID from the source platform
    title: Optional[str]
    content: str
    content_type: ContentType
    url: Optional[str]
    author: Optional[str]
    published_at: Optional[datetime]
    
    # Source information
    source_id: str  # Our internal source identifier
    source_type: str  # youtube, twitter, reddit
    
    # Metadata (platform-specific data)
    metadata: Dict[str, Any]
    
    # Processing flags
    needs_processing: bool = True
    
    def __post_init__(self):
        """Validate and clean the content after initialization."""
        if not self.content or not self.content.strip():
            raise ValueError("Content cannot be empty")
        
        # Clean up content
        self.content = self.content.strip()
        
        # Ensure metadata is a dict
        if self.metadata is None:
            self.metadata = {}


@dataclass
class IngestionResult:
    """Results from an ingestion operation."""
    
    success: bool
    content_items: List[IngestedContent]
    errors: List[str]
    total_fetched: int
    total_new: int
    total_updated: int
    
    def add_content(self, content: IngestedContent):
        """Add a content item to the results."""
        self.content_items.append(content)
        self.total_fetched += 1
    
    def add_error(self, error: str):
        """Add an error to the results."""
        self.errors.append(error)
        logger.error(f"Ingestion error: {error}")


class BaseIngester(ABC):
    """Base class for all content ingesters."""
    
    def __init__(self, source_config: Dict[str, Any]):
        """Initialize the ingester with source-specific configuration."""
        self.source_config = source_config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def get_source_type(self) -> str:
        """Return the source type (e.g., 'youtube', 'twitter', 'reddit')."""
        pass
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate the configuration and return any error messages."""
        pass
    
    @abstractmethod
    def fetch_content(self, 
                     source_id: str, 
                     limit: Optional[int] = None,
                     since: Optional[datetime] = None) -> IngestionResult:
        """
        Fetch content from the source.
        
        Args:
            source_id: Source identifier (channel ID, username, subreddit, etc.)
            limit: Maximum number of items to fetch
            since: Only fetch content newer than this datetime
            
        Returns:
            IngestionResult with fetched content and metadata
        """
        pass
    
    def process_content(self, content: IngestedContent) -> IngestedContent:
        """
        Process and clean content before storage.
        Can be overridden by specific ingesters for custom processing.
        """
        # Basic text cleaning
        if content.content:
            # Remove excessive whitespace
            content.content = ' '.join(content.content.split())
            
            # Basic content validation
            if len(content.content) < 10:
                raise ValueError("Content too short after processing")
        
        return content
    
    def should_ingest_content(self, content: IngestedContent) -> bool:
        """
        Determine if content should be ingested.
        Can be overridden for custom filtering logic.
        """
        # Basic filtering rules
        if not content.content or len(content.content.strip()) < 10:
            return False
        
        # Skip very long content (might be spam or low quality)
        if len(content.content) > 50000:
            self.logger.warning(f"Skipping very long content: {content.external_id}")
            return False
        
        return True
    
    def create_ingestion_result(self) -> IngestionResult:
        """Create a new ingestion result object."""
        return IngestionResult(
            success=True,
            content_items=[],
            errors=[],
            total_fetched=0,
            total_new=0,
            total_updated=0
        )


class RateLimitHandler:
    """Handles rate limiting for API requests."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_times: List[datetime] = []
    
    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        import time
        
        now = datetime.now()
        
        # Remove requests older than 1 minute
        cutoff = now.timestamp() - 60
        self.request_times = [
            req_time for req_time in self.request_times 
            if req_time.timestamp() > cutoff
        ]
        
        # If we're at the limit, wait
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now.timestamp() - self.request_times[0].timestamp())
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(now)


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove common unwanted characters
    text = text.replace('\x00', '')  # Null bytes
    text = text.replace('\ufeff', '')  # BOM
    
    return text.strip()


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text."""
    import re
    return re.findall(r'#\w+', text)


def extract_mentions(text: str) -> List[str]:
    """Extract @mentions from text."""
    import re
    return re.findall(r'@\w+', text)


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split()) if text else 0


def detect_language(text: str) -> str:
    """Detect the language of text."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return 'unknown' 