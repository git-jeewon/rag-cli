"""YouTube content ingester."""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from rag_cli.ingestion.base import (
    BaseIngester, IngestedContent, IngestionResult, ContentType, RateLimitHandler
)
from config.config import config

logger = logging.getLogger(__name__)


class YouTubeIngester(BaseIngester):
    """Ingests video transcripts from YouTube channels."""
    
    def __init__(self):
        super().__init__(asdict(config.youtube))
        self.api_key = config.youtube.api_key
        self.rate_limiter = RateLimitHandler(requests_per_minute=100)  # YouTube API limit
    
    def get_source_type(self) -> str:
        """Return the source type."""
        return "youtube"
    
    def validate_config(self) -> List[str]:
        """Validate YouTube API configuration."""
        errors = []
        
        if not self.api_key:
            errors.append("YouTube API key is required (YOUTUBE_API_KEY)")
        
        return errors
    
    def fetch_content(self, 
                     channel_id: str, 
                     limit: Optional[int] = None,
                     since: Optional[datetime] = None) -> IngestionResult:
        """
        Fetch video transcripts from a YouTube channel.
        
        Args:
            channel_id: YouTube channel ID (e.g., 'UC_x5XG1OV2P6uZZ5FSM9Ttw')
            limit: Maximum number of videos to process
            since: Only fetch videos newer than this date
            
        Returns:
            IngestionResult with video transcripts
        """
        result = self.create_ingestion_result()
        
        try:
            # Validate config
            config_errors = self.validate_config()
            if config_errors:
                for error in config_errors:
                    result.add_error(error)
                result.success = False
                return result
            
            # Get channel videos
            videos = self._get_channel_videos(channel_id, limit, since)
            
            # Process each video
            for video in videos:
                try:
                    content = self._process_video(video, channel_id)
                    if content and self.should_ingest_content(content):
                        content = self.process_content(content)
                        result.add_content(content)
                except Exception as e:
                    result.add_error(f"Failed to process video {video.get('id', 'unknown')}: {e}")
            
            self.logger.info(f"YouTube ingestion completed: {result.total_fetched} videos processed")
            
        except Exception as e:
            result.add_error(f"YouTube ingestion failed: {e}")
            result.success = False
        
        return result
    
    def _get_channel_videos(self, 
                           channel_id: str, 
                           limit: Optional[int] = None,
                           since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get videos from a YouTube channel."""
        try:
            from googleapiclient.discovery import build
            from googleapiclient.errors import HttpError
        except ImportError:
            raise ImportError("Google API client not installed. Run: pip install google-api-python-client")
        
        videos = []
        
        try:
            # Build YouTube API client
            youtube = build('youtube', 'v3', developerKey=self.api_key)
            
            # Get channel's uploads playlist
            self.rate_limiter.wait_if_needed()
            channel_response = youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()
            
            if not channel_response['items']:
                raise ValueError(f"Channel not found: {channel_id}")
            
            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Get videos from uploads playlist
            next_page_token = None
            videos_fetched = 0
            
            while True:
                self.rate_limiter.wait_if_needed()
                playlist_response = youtube.playlistItems().list(
                    part='snippet,contentDetails',
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, limit - videos_fetched) if limit else 50,
                    pageToken=next_page_token
                ).execute()
                
                for item in playlist_response['items']:
                    video_data = {
                        'id': item['contentDetails']['videoId'],
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'published_at': item['snippet']['publishedAt'],
                        'channel_title': item['snippet']['channelTitle'],
                        'thumbnail': item['snippet']['thumbnails'].get('default', {}).get('url')
                    }
                    
                    # Check date filter
                    if since:
                        published_date = datetime.fromisoformat(
                            video_data['published_at'].replace('Z', '+00:00')
                        )
                        if published_date <= since:
                            continue
                    
                    videos.append(video_data)
                    videos_fetched += 1
                    
                    if limit and videos_fetched >= limit:
                        break
                
                # Check for more pages
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token or (limit and videos_fetched >= limit):
                    break
            
            self.logger.info(f"Found {len(videos)} videos from channel {channel_id}")
            return videos
            
        except HttpError as e:
            if e.resp.status == 403:
                raise ValueError("YouTube API quota exceeded or invalid API key")
            elif e.resp.status == 404:
                raise ValueError(f"Channel not found: {channel_id}")
            else:
                raise ValueError(f"YouTube API error: {e}")
    
    def _process_video(self, video_data: Dict[str, Any], channel_id: str) -> Optional[IngestedContent]:
        """Process a single video and extract its transcript."""
        video_id = video_data['id']
        
        try:
            # Get video transcript
            transcript_text = self._get_video_transcript(video_id)
            
            if not transcript_text or len(transcript_text.strip()) < 50:
                self.logger.debug(f"Skipping video {video_id}: no sufficient transcript")
                return None
            
            # Get additional video metadata
            video_metadata = self._get_video_metadata(video_id)
            
            # Parse published date
            published_at = datetime.fromisoformat(
                video_data['published_at'].replace('Z', '+00:00')
            )
            
            # Create content object
            content = IngestedContent(
                external_id=video_id,
                title=video_data['title'],
                content=transcript_text,
                content_type=ContentType.YOUTUBE_TRANSCRIPT,
                url=f"https://www.youtube.com/watch?v={video_id}",
                author=video_data['channel_title'],
                published_at=published_at,
                source_id=channel_id,
                source_type="youtube",
                metadata={
                    'description': video_data['description'][:1000],  # Truncate long descriptions
                    'thumbnail': video_data.get('thumbnail'),
                    'channel_id': channel_id,
                    **video_metadata
                }
            )
            
            self.logger.debug(f"Processed video: {video_id} - {video_data['title'][:50]}...")
            return content
            
        except Exception as e:
            self.logger.warning(f"Failed to process video {video_id}: {e}")
            return None
    
    def _get_video_transcript(self, video_id: str) -> Optional[str]:
        """Get transcript for a YouTube video."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
        except ImportError:
            raise ImportError("YouTube transcript API not installed. Run: pip install youtube-transcript-api")
        
        try:
            # Try to get transcript (prefer English, but accept any language)
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try English first
            try:
                transcript = transcript_list.find_transcript(['en'])
            except NoTranscriptFound:
                # Fall back to any available transcript
                transcript = transcript_list.find_generated_transcript(['en'])
            
            # Get the transcript data
            transcript_data = transcript.fetch()
            
            # Combine all transcript segments
            transcript_text = ' '.join([item['text'] for item in transcript_data])
            
            # Clean up the transcript
            transcript_text = transcript_text.replace('\n', ' ')
            transcript_text = ' '.join(transcript_text.split())  # Remove extra whitespace
            
            return transcript_text
            
        except (TranscriptsDisabled, NoTranscriptFound):
            self.logger.debug(f"No transcript available for video {video_id}")
            return None
        except Exception as e:
            self.logger.debug(f"Failed to get transcript for video {video_id}: {e}")
            return None
    
    def _get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Get additional metadata for a video."""
        try:
            from googleapiclient.discovery import build
        except ImportError:
            return {}
        
        try:
            youtube = build('youtube', 'v3', developerKey=self.api_key)
            
            self.rate_limiter.wait_if_needed()
            response = youtube.videos().list(
                part='statistics,contentDetails',
                id=video_id
            ).execute()
            
            if not response['items']:
                return {}
            
            video = response['items'][0]
            statistics = video.get('statistics', {})
            content_details = video.get('contentDetails', {})
            
            return {
                'view_count': int(statistics.get('viewCount', 0)),
                'like_count': int(statistics.get('likeCount', 0)),
                'comment_count': int(statistics.get('commentCount', 0)),
                'duration': content_details.get('duration'),
                'definition': content_details.get('definition'),
                'caption': content_details.get('caption') == 'true'
            }
            
        except Exception as e:
            self.logger.debug(f"Failed to get metadata for video {video_id}: {e}")
            return {}
    
    def should_ingest_content(self, content: IngestedContent) -> bool:
        """YouTube-specific content filtering."""
        # Use base filtering first
        if not super().should_ingest_content(content):
            return False
        
        # Skip very short transcripts (likely low quality)
        if len(content.content) < 100:
            return False
        
        # Skip videos that are too old (optional - can be configured)
        if content.published_at:
            days_old = (datetime.now(timezone.utc) - content.published_at).days
            if days_old > 365:  # Skip videos older than 1 year
                self.logger.debug(f"Skipping old video: {content.external_id}")
                return False
        
        return True


def get_channel_id_from_url(url: str) -> Optional[str]:
    """Extract channel ID from various YouTube URL formats."""
    import re
    
    # Match different YouTube URL patterns
    patterns = [
        r'youtube\.com/channel/([a-zA-Z0-9_-]+)',
        r'youtube\.com/c/([a-zA-Z0-9_-]+)',
        r'youtube\.com/user/([a-zA-Z0-9_-]+)',
        r'youtube\.com/@([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def get_channel_info(channel_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Get basic information about a YouTube channel."""
    try:
        from googleapiclient.discovery import build
        
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        response = youtube.channels().list(
            part='snippet,statistics',
            id=channel_id
        ).execute()
        
        if not response['items']:
            return None
        
        channel = response['items'][0]
        snippet = channel['snippet']
        statistics = channel.get('statistics', {})
        
        return {
            'id': channel['id'],
            'title': snippet['title'],
            'description': snippet['description'],
            'url': f"https://www.youtube.com/channel/{channel['id']}",
            'subscriber_count': int(statistics.get('subscriberCount', 0)),
            'video_count': int(statistics.get('videoCount', 0)),
            'view_count': int(statistics.get('viewCount', 0)),
            'thumbnail': snippet['thumbnails'].get('default', {}).get('url')
        }
        
    except Exception as e:
        logger.error(f"Failed to get channel info for {channel_id}: {e}")
        return None 