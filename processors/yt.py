import threading
from core.config import GOOGLE_API_KEY, MAX_SEARCH_RESULTS
from googleapiclient.discovery import build
from typing import List, Optional, Tuple
from clients.openai_client import OpenAIClient
from processors.img import ImageProcessor
import logging
import re
import asyncio

# Add the ability to prioritize getting videos from:
# 1. tonebone https://www.youtube.com/@tonebone740
# 2. Gregg Limper https://www.youtube.com/@gregglimper864
# 3. Brainfreeeeeze https://www.youtube.com/@Brainfreezzzzz


# Youtube transcript processing to add on to the youtube description.
# We can get the transcript -> summarize with OpenAI -> add to the description.

logger = logging.getLogger("YouTubeProcessor")

class YouTubeProcessor:
    _instance = None
    _lock = threading.Lock()    # Lock object to ensure thread safety

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of YouTubeProcessor is created, even in a multithreaded context."""
        if cls._instance is None:
            with cls._lock:    # Lock this section to prevent race conditions
                if cls._instance is None:   # Double-check inside the lock
                    cls._instance = super(YouTubeProcessor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the YouTubeProcessor. Prevent reinitialization by checking an attribute."""
        if not hasattr(self, "initialized"):
            # Other init methods....

            self.openai_client = OpenAIClient.get_instance()
            self.img_processor = ImageProcessor()
            self.youtube_client = build("youtube", "v3", developerKey=GOOGLE_API_KEY)
            self.initialized = True    # Mark this instance as initialized

    # ------------------ YOUTUBE URL PROCESSING ------------------

    async def search_by_url(self, url: str) -> str:
        """Process a YouTube URL and format the response.
        Args:
            url (str): The YouTube video URL to process
        Returns:
            str: The formatted description of the YouTube video.
        """
        logger.info(f"Searching YouTube for '{url}'...")
        video = await self._get_video_details(url)
        if not video:
            logger.error("Failed to search YouTube.")
            return None
        
        # Process thumbnail for a description and add it to the video dictionary
        thumbnail_url = video.get("thumbnail_url")
        thumbnail_description = await self.img_processor.describe_image(thumbnail_url) if thumbnail_url else "No Thumbnail Description Available"
        if thumbnail_description:
            thumbnail_description = await self.openai_client.text_summarizer(thumbnail_description)
        video["thumbnail_description"] = thumbnail_description

        # Also summarize the video's description
        description = video.get("description")
        if description:
            description = await self.openai_client.text_summarizer(description)
            video["description"] = description

        _, message_to_cache = self._format_video_message(video)
        return message_to_cache
        
    async def _get_video_details(self, url: str) -> Optional[dict]:
        """Fetch video details from a YouTube URL.
        Args:
            url (str): The YouTube URL to fetch details from.
        Returns:
            Optional[dict]: A dictionary containing video details, or None if failed. 
        """
        try:
            video_id_match = re.search(r"(?:v=|/)([A-Za-z0-9_-]{11})", url)
            if not video_id_match:
                logger.error(f"Invalid YouTube URL format: {url}")
                return None
            video_id = video_id_match.group(1)

            request = self.youtube_client.videos().list(part="snippet", id=video_id)
            response = request.execute()
            items = response.get("items", [])
            if items:
                snippet = items[0].get("snippet", {})
                return {
                    "video_id": video_id,
                    "title": snippet.get("title"),
                    "author": snippet.get("channelTitle"),
                    "description": snippet.get("description"),
                    "thumbnail_url": snippet.get("thumbnails", {}).get("default", {}).get("url"),
                    "published_at": snippet.get("publishedAt")
                }
            else:
                logger.warning(f"No YouTube results found for URL: {url}")
                return None
        except Exception as e:
            logger.error(f"Error fetching YouTube video details: {str(e)}")
            return None        
        
    # ------------------ YOUTUBE KEYWORD SEARCH ------------------

    async def search_by_keyword(self, keyword: str, oai_messages: List[dict]) -> Tuple[str, str]:
        """
        Process a YouTube keyword search and select the most relevant video based on the given OpenAI messages.
        
        Args:
            keyword (str): The keyword to search for.
            oai_messages (List[dict]): The user's GLThread converted to OpenAI messages.
            
        Returns:
            Tuple[str, str]: A tuple containing the message to send and cache.
        """
        # 1. Search YouTube for the keyword
        logger.info(f"Searching YouTube for '{keyword}'...")
        videos = await self._search_youtube(keyword)
        if not videos:
            logger.error("Failed to search YouTube.")
            return None
        
        logger.info(f"Found {len(videos)} videos.")

        # 2. Process thumbnails and descriptions in parallel
        # Process thumbnail descriptions
        thumbnail_tasks = [
            self.img_processor.describe_image(video.get("thumbnail_url"))
            if video.get("thumbnail_url")
            else self._async_no_thumbnail_description()
            for video in videos
        ]
        # Process video descriptions
        description_tasks = [
            self.openai_client.text_summarizer(video.get("description"))
            if video.get("description")
            else self._async_no_video_description()
            for video in videos
        ]

        # Wait for all tasks to complete
        thumbnail_descriptions = await asyncio.gather(*thumbnail_tasks)
        summarized_descriptions = await asyncio.gather(*description_tasks)

        # Update videos with results
        for video, thumbnail_desc, summarized_desc in zip(videos, thumbnail_descriptions, summarized_descriptions):
            video["thumbnail_description"] = thumbnail_desc
            video["description"] = summarized_desc

        # 3. Process and select the most relevant video based on the OpenAI messages
        concat_descriptions = [f"{video.get('description')} {video.get('thumbnail_description')}" for video in videos]
        best_video_idx = await self.openai_client.select_most_relevant_media(keyword, concat_descriptions, oai_messages)
        best_video = videos[best_video_idx]

        # 4. Format the selected video message to send and cache
        return self._format_video_message(best_video)

    # Helper methods for fallbacks
    async def _async_no_thumbnail_description(self):
        return "No Thumbnail Description Available"

    async def _async_no_video_description(self):
        return "No Description Available"


    async def _search_youtube(self, keyword: str) -> List[dict]:
        """Search YouTube and return a list of video details.
        Args:
            query (str): The search query to use.
        Returns:
            List[dict]: A list of video dictionaries containing video details.
        """
        try:
            request = self.youtube_client.search().list(
                part="snippet",
                maxResults=MAX_SEARCH_RESULTS,
                q=keyword,
                type="video"
            )
            response = request.execute()
            items = response.get("items", [])
            videos = []
            for item in items:
                video_id = item.get("id", {}).get("videoId")
                snippet = item.get("snippet", {})
                videos.append({
                    "video_id": video_id,
                    "title": snippet.get("title"),
                    "author": snippet.get("channelTitle"),
                    "description": snippet.get("description"),
                    "thumbnail_url": snippet.get("thumbnails", {}).get("default", {}).get("url"),
                    "published_at": snippet.get("publishedAt")
                })
            return videos
        except Exception as e:
            logger.error(f"Error searching YouTube: {str(e)}")
            return []

    # ------------------ YOUTUBE FORMATTING ------------------

    def _format_video_message(self, video: dict) -> Tuple[str, str]:
        """Format the selected video into a message to send and cache.
        Args:
            video (dict): The video dictionary containing video details.
        Returns:
            Tuple[str, str]: A tuple containing the message to send and cache.
        """
        title = video.get("title") if video.get("title") else "No title available"
        author = video.get("author") if video.get("author") else "No author available"
        description = video.get("description")
        thumbnail_description = video.get("thumbnail_description") if video.get("thumbnail_description") else "No thumbnail description available"
        url = f"https://www.youtube.com/watch?v={video.get('video_id')}" if video.get("video_id") else "No URL available"

        message_to_send = f"[{self.sanitize_text(title)}]({url})"
        message_to_cache = f"[YouTube ::: {title} ::: {author} ::: {description} ::: Thumbnail Description: {thumbnail_description}]"
        return message_to_send, message_to_cache
    
    def sanitize_text(self, text: str) -> str:
        """Remove any character that could break Markdown formatting."""
        return re.sub(r'[^\w\s.,!?:;\'"-]', '', text)