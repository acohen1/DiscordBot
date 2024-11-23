import threading
from core.config import GIPHY_API_KEY, MAX_SEARCH_RESULTS
from typing import List, Optional, Tuple
from clients.openai_client import OpenAIClient
from processors.img import ImageProcessor
import aiohttp
from bs4 import BeautifulSoup
import logging
import re

logger = logging.getLogger("GifProcessor")

class GIFProcessor:
    _instance = None
    _lock = threading.Lock()  # Lock object to ensure thread safety

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of GIFProcessor is created, even in a multithreaded context."""
        if cls._instance is None:
            with cls._lock:  # Lock this section to prevent race conditions
                if cls._instance is None:
                    cls._instance = super(GIFProcessor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the GIFProcessor. Prevent reinitialization by checking an attribute."""
        if not hasattr(self, "initialized"):
            # Other init methods....

            self.openai_client = OpenAIClient.get_instance()
            self.img_processor = ImageProcessor()
            self.initialized = True    # Mark this instance as initialized

    async def search_by_url(self, url: str) -> str:
        """Process a Giphy URL and format the response.
        Args:
            url (str): The Giphy URL to process
        Returns:
            str: The formatted description of the Giphy GIF.
        """
        logger.info(f"Searching GIF '{url}'...")
        # Extract the GIF URL and title from the page
        gif_url, gif_title = await self._extract_gif_data_from_page(url)
        
        # Process first frame for a description
        frame_description = await self.img_processor.describe_image(gif_url, is_gif=True)

        gif = {
            "title": gif_title,
            "url": gif_url,
            "description": frame_description
        }
        _, message_to_cache = self._format_gif_message(gif)
        return message_to_cache

    async def _extract_gif_data_from_page(self, page_url: str) -> Tuple[str, str]:
        """Extract the direct GIF URL from a Giphy page.
        Args:
            page_url (str): The URL of the Giphy page.
        Returns:
            Tuple[str, str]: The GIF URL and the title of the GIF.
        """
        try:
            # Fetch the page content
            async with aiohttp.ClientSession() as session:
                headers = {'User-Agent': 'Mozilla/5.0'}
                async with session.get(page_url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch page content: HTTP {response.status}")
                        return None

                    # Parse the page content
                    page_content = await response.text()
                    soup = BeautifulSoup(page_content, 'html.parser')

                    # Extract the GIF URL
                    gif_url = (
                        soup.find("meta", property="og:image") or
                        soup.find("link", rel="image_src")
                    )
                    gif_url = gif_url.get("content") if gif_url and gif_url.has_attr("content") else gif_url.get("href") if gif_url else None

                    # Extract the title
                    title = soup.title.string.strip() if soup.title else "No title"
                    title = re.sub(r"\s*-\s*Discover\s*&\s*Share\s*GIFs", "", title, flags=re.IGNORECASE).strip()   # Remove uneccessary text from title

                    if gif_url:
                        return gif_url.strip(), title
                    else:
                        logger.warning("GIF URL not found in the page.")
                        return "No URL", title
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
            return "No URL", "No title"
    
    def _format_gif_message(self, gif: dict) -> Tuple[str, str]:
        """Format the selected video into a message to send and cache.
        Args:
            video (dict): The video dictionary containing video details.
        Returns:
            Tuple[str, str]: A tuple containing the message to send and cache.
        """
        message_to_send = gif.get("url")
        message_to_cache = f"[GIF ::: {gif.get("title")} ::: {gif.get("description")}]"
        return message_to_send, message_to_cache