import threading
from core.config import GOOGLE_API_KEY, MAX_SEARCH_RESULTS
from typing import List, Tuple
from clients.openai_client import OpenAIClient
import aiohttp
from bs4 import BeautifulSoup
import logging
import re

logger = logging.getLogger("WebProcessor")

class WebProcessor:
    _instance = None
    _lock = threading.Lock()  # Lock object to ensure thread safety

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of WebProcessor is created, even in a multithreaded context."""
        if cls._instance is None:
            with cls._lock:  # Lock this section to prevent cade conditions
                if cls._instance is None:
                    cls._instance = super(WebProcessor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the WebProcessor. Prevent reinitialization by checking an attribute."""
        if not hasattr(self, "initialized"):
            # Other init methods....

            self.openai_client = OpenAIClient.get_instance()
            self.initialized = True    # Mark this instance as initialized

    async def search_by_url(self, url: str) -> str:
        """Extract and format content from a direct link to a webpage.
        Args:
            url (str): The URL of the webpage to process.
        Returns:
            str: The formatted description of the webpage.
        """
        logger.info(f"Searching webpage '{url}'...")
        title, description = await self._extract_web_data_from_page(url)
        url_description = await self.openai_client.link_summarizer(url)

        website = {
            "title": title,
            "description": description,
            "url": url,
            "url_description": url_description
        }
        _, message_to_cache = self._format_website_message(website)
        return message_to_cache


    async def _extract_web_data_from_page(self, url: str) -> Tuple[str, str]:
        """Fetch the title and description of a webpage.
        Args:
            url (str): The URL to fetch data from.
        Returns:
            Tuple[str, str]: The title and description of the page.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch page content: {response.status}")
                        return None, None
                    content = await response.text()
                    soup = BeautifulSoup(content, "html.parser")
                    
                    title_tag = soup.find("title")
                    title = title_tag.string.strip() if title_tag and title_tag.string else "No Title Available"

                    description_meta = (
                        soup.find('meta', attrs={'name': 'description'}) or
                        soup.find('meta', attrs={'property': 'og:description'}) or
                        soup.find('meta', attrs={'name': 'og:description'})
                    )
                    description = description_meta.get('content').strip() if description_meta and description_meta.get('content') else "No Description Available"
                    
                    return title, description
        except Exception as e:
            logger.error(f"Failed to fetch link data: {e}")
            return None, None
        
    def _format_website_message(self, web: dict) -> Tuple[str, str]:
        """Format the extracted website data into a message.
        Args:
            website (dict): The extracted website data.
        Returns:
            Tuple[str, str]: A tuple containing the message to send and the message to cache.
        """
        title = web.get("title")
        description = web.get("description")
        url = web.get("url")
        url_description = web.get("url_description")

        message_to_send = f"[{title}]({url})"
        message_to_cache = f"[Website ::: {title} ::: {description} ::: {url_description}]"
        return message_to_send, message_to_cache