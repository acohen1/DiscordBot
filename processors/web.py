import threading
from core.config import GOOGLE_API_KEY, MAX_SEARCH_RESULTS, SEARCH_ENGINE_ID
from typing import List, Tuple, Dict
from clients.openai_client import OpenAIClient
import aiohttp
from bs4 import BeautifulSoup
import logging
import asyncio
from duckduckgo_search import AsyncDDGS

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

    # ------------------ WEB URL PROCESSING ------------------

    async def search_by_url(self, url: str) -> str:
        """Extract and format content from a direct link to a webpage.
        Args:
            url (str): The URL of the webpage to process.
        Returns:
            str: The formatted description of the webpage.
        """
        logger.info(f"Searching webpage '{url}'...")
        title, description, page_content = await self._extract_web_data_from_page(url)
        url_description = await self.openai_client.link_summarizer(url)
        page_content_summarized = await self.openai_client.text_summarizer(page_content)

        website = {
            "title": title,
            "snippet": description,
            "page_content": page_content_summarized,
            "url": url,
            "url_description": url_description
        }
        _, message_to_cache = self._format_website_message(website)
        return message_to_cache

    async def _extract_web_data_from_page(self, url: str) -> Tuple[str, str, str]:
        """Fetch the title, description, and main content of a webpage.
        Args:
            url (str): The URL to fetch data from.
        Returns:
            Tuple[str, str, str]: The title, description, and main content of the page.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch page content: {response.status}")
                        return "No Title Available", "No Description Available", "No Content Available"
                    
                    content = await response.text()
                    soup = BeautifulSoup(content, "html.parser")

                    # Extract the title
                    title_tag = soup.find("title")
                    title = title_tag.string.strip() if title_tag and title_tag.string else "No Title Available"

                    # Extract the meta description
                    description_meta = (
                        soup.find('meta', attrs={'name': 'description'}) or
                        soup.find('meta', attrs={'property': 'og:description'}) or
                        soup.find('meta', attrs={'name': 'og:description'})
                    )
                    description = description_meta.get('content').strip() if description_meta and description_meta.get('content') else "No Description Available"

                    # Extract main page content (e.g., <p> tags)
                    paragraphs = soup.find_all("p")
                    list_items = soup.find_all("li")

                    # Combine text from paragraphs and list items
                    page_content = "\n".join(
                        p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
                    ) + "\n" + "\n".join(
                        li.get_text(strip=True) for li in list_items if li.get_text(strip=True)
                    )

                    # Fallback if no readable content
                    if not page_content.strip():
                        page_content = "No Content Available"

                    return title, description, page_content
        except Exception as e:
            logger.error(f"Failed to fetch link data: {e}")
            return "No Title Available", "No Description Available", "No Content Available"

    # ------------------ WEB KEYWORD SEARCH ------------------

    async def search_by_keyword(self, keyword: str, oai_messages: List[Dict]) -> Tuple[str, str]:
        logger.info(f"Searching Google for: {keyword}")
        results = await self._search_duckduckgo(keyword)
        if not results:
            return "No results found.", "No results found."
        
        # Extract page content and combine with snippets
        web_data = []
        for result in results:
            url, title, snippet = result
            _, _, page_content = await self._extract_web_data_from_page(url)
            combined_description = f"{snippet} {page_content}"
            web_data.append({
                "url": url,
                "title": title,
                "snippet": snippet,
                "page_content": page_content,
                "combined_description": combined_description
            })

        # Use the combined descriptions to select the most relevant result
        web_descriptions = [data["combined_description"] for data in web_data]
        best_web_idx = await self.openai_client.select_most_relevant_media(keyword, web_descriptions, oai_messages)
        best_web = web_data[best_web_idx]

        logger.info(f"Selected website: {best_web['title']}")
        website = {
            "title": best_web["title"],
            "description": best_web["snippet"],
            "page_content": best_web["page_content"],
            "url": best_web["url"],
            "url_description": await self.openai_client.link_summarizer(best_web["url"])
        }
        return self._format_website_message(website)

    async def _search_duckduckgo(self, keyword: str) -> List[Dict]:
        """Search DuckDuckGo for a keyword and return the top results.
        Args:
            keyword (str): The keyword to search for.
        Returns:
            List[Dict]: A list of dictionaries containing the URL, title, and description of each result.
        """
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            results = await AsyncDDGS(headers=headers).atext(keyword, max_results=MAX_SEARCH_RESULTS)
            return [(result["href"], result["title"], result.get("body", "No description available")) for result in results]
        except Exception as e:
            logger.error(f"Failed to search by keyword: {e}")
            return []

    # ------------------ WEB FORMATTING ------------------

    def _format_website_message(self, web: dict) -> Tuple[str, str]:
        """Format the extracted website data into a message.
        Args:
            website (dict): The extracted website data.
        Returns:
            Tuple[str, str]: A tuple containing the message to send and the message to cache.
        """
        title = web.get("title")
        description = web.get("snippet")
        page_content = web.get("page_content")
        url = web.get("url")
        url_description = web.get("url_description")

        message_to_send = f"[{title}]({url})"
        message_to_cache = f"[Website ::: {title} ::: {description} ::: {page_content} ::: {url_description}]"
        return message_to_send, message_to_cache