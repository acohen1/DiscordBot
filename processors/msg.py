import threading
from models.threads import GLMessage, GLThread
from processors.yt import YouTubeProcessor
from processors.gif import GIFProcessor
from processors.img import ImageProcessor
from processors.web import WebProcessor
from datetime import timezone
import logging
import discord
from clients.discord_client import DiscordClient
from clients.openai_client import OpenAIClient
import re
from typing import List, Dict, Union
logger = logging.getLogger('AsyncOpenAI')

class MessageProcessor:
    _instance = None
    _lock = threading.Lock()  # Lock object to ensure thread safety

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of OpenAIClient is created, even in a multithreaded context."""
        if cls._instance is None:
            with cls._lock:  # Lock this section to prevent race conditions
                if cls._instance is None:  # Double-check inside the lock
                    cls._instance = super(MessageProcessor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the OpenAIClient. Prevent reinitialization by checking an attribute."""
        if not hasattr(self, "initialized"):
            self.discord_client = DiscordClient().client
            self.openai_client = OpenAIClient.get_instance()

            # Initialize all media processors needed for message processing
            self.img_processor = ImageProcessor()
            self.yt_processor = YouTubeProcessor()
            self.gif_processor = GIFProcessor()
            self.web_processor = WebProcessor()

            self.initialized = True  # Mark this instance as initialized

    async def discord_to_GLMessage(self, message: discord.Message) -> GLMessage:
        """Convert a discord.Message object to a GLMessage object.
        Args:
            message (discord.Message): The discord message to convert.
            target_message_id (Optional[int]): The message ID that triggered the bot response.
        Returns:
            GLMessage: The converted GLMessage object.
        """
        processed_message = message.content

        # 1. Replace mentions with the users' name
        processed_message = self._replace_mentions(processed_message)

        # 2. Process links
        processed_message = await self._process_links(processed_message)

        # 3. Process image content
        for attachment in message.attachments:
            if 'image' in attachment.content_type:
                image_url = attachment.url
                image_description = await self.img_processor.describe_image(image_url)
                processed_message += f" [Image ::: {image_description}]"
        
        # TODO: Monitor effect on assistant replies
        # 4. Prefix the message with the user's name if it's a user message
        if message.author.id != self.discord_client.user.id:
            processed_message = f"{message.author.display_name}: {processed_message}"
        
        return GLMessage(
            role='assistant' if message.author.id == self.discord_client.user.id else 'user',
            content=processed_message,
            timestamp=message.created_at.replace(tzinfo=timezone.utc),
            message_id=message.id,
            target_message_id=message.reference.message_id if message.reference else None
        )

    def _replace_mentions(self, message: str):
        """Replace mentions in a message with the users' name."""
        bot_id = self.discord_client.user.id
        guild = self.discord_client.guilds[0]

        def replace(match):
            mention_id = int(match.group(1) or match.group(2) or match.group(3))
            if mention_id == bot_id:
                return ""
            elif match.group(1):
                user = guild.get_member(mention_id)
                return user.display_name if user else "<Unknown User>"
            elif match.group(2):
                role = guild.get_role(mention_id)
                return role.name if role else "<Unknown Role>"
            elif match.group(3):
                channel = guild.get_channel(mention_id)
                return channel.name if channel else "<Unknown Channel>"
            
        return re.sub(r"<@!?(\d+)>|<@&(\d+)>|<#(\d+)>", replace, message)
    
    async def _process_links(self, message: str) -> str:
        processed_message = message

        # Define bot-specific patterns (formatted links)
        bot_patterns = {
            "youtube": re.compile(r"\[(?!.*:::).*?]\(<?(https?://(?:www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+)>?\)"),
            "gif": re.compile(r"\[(?!.*:::).*?]\(<?(https?://(?:\S+\.)?(?:giphy|tenor)\.com/\S+)>?\)"),
            "general": re.compile(r"\[(?!.*:::).*?]\(<?(https?://\S+)>?\)")
        }

        # Define user-generated (raw) link patterns
        raw_patterns = {
            "youtube": re.compile(r"(https?://(?:www\.)?(youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11}))[^\s>]*"),
            "gif": re.compile(r"(https?://(?:\S+\.)?(?:giphy|tenor)\.com/\S+)"),
            "general": re.compile(r"(https?://\S+)")
        }

        # Helper function to process links based on type
        async def match_and_process_link(url, link_type):
            if link_type == "youtube":
                return await self.yt_processor.search_by_url(url)
            elif link_type == "gif":
                return await self.gif_processor.search_by_url(url)
            elif link_type == "general":
                return await self.web_processor.search_by_url(url)
            logger.warning(f"Unrecognized link type: {link_type}")
            return None, f"[Website ::: No Title ::: No Description Available]"

        # Apply patterns in prioritized order to avoid re-processing
        processed_urls = set()  # Track already-processed URLs

        for link_type, pattern in {**bot_patterns, **raw_patterns}.items():
            for match in pattern.finditer(processed_message):
                url = match.group(1)
                if url in processed_urls:
                    continue  # Skip already-processed links

                message_to_cache = await match_and_process_link(url, link_type)
                processed_message = processed_message.replace(match.group(0), message_to_cache)
                processed_urls.add(url)  # Mark this URL as processed

        return processed_message
    
    async def GLThread_to_OAI(self, thread: GLThread) -> List[Dict]:
        """Convert a GLThread object to a format suitable for the OpenAI API."""
        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in thread.get_conversation_messages()
        ]