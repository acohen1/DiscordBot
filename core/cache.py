import logging
from datetime import datetime, timezone, timedelta
from models.threads import GLThread, GLMessage
from core.config import CACHE_CONVERSATIONS_LEN, CACHE_CONVERSATIONS_TIMELIMIT_MINS
from processors.msg import MessageProcessor
from clients.discord_client import DiscordClient
import discord
from clients.openai_client import OpenAIClient
import threading
from typing import List, Optional

logger = logging.getLogger("GLCache")

class GLCache:
    _instance = None
    _lock = threading.Lock()  # Lock for thread-safe singleton

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = super(GLCache, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):  # Ensure __init__ runs only once
            self.threads = {}
            self.openai_client = OpenAIClient()
            self.discord_client = DiscordClient().client
            self.message_processor = MessageProcessor()
            self.initialized = True  # Mark the instance as initialized

    def __str__(self):
        return f"GLCache(threads={self.threads})"

    def __iter__(self):
        return iter(self.threads.values())
    
    async def init_threads(self, channels: List[discord.TextChannel]) -> bool:
        logger.info("Initializing threads...")
        curr_time = datetime.now(timezone.utc)
        time_threshold = timedelta(minutes=CACHE_CONVERSATIONS_TIMELIMIT_MINS)

        for channel in channels:
            async for message in channel.history(limit=CACHE_CONVERSATIONS_LEN):
                # 1. Skip messages older than the time threshold
                if curr_time - message.created_at > time_threshold:
                    continue

                # 2. Process the message
                gl_message = await self.add_discord_message(message)
                if not gl_message:
                    logger.error(f"Empty message for user {message.author.name}. Skipping.")
                    continue
                
        # Threads initialized successfully
        for thread in self.threads.values():
            logger.info(f"Threads:\n_________________\n{thread}\n_________________\n")

        logger.info("Threads initialized.")
        return True 

    async def add_discord_message(self, message: discord.Message) -> Optional[GLMessage]:
        """Send a discord message to the GLCache and OAI Assistant thread.
        Args:
            message (discord.Message): The discord message to send.
        Returns:
            Optional[GLMessage]: The GLMessage object created from the discord message if successful, None otherwise.
        """

        # 1. Ensure thread for all potential users in the message exist
        user_ids_to_check = {message.author.id}  # Start with the message author
        user_ids_to_check.update(user.id for user in message.mentions)  # Add all mentioned users
        for user_id in user_ids_to_check:
            if user_id not in self.threads:
                self.threads[user_id] = GLThread(user_id, max_history_length=CACHE_CONVERSATIONS_LEN)
            
        # 2. Process the message
        gl_message = await self.message_processor.discord_to_GLMessage(message)
        if not gl_message or not gl_message.content.strip():
            logger.error(f"No message for user {message.author.name}.")
            return None
        
        # Track if message has already been added to the bot's thread
        added_to_bot_thread = False

        # 3. Handle direct replies
        reference_message = message.reference.resolved if message.reference else None
        if reference_message and reference_message.author.id != self.discord_client.user.id:
            # Add to the referenced user's thread
            target_thread = self.threads.get(reference_message.author.id)
            if not target_thread.add_message(gl_message):
                logger.error(f"Failed to add message {gl_message.message_id} to conversation for user {reference_message.author.name}.")
                return None
        elif reference_message and reference_message.author.id == self.discord_client.user.id:
            # Add to the bot's thread
            target_thread = self.threads.get(self.discord_client.user.id)
            if not target_thread.add_message(gl_message):
                logger.error(f"Failed to add message {gl_message.message_id} to conversation for the bot (direct reply).")
                return None
            added_to_bot_thread = True

        # 4. Handle messages mentioning the bot (triggering the bot's response)
        if not added_to_bot_thread and self.discord_client.user.mentioned_in(message):
            # Add the message to the bot's thread
            target_thread = self.threads.get(self.discord_client.user.id)
            if not target_thread.add_message(gl_message):
                logger.error(f"Failed to add message {gl_message.message_id} to conversation for the bot.")
                return None
            added_to_bot_thread = True
            
        # 5. Add the message to the author's thread
        target_thread = self.threads.get(message.author.id)
        if not target_thread.add_message(gl_message):
            logger.error(f"Failed to add message {gl_message.message_id} to conversation for user {message.author.name}.")
            return None
        
        # Message added successfully
        logger.info(f"Added message to GLCache: {message.author.name} - {gl_message.content[:50]}")
        return gl_message