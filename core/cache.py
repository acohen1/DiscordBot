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
        async def collect_thread_replies(message: discord.Message):
            """Recursively collect all messages in a thread."""
            thread_messages = []
            mention_ids = set()
        
            while message:
                gl_message = await self.message_processor.discord_to_GLMessage(message)
                if gl_message and gl_message.content.strip():
                    thread_messages.append((message.author, gl_message))
                    mention_ids.update(user.id for user in message.mentions)
                # Move to the parent message if it exists
                message = message.reference.resolved if message.reference else None
            return list(reversed(thread_messages)), mention_ids  # Reverse to get chronological order

        # Step 1: Collect all messages in the thread
        full_thread, mention_ids = await collect_thread_replies(message)
        if not full_thread:
            logger.error("Failed to collect thread for message.")
            return None

        # Step 2: Identify all participating users
        participating_user_ids = set()
        for author, _ in full_thread:
            participating_user_ids.add(author.id)
        # Add mentioned users from the collected thread
        participating_user_ids.update(mention_ids)
        # Add mentions in the current message
        participating_user_ids.update(user.id for user in message.mentions)

        # Step 3: Ensure threads exist for all participants
        for user_id in participating_user_ids:
            if user_id not in self.threads:
                self.threads[user_id] = GLThread(user_id, max_history_length=CACHE_CONVERSATIONS_LEN)

       # Step 4: Add all messages in the thread to each participant's thread
        for author, gl_message in full_thread:
            for user_id in participating_user_ids:
                target_thread = self.threads.get(user_id)
                # Check for duplicates before adding
                if not target_thread.contains_message(gl_message.message_id):
                    if not target_thread.add_message(gl_message):
                        logger.error(f"Failed to add message {gl_message.message_id} to conversation for user {user_id}.")
                        return None
                else:
                    logger.debug(f"Message {gl_message.message_id} already exists in conversation for user {user_id}.")

        # Step 5: Log success
        logger.info(f"Added full thread to all participants' threads: {[m[1].content[:50] for m in full_thread]}")
        return full_thread[-1][1]  # Return the GLMessage for the original message
