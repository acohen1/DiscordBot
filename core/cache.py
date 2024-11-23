import logging
from datetime import datetime, timezone, timedelta
from models.threads import GLThread, GLMessage
from core.config import CACHE_CONVERSATIONS_LEN, CACHE_CONVERSATIONS_TIMELIMIT_MINS
from processors.msg import MessageProcessor
from clients.discord_client import DiscordClient
import discord
from clients.openai_client import OpenAIClient
import threading
from typing import List, Optional, Set
import asyncio

# TODO: FIX THREAD INITIALIZATION FOR PROPER CHRONOLOGICAL ORDER
# TODO: WE WILL ADD A 'TARGET_MESSAGE_ID' TO THE GLMESSAGE OBJECT TO TRACK THE MESSAGE THAT TRIGGERED THE BOT RESPONSE

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
        if not await self._ensure_threads_exist(message):
            logger.error(f"Failed to ensure threads exist for message {message.id}.")
            return None
            
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
            if not await self._add_to_threads(gl_message, self.threads.get(reference_message.author.id)):
                logger.error(f"Failed to add message {gl_message.message_id} to conversation for user {reference_message.author.name}.")
                return None
        elif reference_message and reference_message.author.id == self.discord_client.user.id:
            # Add to the bot's thread
            if not await self._add_to_threads(gl_message, self.threads.get(self.discord_client.user.id)):
                logger.error(f"Failed to add message {gl_message.message_id} to conversation for the bot (direct reply).")
                return None
            added_to_bot_thread = True

        # 4. Handle messages mentioning the bot (triggering the bot's response)
        if not added_to_bot_thread and self.discord_client.user.mentioned_in(message):
            # Add the message to the bot's thread
            if not await self._add_to_threads(gl_message, self.threads.get(self.discord_client.user.id)):
                logger.error(f"Failed to add message {gl_message.message_id} to conversation for the bot.")
                return None
            added_to_bot_thread = True
            
        # 6. Add the message to the author's thread
        if not await self._add_to_threads(gl_message, self.threads.get(message.author.id)):
            logger.error(f"Failed to add message {gl_message.message_id} to conversation for user {message.author.name}.")
            return None
        
        # Message added successfully
        logger.info(f"Added {message.author.name} message to GLCache and OAI Assistant thread.")
        return gl_message

    async def _ensure_threads_exist(self, message: discord.Message) -> bool:
        """Ensure the GLCache and assistant threads exist for all users mentioned in the given message."""
        user_ids_to_check = {message.author.id}  # Start with the message author
        user_ids_to_check.update(user.id for user in message.mentions)  # Add all mentioned users

        tasks = []  # Prepare async tasks for creating threads
        for user_id in user_ids_to_check:
            if user_id not in self.threads:
                tasks.append(self._create_thread_for_user(user_id))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for user_id, result in zip(user_ids_to_check, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to create thread for user {user_id}: {result}")
                    continue

        return True

    async def _create_thread_for_user(self, user_id: int) -> bool:
        """Create a thread for the given user and add it to GLCache."""
        try:
            asst_id = await self.openai_client.create_asst_thread(user_id)
            self.threads[user_id] = GLThread(user_id, asst_id, max_history_length=CACHE_CONVERSATIONS_LEN)
            return True
        except Exception as e:
            logger.error(f"Exception while creating thread for user {user_id}: {e}")
            return False
        
    async def _add_to_threads(self, gl_message: GLMessage, target_thread: GLThread) -> bool:
        """Add a GLMessage to the target GLCache and OAI Assistant thread 
        """        
        # Add the message to the target thread
        if not target_thread.add_message(gl_message):
            logger.error(f"Failed to add GLMessage {gl_message.message_id} to GLThread.")
            return False
        
        # Add the message to the user's OAI assistant thread
        if not await self.openai_client.add_to_asst_thread(target_thread.assistant_thread_id, gl_message):
            logger.error(f"Failed to add GLMessage {gl_message.message_id} to OAI assistant thread.")
            return False
        
        return True
    