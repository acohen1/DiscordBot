from pydispatch import dispatcher
import threading
import discord
from clients.discord_client import DiscordClient
from models.threads import GLThread, CombinedGLThread
from processors.msg import MessageProcessor
from processors.cmd import CommandProcessor
from core.cache import GLCache
from core.event_bus import ON_MESSAGE, ON_READY, AWAITING_RESPONSE, emit_event
from datetime import datetime, timezone, timedelta
import asyncio
import logging

logger = logging.getLogger("DTGLBroker")

class DTGLBroker:
    _instance = None
    _lock = threading.Lock()  # Lock object to ensure thread safety

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of OpenAIClient is created, even in a multithreaded context."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DTGLBroker, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, "initialized"):

            self.discord_client = DiscordClient().client
            self.cache = GLCache()

            self.message_processor = MessageProcessor()
            self.command_processor = CommandProcessor()

            dispatcher.connect(self._on_ready_wrapper, signal=ON_READY)
            dispatcher.connect(self._on_message_wrapper, signal=ON_MESSAGE)
            self.initialized = True

    def _on_ready_wrapper(self):
        """Synchronous wrapper for async _on_ready method."""
        asyncio.create_task(self._on_ready())

    async def _on_ready(self):
        """Called when discord bot is logged in."""
        # Grab all channels bot has read permissions for
        guild = self.discord_client.guilds[0]
        bot_member = guild.me
        channels = [c for c in guild.text_channels if c.permissions_for(bot_member).read_messages]

        # Initialize GLCache threads for all users in the channels
        if not await self.cache.init_threads(channels):
            logger.error("Failed to initialize threads.")
            return

    def _on_message_wrapper(self, message: discord.Message):
        """Synchronous wrapper for async _on_message method."""
        asyncio.create_task(self._on_message(message))
    
    # TODO: IF OTHER USERS ARE MENTIONED THAT ARENT THE BOT, COMBINE THEIR THREADS FOR CONTEXT
    async def _on_message(self, message: discord.Message):
        """Called when a discord message is received in any channel the bot has access to."""

        # 1. Ignore the bot's own messages
        if message.author.id == self.discord_client.user.id:
            return
        
        # 2. Process commands and return if processed
        if await self.command_processor.process_commands(message, self.cache.threads):
            return
        
        # 3. Add the message to the internal GLCache and OAI Assistant thread for necessary users
        gl_message = await self.cache.add_discord_message(message)

        # 4. If the message was added, emit awaiting response to begin the CoT pipeline if the bot was mentioned
        if gl_message and self.discord_client.user.mentioned_in(message):
            emit_event(signal=AWAITING_RESPONSE, message=message)



        
