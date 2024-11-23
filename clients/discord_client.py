import discord
from core.event_bus import emit_event, ON_READY, ON_MESSAGE, ON_REACTION_ADD, ON_PRESENCE_UPDATE
from pydispatch import dispatcher
from core.config import DISCORD_API_TOKEN
import logging
import threading

logger = logging.getLogger('DiscordClient')

class DiscordClient:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = super(DiscordClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            intents = discord.Intents.default()
            intents.message_content = True
            intents.members = True
            intents.reactions = True
            intents.presences = True

            self.client = discord.Client(intents=intents)

            # Register event handlers
            self.client.event(self.on_ready)
            self.client.event(self.on_message)
            self.client.event(self.on_reaction_add)
            self.client.event(self.on_precense_update)

            self.initialized = True  # Prevent reinitialization

    async def run(self):
        await self.client.start(token=DISCORD_API_TOKEN)

    async def on_ready(self):
        logger.info(f'Logged in as {self.client.user.display_name}!')
        emit_event(ON_READY)

    async def on_message(self, message: discord.Message):
        logger.debug(f"Message from {message.author}: {message.content}")
        emit_event(ON_MESSAGE, message=message)

    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        logger.debug(f"Reaction added by {user}: {reaction.emoji}")
        emit_event(ON_REACTION_ADD, reaction=reaction, user=user)
    
    async def on_precense_update(self, before: discord.Member, after: discord.Member):
        logger.debug(f"{before.display_name} is now {after.status}")
        emit_event(ON_PRESENCE_UPDATE, before=before, after=after)