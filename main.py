import logging
import asyncio
from core.config import OPENAI_API_KEY, setup_logging
from core.cache import GLCache
from clients.discord_client import DiscordClient
from clients.openai_client import OpenAIClient
from services.dtgl import DTGLBroker
from services.cot import ChainOfThoughtPipeline
from core.event_bus import emit_event, ON_READY, ON_MESSAGE, ON_REACTION_ADD, ON_PRESENCE_UPDATE

# Configure logging
logging.basicConfig(level=logging.INFO)  # Adjust the level as needed
logger = logging.getLogger("Main")

class GreggLimper:
    def __init__(self):
        # Setup centralized logging
        setup_logging(level=logging.INFO)

        # Initialize placeholders for components
        self.openai_client = None
        self.discord_client = None
        self.cache = None
        self.message_processor = None

    async def async_init(self):
        """Asynchronous initialization for GreggLimper components."""
        logger.info("Initializing OpenAI...")
        self.openai_client = await OpenAIClient.create()  # Ensure OpenAIClient is asynchronously initialized

        logger.info("Initializing Discord Client...")
        self.discord_client = DiscordClient()

        logger.info("Initializing GLCache...")
        self.cache = GLCache()

        logger.info("Initializing Brokers...")
        self.dtgl_broker = DTGLBroker()

        logger.info("Initializing Chain of Thought Pipeline...")
        self.cot_pipeline = ChainOfThoughtPipeline()

    def run(self):
        # Ensure the Discord client runs in an asynchronous context
        asyncio.run(self._run_discord())

    async def _run_discord(self):
        """Run the Discord client after completing asynchronous initialization."""
        await self.async_init()  # Perform asynchronous initialization
        await self.discord_client.run()  # Start the Discord client

if __name__ == "__main__":
    try:
        gregg_limper = GreggLimper()
        gregg_limper.run()
    except Exception as e:
        logger.error(f"Application failed to start: {e}", exc_info=True)
