import logging
from core.event_bus import  emit_event, AWAITING_RESPONSE, RESPONSE_COMPLETE
from clients.openai_client import OpenAIClient
from core.cache import GLCache, GLMessage
from core.config import COT_MAX_ATTEMPTS
from models.threads import CombinedGLThread, GLThread
from pydispatch import dispatcher
from processors.msg import MessageProcessor
from processors.gif import GIFProcessor
import discord
from clients.discord_client import DiscordClient
import asyncio

logger = logging.getLogger("ChainOfThoughtPipeline")

class ChainOfThoughtPipeline:
    """Handles the pipeline for processing events after AWAITING_RESPONSE."""

    def __init__(self):
        self.openai_client = OpenAIClient.get_instance()
        self.discord_client = DiscordClient().client
        self.cache = GLCache()
        self.message_processor = MessageProcessor()
        self.gif_processor = GIFProcessor()
        dispatcher.connect(self._on_pipeline_wrapper, signal=AWAITING_RESPONSE)
        dispatcher.connect(self._on_response_complete_wrapper, signal=RESPONSE_COMPLETE)

        self.final_response = None
        self.target_message = None

    def _on_pipeline_wrapper(self, message: discord.Message):
        """Synchronous wrapper for async run_pipeline method."""
        asyncio.create_task(self.run_pipeline(message))

    async def run_pipeline(self, message: discord.Message):
        """Run the Chain of Thought Pipeline
        Args:
            user_id (int): The user id of the person to respond to.
        """
        user_id = message.author.id
        channel_id = message.channel.id

        logger.info(f"Starting Chain of Thought Pipeline for user {message.author.display_name}")

        # Get the user and bot GLThreads
        user_thread = self.cache.threads[user_id]
        user_channel = self.discord_client.get_channel(channel_id)

        if not user_thread:
            logger.error(f"User or bot thread not found for user {user_id}")
            return
    
        # Convert GLThread to format suitable for OpenAI
        oai_messages = await self.message_processor.GLThread_to_OAI(user_thread)

        # Determine the content type of the messages
        content_type = None
        for i in range(COT_MAX_ATTEMPTS):
            content_type = await self.openai_client.determine_content_type(oai_messages)
            if content_type:
                break
            else:
                logger.error(f"Attempt {i+1}: failed to determine content type for user {user_id}. Retrying.")
                await asyncio.sleep(2**i)   # exponential backoff
        else:
            logger.error(f"Failed to determine content type for user {user_id} after {COT_MAX_ATTEMPTS} attempts. Defaulting to message.")
            content_type = "message"

        # Process a message response
        if content_type == "message":
            for i in range(COT_MAX_ATTEMPTS):
                # Once run, this will emit RESPONSE_COMPLETE on the BUS via the assistant broker which is picked up here
                await self.openai_client.request_asst_response(user_thread.assistant_thread_id)
                if not self.final_response:
                    logger.error(f"Attempt {i+1}: failed to get response for user {user_id}. Retrying.")
                    await asyncio.sleep(2**i)   # exponential backoff
                else:
                    # TODO: We need to figure out a way to get the reference to the message that was sent
                    # Send the message to the user
                    discord_msg = await user_channel.send(self.final_response, reference=message)
                    if discord_msg:
                        # If sent, add the message to the user's GLThread and OAI asst thread 
                        logger.info(f"Successfully sent response for user {user_id}")
                        gl_msg = await self.cache.add_discord_message(discord_msg)
                        if not gl_msg:
                            logger.error(f"Failed to add response to thread for user {user_id}")
                        else:
                            logger.info(f"Added response to thread for user {user_id}")
                        return
                    else:
                        logger.error(f"Attempt {i+1}: failed to send response for user {user_id}. Retrying.")
            else:
                logger.error(f"Failed to get response for user {user_id} after {COT_MAX_ATTEMPTS} attempts.")
                return

        # Process a media response
        else:
            search_query = await self.openai_client.generate_search_query(content_type, oai_messages)
            # Search for a gif
            if content_type == "gif":
                gif_url, message_to_cache = await self.gif_processor.search_by_query(search_query)
                if not gif_url:
                    logger.error(f"Failed to find GIF for user {user_id}")
                    return
                # Send the gif to the user
                discord_msg = await user_channel.send(gif_url, reference=message)
                if discord_msg:
                    # If sent, add the message to the user's GLThread and OAI asst thread 
                    logger.info(f"Successfully sent GIF for user {user_id}")
                    discord_msg.content = message_to_cache
                    gl_msg = await self.cache.add_discord_message(discord_msg)
                    if not gl_msg:
                        logger.error(f"Failed to add GIF to thread for user {user_id}")
                    else:
                        logger.info(f"Added GIF to thread for user {user_id}")
                else:
                    logger.error(f"Failed to send GIF for user {user_id}")

            # Search for a youtube video
            elif content_type == "youtube":
                pass
            # Search for a website
            elif content_type == "website":
                pass

    def _on_response_complete_wrapper(self, message: str):
        asyncio.create_task(self.on_response_complete(message))
    
    async def on_response_complete(self, message: str):
        self.final_response = message

    

        

