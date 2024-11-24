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
from typing import List, Dict

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
            message (discord.Message): The user's message to process.
        """
        user_id = message.author.id
        channel_id = message.channel.id

        logger.info(f"Starting Chain of Thought Pipeline for user {message.author.display_name}")

        # Get the user and bot GLThreads
        user_thread = self.cache.threads[user_id]
        user_channel = self.discord_client.get_channel(channel_id)

        if not user_thread:
            logger.error(f"User thread not found for user {user_id}")
            return

        # Convert GLThread to OpenAI format
        oai_messages = await self.message_processor.GLThread_to_OAI(user_thread)

        # Determine content type
        content_type = None
        for i in range(COT_MAX_ATTEMPTS):
            content_type = await self.openai_client.determine_content_type(oai_messages)
            if content_type:
                break
            else:
                logger.error(f"Attempt {i+1}: Failed to determine content type for user {user_id}. Retrying.")
                await asyncio.sleep(2**i)  # Exponential backoff
        else:
            logger.error(f"Failed to determine content type for user {user_id} after {COT_MAX_ATTEMPTS} attempts.")
            content_type = "message"

        # Process content based on type
        attempts = 0
        while attempts < COT_MAX_ATTEMPTS:
            if content_type == "message":
                # Request assistant response
                await self.openai_client.request_asst_response(user_thread.assistant_thread_id)
                if not self.final_response:
                    logger.error(f"Attempt {attempts + 1}: Failed to get response for user {user_id}. Retrying.")
                    await asyncio.sleep(2**attempts)  # Exponential backoff
                    attempts += 1
                    continue

                # Send the response to the user
                discord_msg = await user_channel.send(self.final_response, reference=message)
                if discord_msg:
                    # Add the message to the user's GLThread and OAI assistant thread
                    logger.info(f"Successfully sent response for user {user_id}")
                    gl_msg = await self.cache.add_discord_message(discord_msg)
                    if not gl_msg:
                        logger.error(f"Failed to add response to thread for user {user_id}")
                    logger.info(f"Added response to thread for user {user_id}")

                    # Check if the response needs a follow-up
                    updated_oai_messages = await self.message_processor.GLThread_to_OAI(user_thread)
                    if not await self.openai_client.is_followup_required(updated_oai_messages):
                        logger.info(f"No follow-up required for user {user_id}.")
                        break  # Exit loop if no follow-up is needed

                    logger.info(f"Follow-up required for user {user_id}. Requesting additional response.")
                else:
                    logger.error(f"Attempt {attempts + 1}: Failed to send response for user {user_id}. Retrying.")
            elif content_type == "gif":
                await self._process_gif_response(user_id, user_channel, message, oai_messages)
                break
            elif content_type == "youtube":
                logger.info(f"Content type is YouTube for user {user_id}.")
                break
            elif content_type == "website":
                logger.info(f"Content type is website for user {user_id}.")
                break

            attempts += 1

        if attempts >= COT_MAX_ATTEMPTS:
            logger.error(f"Max follow-up attempts reached for user {user_id}.")
            return

    async def _process_gif_response(self, user_id: int, user_channel: discord.TextChannel, message: discord.Message, oai_messages: List[Dict]):
        """Handle GIF content response from the assistant.
        Args:
            user_id (int): The user's Discord ID.
            user_channel (discord.TextChannel): The user's Discord channel.
            message (discord.Message): The user's message.
            oai_messages (List[Dict]): The user's messages in OpenAI
        """
        search_query = await self.openai_client.generate_search_query("gif", oai_messages)
        gif_url, message_to_cache = await self.gif_processor.search_by_query(search_query)
        if not gif_url:
            logger.error(f"Failed to find GIF for user {user_id}")
            return
        discord_msg = await user_channel.send(gif_url, reference=message)
        if discord_msg:
            discord_msg.content = message_to_cache
            gl_msg = await self.cache.add_discord_message(discord_msg)
            if gl_msg:
                logger.info(f"Successfully added GIF to thread for user {user_id}.")
            else:
                logger.error(f"Failed to add GIF to thread for user {user_id}.")

    def _on_response_complete_wrapper(self, message: str):
        asyncio.create_task(self.on_response_complete(message))
    
    async def on_response_complete(self, message: str):
        self.final_response = message