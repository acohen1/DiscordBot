import logging
from core.event_bus import  emit_event, AWAITING_RESPONSE, ON_RESPONSE_SENT
from clients.openai_client import OpenAIClient
from core.cache import GLCache
from core.config import COT_MAX_ATTEMPTS, MSG_MAX_FOLLOWUPS
from models.threads import GLThread, GLMessage
from pydispatch import dispatcher
from processors.msg import MessageProcessor
from processors.gif import GIFProcessor
from processors.yt import YouTubeProcessor
from processors.web import WebProcessor
import discord
from clients.discord_client import DiscordClient
import asyncio
from typing import List, Dict
from datetime import datetime, timezone

logger = logging.getLogger("ChainOfThoughtPipeline")

# TODO: Add 'research' response type that searches the web for information and returns a summary of the results.
# We inject this sumnmary as a system message in the chat to send for OpenAI to process.

class ChainOfThoughtPipeline:
    """Handles the pipeline for processing events after AWAITING_RESPONSE."""

    def __init__(self):
        self.openai_client = OpenAIClient.get_instance()
        self.discord_client = DiscordClient().client
        self.cache = GLCache()
        self.message_processor = MessageProcessor()
        self.gif_processor = GIFProcessor()
        self.youtube_processor = YouTubeProcessor()
        self.web_processor = WebProcessor()
        dispatcher.connect(self._on_pipeline_wrapper, signal=AWAITING_RESPONSE)

    def _on_pipeline_wrapper(self, message: discord.Message):
        """Synchronous wrapper for async run_pipeline method."""
        asyncio.create_task(self.run_pipeline(message))

    async def run_pipeline(self, message: discord.Message):
        """Run the Chain of Thought Pipeline
        Args:
            message (discord.Message): The user's message to process.
        """
        logger.info(f"Starting Chain of Thought Pipeline for user {message.author.display_name}")

        # 1. Get the user and bot GLThreads, convert to OpenAI format
        user_id = message.author.id
        user_thread = self.cache.threads[user_id]
        user_channel = self.discord_client.get_channel(message.channel.id)
        if not user_thread:
            logger.error(f"User thread not found for user {user_id}")
            return
        oai_messages = await self.message_processor.GLThread_to_OAI(user_thread)

        # 2. Determine content type to respond with
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

        # 3. Process content based on type
        if content_type == "message":
            if await self._process_message_response(user_id, user_channel, message, oai_messages):
                emit_event(ON_RESPONSE_SENT)
            else:
                logger.error(f"Failed to process message response for user {user_id}.")
        
        # GIF
        elif content_type == "gif":
            if await self._process_gif_response(user_id, user_channel, message, oai_messages):
                emit_event(ON_RESPONSE_SENT)
            else:
                logger.error(f"Failed to process GIF response for user {user_id}.")

        # YouTube
        elif content_type == "youtube":
            logger.info(f"Content type is YouTube for user {user_id}.")
            if await self._process_youtube_response(user_id, user_channel, message, oai_messages):
                emit_event(ON_RESPONSE_SENT)
            else:
                logger.error(f"Failed to process YouTube response for user {user_id}.")

        # Website
        elif content_type == "website":
            logger.info(f"Content type is website for user {user_id}.")
            if await self._process_web_response(user_id, user_channel, message, oai_messages):
                emit_event(ON_RESPONSE_SENT)

        # Research
        elif content_type == "research":
            logger.info(f"Content type is research for user {user_id}.")
            if await self._begin_resarch_pipeline(user_id, user_channel, message, oai_messages):
                emit_event(ON_RESPONSE_SENT)

    # ------------------ Response Handling ------------------

    async def _process_message_response(self, user_id: int, user_channel: discord.TextChannel, message: discord.Message, oai_messages: List[Dict]) -> bool:
        attempts = 0
        sent_msg_count = 0
        while attempts < COT_MAX_ATTEMPTS and sent_msg_count < MSG_MAX_FOLLOWUPS:
            # 1. Request OpenAI response
            response = await self.openai_client.generate_message_response(oai_messages)
            if not response:
                logger.error(f"Attempt {attempts + 1}: Failed to get response for user {user_id}. Retrying.")
                await asyncio.sleep(2**attempts)  # Exponential backoff
                attempts += 1
                continue
            
            # TODO: Fix for current processor.msg implementation; still monitoring for user prefixes
            # Remove ANY user prefix from the response if it appears
            for member in user_channel.members:
                if response.startswith(f"{member.display_name}: "):
                    logger.info(f"Removing prefixed '{member.display_name}' from bot response...")
                    response = response.replace(f"{member.display_name}: ", "")
                    break

            logger.info(f"Generated response for user {user_id}: {response[:50]}...")

            # 2. Send the response to the user
            sent_message = await user_channel.send(response, reference=message)
            if not sent_message:
                logger.error(f"Attempt {attempts + 1}: Failed to send response for user {user_id}. Retrying.")
                await asyncio.sleep(2**attempts)
                attempts += 1
                continue
            logger.info(f"Successfully sent response to user {user_id}")

            # 3. Add the message to all necessary GLThreads
            gl_msg = await self.cache.add_discord_message(sent_message)
            if not gl_msg:
                logger.error(f"Failed to add response to GLThread for user {user_id}. Aborting.")
                return False
            logger.info(f"Added response to GLThreads")

            # 4. Check if the response needs a follow-up
            user_thread = self.cache.threads[user_id]
            updated_oai_messages = await self.message_processor.GLThread_to_OAI(user_thread)
            if not await self.openai_client.is_followup_required(updated_oai_messages):
                logger.info(f"No follow-up required for user {user_id}.")
                break  # Exit loop if no follow-up is needed
            else:
                logger.info(f"Follow-up required for user {user_id}.")
                sent_msg_count += 1

        return True

    async def _process_gif_response(self, user_id: int, user_channel: discord.TextChannel, message: discord.Message, oai_messages: List[Dict]) -> bool:
        """Handle GIF content response from the assistant.
        Args:
            user_id (int): The user's Discord ID.
            user_channel (discord.TextChannel): The user's Discord channel.
            message (discord.Message): The user's message.
            oai_messages (List[Dict]): The user's messages in OpenAI
        Returns:
            bool: True if the GIF was sent successfully, False otherwise.
        """
        search_query = await self.openai_client.generate_search_query("gif", oai_messages)
        gif_url, message_to_cache = await self.gif_processor.search_by_query(search_query)
        if not gif_url:
            logger.error(f"Failed to find GIF for user {user_id}")
            return False
        discord_msg = await user_channel.send(gif_url, reference=message)
        if discord_msg:
            discord_msg.content = message_to_cache
            if not await self.cache.add_discord_message(discord_msg):
                logger.error(f"Failed to add GIF to GLThread for user {user_id}.")
                return False
            
        logger.info(f"Successfully sent GIF to user {user_id}")
        return True
            
    async def _process_youtube_response(self, user_id: int, user_channel: discord.TextChannel, message: discord.Message, oai_messages: List[Dict]) -> bool:
        """Handle YouTube content response from the assistant.
        Args:
            user_id (int): The user's Discord ID.
            user_channel (discord.TextChannel): The user's Discord channel.
            message (discord.Message): The user's message.
            oai_messages (List[Dict]): The user's messages in OpenAI
        Returns:
            bool: True if the YouTube video was sent successfully, False otherwise.
        """
        search_query = await self.openai_client.generate_search_query("youtube", oai_messages)
        message_to_send, message_to_cache = await self.youtube_processor.search_by_keyword(search_query, oai_messages)
        if not message_to_send:
            logger.error(f"Failed to find YouTube video for user {user_id}")
            return False
        discord_msg = await user_channel.send(message_to_send, reference=message)
        if discord_msg:
            discord_msg.content = message_to_cache
            if not await self.cache.add_discord_message(discord_msg):
                logger.error(f"Failed to add YouTube video to GLThread for user {user_id}.")
                return False

        logger.info(f"Successfully sent YouTube video to user {user_id}")
        return True
    
    async def _process_web_response(self, user_id: int, user_channel: discord.TextChannel, message: discord.Message, oai_messages: List[Dict]) -> bool:
        """Handle website content response from the assistant.
        Args:
            user_id (int): The user's Discord ID.
            user_channel (discord.TextChannel): The user's Discord channel.
            message (discord.Message): The user's message.
            oai_messages (List[Dict]): The user's messages in OpenAI
        Returns:
            bool: True if the website was sent successfully, False otherwise.
        """
        search_query = await self.openai_client.generate_search_query("website", oai_messages)
        message_to_send, message_to_cache = await self.web_processor.search_by_keyword(search_query, oai_messages)
        if not message_to_send:
            logger.error(f"Failed to find website for user {user_id}")
            return False
        discord_msg = await user_channel.send(message_to_send, reference=message)
        if discord_msg:
            discord_msg.content = message_to_cache
            if not await self.cache.add_discord_message(discord_msg):
                logger.error(f"Failed to add website to GLThread for user {user_id}.")
                return False
            
        logger.info(f"Successfully sent website to user {user_id}")
        return True
    
    async def _begin_resarch_pipeline(self, user_id: int, user_channel: discord.TextChannel, message: discord.Message, oai_messages: List[Dict]) -> bool:
        logger.info(f"Starting research pipeline for user {self.discord_client.get_user(user_id).display_name}")

        # 1. Generate a research query
        logger.info("Generating research query...")
        search_query = await self.openai_client.generate_search_query("research", oai_messages)
        if not search_query:
            logger.error(f"Failed to generate research query for user {user_id}")
            return False
        
        # 2. Search the web and YouTube for information
        logger.info("Searching the web for information...")
        _, research_to_cache = await self.web_processor.search_by_keyword(search_query, oai_messages)
        if not research_to_cache:
            logger.error(f"Failed to find information for user {user_id}")
            return False

        logger.info("Searching youtube for information...")
        _, video_to_cache = await self.youtube_processor.search_by_keyword(search_query, oai_messages)
        if not video_to_cache:
            logger.error(f"Failed to find YouTube video for user {user_id}")
            return False

        # 3. Create the research note
        research_note = f"**Research Summary**: {research_to_cache}\n\nYouTube Video: {video_to_cache}"

        # TODO: Monitor if we should add the research note as a 'assistant' message in the GLThread,
        # At the moment, it's only temporary for the informed response

        # 4. Generate an informed response from OpenAI
        logger.info("Generating informed response...")
        response = await self.openai_client.generate_message_response(oai_messages, research_note=research_note)
        if not response:
            logger.error(f"Failed to generate informed response for user {user_id}")
            return False

        # 5. Send the bot's response to the user
        logger.info("Sending informed response to user...")
        sent_message = await user_channel.send(response, reference=message)
        if not sent_message:
            logger.error(f"Failed to send informed response for user {user_id}")
            return False
        
        # 3. Add the bot's message to all necessary GLThreads
        gl_msg = await self.cache.add_discord_message(sent_message)
        if not gl_msg:
            logger.error(f"Failed to add response to GLThread for user {user_id}. Aborting.")
            return False
        logger.info(f"Added response to GLThreads")
        return True