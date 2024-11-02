# Written by: Alex Cohen
# Latest Updaet: 11/01/2024
import os
import json
import re
import aiohttp
import aiofiles
import base64
import tempfile
import logging
import urllib.parse
from collections import deque
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import discord
from openai import OpenAI
from googleapiclient.discovery import build
from config import CONFIG
from typing import Tuple, List, Dict, Optional, Union, Deque

# TODO: Potentially append multiple user messages to a single "user" role reply, instead of sending multiple

# TODO: Fix image attachment processing: currently [Image Description: ...] is not showing up in the cache

# Define type aliases for complex structures
MessageEntry = Dict[str, Union[str, int]]
ConversationHistory = List[MessageEntry]
OpenAIResponse = Dict[str, Optional[Union[str, Dict[str, Optional[str]]]]]
GreggAction = Dict[str, Optional[Union[str, Dict[str, Optional[str]]]]]
RecentMessageCache = Dict[int, Deque[MessageEntry]]
GIFData = Tuple[Optional[str], Optional[str], Optional[str]]
YouTubeData = Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]
LinkResult = Optional[Tuple[str, str]]  # Return type for _process_* methods

class GreggLimperBot:
    def __init__(self) -> None:
        # Initialize components from the configuration
        self._initialize_components()

        # Setup Directories
        os.makedirs(self.training_data_dir, exist_ok=True)

        # Initialize cached conversation history
        self.recent_message_cache: RecentMessageCache = {channel_id: deque(maxlen=self.cached_history_length) for channel_id in self.allowed_channel_ids}
 
    def _initialize_components(self) -> None:
        """Initialize constants, clients, and API keys from the config."""
        
        # Assign constants from CONFIG
        self.fine_tuned_model = CONFIG["FINE_TUNED_MODEL"]
        self.image_detection_model = CONFIG["IMAGE_DETECTION_MODEL"]
        self.search_engine_id = CONFIG["SEARCH_ENGINE_ID"]
        self.allowed_channel_ids = CONFIG["ALLOWED_CHANNEL_IDS"]
        self.training_data_dir = CONFIG["TRAINING_DATA_DIR"]
        self.cache_init_time_limit = timedelta(minutes=CONFIG["CACHE_INIT_TIME_LIMIT_MINUTES"])
        self.cached_history_length = CONFIG["CACHED_HISTORY_LENGTH"]
        self.assistant_context_length = CONFIG["ASSISTANT_CONTEXT_LENGTH"]
        self.reaction_history_length = CONFIG["REACTION_HISTORY_LENGTH"]
        self.gregg_limper_attributes = CONFIG["GREGG_LIMPER_ATTRIBUTES"]
        self.system_prompt = CONFIG["SYSTEM_PROMPT"]

        # API keys and tokens
        self.openai_api_key = CONFIG["OPENAI_API_KEY"]
        self.discord_api_token = CONFIG["DISCORD_API_TOKEN"]
        self.google_api_key = CONFIG["GOOGLE_API_KEY"]
        self.giphy_api_key = CONFIG["GIPHY_API_KEY"]

        # Initialize Discord Client, OpenAI API, and YouTube API client
        self.client = OpenAI(api_key=self.openai_api_key)
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.reactions = True
        self.bot = discord.Client(intents=intents)
        self.youtube_client = build("youtube", "v3", developerKey=self.google_api_key)
        self.google_client = build("customsearch", "v1", developerKey=self.google_api_key)

        # Bind events
        self.bot.event(self.on_ready)
        self.bot.event(self.on_message)
        self.bot.event(self.on_reaction_add)

        # Loggers :O
        self.logger = logging.getLogger("GreggLimperBot")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

    def run(self) -> None:
        """Start tge bot using the initiaized Discord API token."""
        self.bot.run(self.discord_api_token)
    
# ==================== Event Handlers ====================

    async def on_ready(self) -> None:
        """Event handler triggered on bot startup. Load and cache recent message history for each allowed channel."""
        
        self.logger.info(f'Logged in as {self.bot.user}')

        # For each of the allowed channels, grab chat history from discord up to the cached history length and store in the deque
        for channel_id in self.allowed_channel_ids:
            channel = self.bot.get_channel(channel_id)
            if not channel:
                continue

            # Discord gives messages newest first
            async for message in channel.history(limit=self.cached_history_length):
                # Skip messages older than TIME_LIMIT
                if datetime.now(timezone.utc) - message.created_at > self.cache_init_time_limit:
                    continue

                # Process discord message content
                processed_message = await self.process_message_from_discord(message)

                # Create a dictionary entry for the message
                message_entry = {
                    "displayname": message.author.display_name,
                    "role": "user" if message.author != self.bot.user else "assistant",
                    "content": processed_message,
                    "timestamp": message.created_at.isoformat()
                }

                # Append the processed message to the channel's deque cache
                self.recent_message_cache[channel_id].append(message_entry)

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming messages, process its content, cache it, and send response if bot is mentioned.
        Args:
            message (discord.Message): The message object from Discord.
        """
        
        # =============== INCOMING MESSAGE HANDLING ===============

        # Check if the message is from the bot or not in an allowed channel
        channel_id = message.channel.id
        if message.author == self.bot.user or channel_id not in self.allowed_channel_ids:
            return
        
        # Check the message content for the /lobotomy command
        if message.content.startswith(f"<@{self.bot.user.id}>") and message.content.strip().endswith("/lobotomy"):
            # Clear the cache for the channel
            if self.clear_cache(channel_id):
                await message.channel.send("All gone <:brainlet:1300560937778155540>")
            else:
                await message.channel.send("I can't do that right now. <:GreggLimper:975696064478847016>")
            return
        
        # Process the message content (replace links, mentions, and image content)
        processed_message = await self.process_message_from_discord(message)

        # Append the processed message to the channel's deque cache
        message_entry = {
            "displayname": message.author.display_name,
            "role": "user",
            "content": processed_message,
            "timestamp": message.created_at.isoformat()
        }
        self.recent_message_cache[channel_id].appendleft(message_entry)

        # =============== OPENAI RESPONSE PROCESSING ===============

        # Check if bot was mentioned in the message
        if not self.bot.user in message.mentions:
            return

        # Determine the content type to send based on the chat history
        content_type = await self.determine_content_type(channel_id)
        if not content_type:
            logging.error("Error determining content type.")
            return

        message_to_cache = ""
        message_to_send = ""
        
        # Query Gregg Limper for a message response
        if content_type == "message":
            logging.info("Querying Gregg Limper for a text message response")
            # Query Gregg Limper for a text message response
            assistant_reply = await self.query_gregg_for_message(channel_id)

            # Remove any mentions in the assistant's reply
            message_to_send = self.replace_mentions(assistant_reply, message.guild)
            message_to_cache = message_to_send

        # Query Gregg Limper for Media content
        else:
            logging.info(f"Querying Gregg Limper for {content_type} content")
            query = await self.request_search_query(channel_id)
            if not query:
                logging.error("Error generating search query for media content.")
                return
            
            logging.info(f"Generated media query: {query}")
            if content_type == "gif":
                logging.info("Querying GIF content")
                result = await self._process_gif_link(query=query)
                if not result:
                    logging.error("Error querying Giphy for GIF content.")
                    return
                message_to_send, message_to_cache = result

            elif content_type == "youtube":
                logging.info("Querying YouTube content")
                result = await self._process_youtube_link(query=query)
                if not result:
                    logging.error("Error querying YouTube for video content.")
                    return
                message_to_send, message_to_cache = result

            elif content_type == "website":
                logging.info("Querying website content")
                result = await self._process_generic_link(query=query)
                if not result:
                    logging.error("Error querying website for content.")
                    return
                message_to_send, message_to_cache = result
        
        # Append the processed message to the channel's deque cache
        message_entry = {
            "displayname": "Gregg Limper",
            "role": "assistant",
            "content": message_to_cache,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.recent_message_cache[channel_id].appendleft(message_entry)

        # Send the message to Discord
        await message.channel.send(message_to_send)

    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
        """Handle reactions added to messages to save context as training data.
        Args:
            reaction (discord.Reaction): Reaction added to a message.
            user (discord.User): User who added the reaction.
        """
        
        channel_id = reaction.message.channel.id
        reacted_msg_time = reaction.message.created_at

        # Ignore reactions from the bot and messages not sent by the bot
        if user == self.bot.user or reaction.message.author != self.bot.user:
            return
        
        # Check if the reacted message exists in the cache
        cache_messages = list(self.recent_message_cache[channel_id])
        reacted_message = next(
            (
                msg for msg in cache_messages
                if abs(datetime.fromisoformat(msg["timestamp"]) - reacted_msg_time) < timedelta(seconds=1)
            ),
            None
        )
        # Only proceed if the message is in the cache
        if not reacted_message:
            self.logger.info("Reacted message is not in cache, skipping training data addition.")
            return
        
        # Get the reaction history length number of newest messages from the cache
        training_data = {
            "message_id": reaction.message.id,
            "messages": [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"]
                }
                for msg in cache_messages[:self.reaction_history_length]
            ]
        }

        # Load the training data file and append the new training data
        training_data_file = os.path.join(self.training_data_dir, "training_data.json")
        training_data_list = []
        if os.path.exists(training_data_file):
            async with aiofiles.open(training_data_file, "r") as file:
                try:
                    training_data_list = json.loads(await file.read())
                except json.JSONDecodeError:
                    self.logger.error("Error decoding training data JSON file. Initializing as empty list.")
        
        # Check if message_id is already present in training data
        if any(entry["message_id"] == reaction.message.id for entry in training_data_list):
            self.logger.info("Message ID already exists in training data. Skipping addition.")
            return
        
        # Append the new training data to the training data list
        training_data_list.append(training_data)

        # Ensure the directory still exists in case it was deleted during runtime
        os.makedirs(self.training_data_dir, exist_ok=True)

        # Write the updated training data to the file
        async with aiofiles.open(training_data_file, "w") as file:
            try:
                await file.write(json.dumps(training_data_list, indent=4))
                self.logger.info(f"Added message to training data: {reacted_message['content']}")
            except Exception as e:
                self.logger.error(f"Error writing training data to file: {str(e)}")

# ==================== Discord Message Processing ====================

    async def process_message_from_discord(self, message: discord.Message) -> str:
        """Process and clean up a Discoes message to a format compatible with OpenAI.
        Args:
            message (discord.Message): The message object from Discord.
        Returns:
            str: Cleaned-up message content after processing mentions, links, and images.
        """
        message_content = message.content
        self.logger.debug(f"Original message content: {message_content}")

        # 1. Process mentions in the message content
        message_content = self.replace_mentions(message_content, message.guild)
        self.logger.debug(f"After mention processing: {message_content}")

        # 2. Replace links with titles for user ^ bot messages
        is_bot = message.author.id == self.bot.user.id
        message_content = await self.process_links(message_content, is_bot)
        self.logger.debug(f"After link processing: {message_content} from_bot: {is_bot}")

        # 3. Process image content by downloading, encoding, and sending to OpenAI
        image_description = await self.process_image_content(message)
        if image_description:
            message_content += f" [Image description: {image_description}]"
        
        self.logger.debug(f"Final processed content: {message_content}")
        # 4. Return the processed message content
        return message_content

    def replace_mentions(self, message: str, guild: discord.Guild) -> str:
        """Replace all mentions in the message with display names or channel names.
        Args:
            message (str): The content of the message.
            guild (discord.Guild): The guild to fetch members, roles, and channels from.
        Returns:
            str: The content with mentions replaced by display names and the bot's mention removed.
        """
        bot_id = self.bot.user.id  # Store the bot's ID for comparison
        
        def replace(match):
            """Replace the mention with the display name, role name, or channel name."""
            mention_id = int(match.group(1) or match.group(2) or match.group(3))

            # Skip the bot's mention entirely
            if mention_id == bot_id:
                return ""  # Return empty string to remove mention of the bot
            
            if match.group(1):  # User mention
                user = guild.get_member(mention_id)
                return user.display_name if user else "<Unknown User>"

            elif match.group(2):  # Role mention
                role = guild.get_role(mention_id)
                return role.name if role else "<Unknown Role>"

            elif match.group(3):  # Channel mention
                channel = guild.get_channel(mention_id)
                return channel.name if channel else "<Unknown Channel>"

        # This regex matches user mentions (<@userID> or <@!userID>), role mentions (<@&roleID>), and channel mentions (<#channelID>)
        msg_content = re.sub(r"<@!?(\d+)>|<@&(\d+)>|<#(\d+)>", replace, message)

        return msg_content

    async def process_links(self, message_content: str, from_bot: bool = False) -> str:
        """Process and format links in the message content for Discord and cache.
        Args:
            message_content (str): The content of the message.
            from_bot (bool): Whether the message is from the bot.
        Returns:
            str: Message content with links replaced by formatted text.
        """
        self.logger.debug(f"process_links called with content: {message_content}")

        if from_bot:
            # Patterns for bot-generated links (Replace [Title](URL))
            youtube_pattern = re.compile(r"\[(?!.*:::).*?]\(<?(https?://(?:www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+)>?\)")
            gif_pattern = re.compile(r"\[(?!.*:::).*?]\(<?(https?://\S*\.(?:giphy|tenor)\.com/\S+)>?\)")
            link_pattern = re.compile(r"\[(?!.*:::).*?]\(<?(https?://\S+)>?\)")
        else:
            # Patterns for user-generated links (Replace the URLs) 
            youtube_pattern = re.compile(r"(https?://(?:www\.)?(youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11}))[^\s>]*")
            gif_pattern = re.compile(r"(https?://(?:\S+\.)?(?:giphy|tenor)\.com/\S+)")
            link_pattern = re.compile(r"(https?://\S+)")


        # Process YouTube links as [YouTube ::: {title} ::: {author} ::: {description}]
        for match in youtube_pattern.finditer(message_content):
            url = match.group(1)
            logging.debug(f"Processing YouTube link: {url}")
            result = await self._process_youtube_link(url=url)
            if not result:
                logging.error("Error processing YouTube link.")
                return message_content
            _, message_to_cache = result
            message_content = message_content.replace(match.group(0), message_to_cache)
            return message_content

        # Process GIF links as [GIF ::: {title} ::: {description}]
        for match in gif_pattern.finditer(message_content):
            url = match.group(1)
            logging.debug(f"Processing GIF link: {url}")
            result = await self._process_gif_link(url=url)
            if not result:
                logging.error("Error processing GIF link.")
                return message_content
            _, message_to_cache = result
            message_content = message_content.replace(match.group(0), message_to_cache)
            return message_content

        # Process general links as [Website ::: {title} ::: {description}]
        for match in link_pattern.finditer(message_content):
            url = match.group(1)
            logging.debug(f"Processing general link: {url}")
            result = await self._process_generic_link(url=url)
            if not result:
                logging.error("Error processing general link.")
                return message_content
            _, message_to_cache = result
            message_content = message_content.replace(match.group(0), message_to_cache)
            return message_content

        return message_content

    # TODO: see if we should keep gregg limper attributes in the prompt.
    async def process_image_content(self, message: discord.Message) -> Optional[str]:
        """Process image attachments and generate a description.
        Args:
            message (discord.Message): The message object with image attachments.
        Returns:
            Optional[str]: Description of image content or None if no image is found.
        """
        system_prompt = (
            "You are Gregg Limper\n\n"
            f"{self.gregg_limper_attributes}\n\n"
            "Your purpose is to provide a description of the image content in the message.\n\n"
            "Provide a succinct description useful for someone who can't see it. "
            "Include any relevant text or context in the image."
            "Respond in a way that reflects Gregg Limper's personality and style."
        )
        user_prompt = "What is in this image? Provide a succinct description useful for someone who can't see it."

        for attachment in message.attachments:
            # Check if the attachment is an image
            if attachment.content_type and 'image' in attachment.content_type:
                temp_dir = tempfile.gettempdir()
                image_path = os.path.join(temp_dir, attachment.filename)
                base64_image = None
                
                try:
                    # Download the image
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status == 200:
                                async with aiofiles.open(image_path, "wb") as f:
                                    await f.write(await resp.read())
                                self.logger.info(f"Downloaded image {attachment.filename}")
                            else:
                                self.logger.error(f"Failed to download image {attachment.filename} - HTTP {resp.status}")
                                return None

                    # Encode the downloaded image to base64
                    base64_image = await self.encode_image_base64(image_path)
                    if not base64_image:
                        self.logger.error("Failed to encode image to base64.")
                        return None

                    # Prepare and send the request to OpenAI for image analysis
                    response = self.client.chat.completions.create(
                        model=self.image_detection_model,
                        messages=[
                            { "role" : "system", "content" : system_prompt },
                            { "role" : "user", "content" : user_prompt },
                        ],
                        max_tokens=300,
                    )
                    
                    # Retrieve and return the result from OpenAI
                    result = response.choices[0].message.content if response.choices else "No description available"
                    return result

                except Exception as e:
                    self.logger.error(f"Error processing image content: {str(e)}")
                    return None

                finally:
                    # Clean up the temporary file
                    try:
                        os.remove(image_path)
                        self.logger.info(f"Deleted temporary image file: {image_path}")
                    except Exception as e:
                        self.logger.error(f"Error deleting temporary image file: {e}")
        
        # Return None if no image attachments are found
        return None
 
# ==================== OpenAI Response Processing ====================

    # TODO: see if we should keep gregg limper attributes in the prompt.
    async def determine_content_type(self, channel_id: int, max_retries: int = 3) -> Optional[str]:
        """Determine appropriate content type for response based on chat history.
        Args:
            channel_id (int): Channel ID to retrieve history from.
            max_retries (int): Number of retry attempts in case of error.
        Returns:
            Optional[str]: 'message', 'GIF', 'YouTube', or 'Website' or None if determination fails.
        """

        prompt = (
            "You are Gregg Limper.\n\n"
            f"{self.gregg_limper_attributes}\n\n"
            "Based on the chat history, reply with one word that best describes the type of response that would be most relevant and helpful: "
            "'message', 'GIF', 'YouTube', or 'Website'. Do not provide any additional text or explanations. **ONLY REPLY WITH ONE OF THE FOLLOWING WORDS:**: "
            "message, GIF, YouTube, or Website."
        )
        conversation_history = self.process_cache_for_openai(channel_id, custom_prompt=prompt)
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=conversation_history,
                    max_tokens=10,
                    temperature=0.2 + (0.2 * attempt),  # Increase temperature with each attempt
                )
                content_type = response.choices[0].message.content.strip().lower()
                
                if content_type in ["message", "gif", "youtube", "website"]:
                    return content_type
                else:
                    self.logger.warning(f"Attempt {attempt + 1}: Invalid content type response: {content_type}")

            except Exception as e:
                self.logger.error(f"Error determining content type on attempt {attempt + 1}: {str(e)}")

        # If all attempts fail, return None
        self.logger.error(f"Failed to determine content type after {max_retries} attempts. Defaulting to 'message'.")
        return None

    async def query_gregg_for_message(self, channel_id: int) -> Optional[str]:
        """Query Gregg Limper for a message response based on the recent chat history.
        Args:
            channel_id (int): The ID of the channel to fetch the message for.
        Returns:
            str: The assistant's reply as a string, or None if an error occurs.
        """
        conversation_history = self.process_cache_for_openai(channel_id)
        try:
            response = self.client.chat.completions.create(
                model=self.fine_tuned_model,
                messages=conversation_history,
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error querying Gregg Limper for message: {str(e)}")
            return None
    
    async def request_search_query(self, channel_id: int) -> Optional[str]:
        """Generate a search query based on recent messages in the channel
        Args:
            channel_id (int): The ID of the channel to generate the search query for.
        Returns:
            str: The search query generated based on the recent messages, or None if an error occurs.
        """

        # Format the conversation history for OpenAI
        prompt = f"""
        You are Gregg Limper.
        {self.gregg_limper_attributes}
        Based on the recent chat history, generate a concise search query that directly supports or addresses the latest message's content.
        Limit the query length to ensure clarity and relevance in a response. Reply only with the query itself—no links or media.
        Do not reuse the same media or queries within the same conversation.
        """

        conversation_history = self.process_cache_for_openai(channel_id, custom_prompt=prompt)

        # Send to OpenAI to generate a query
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=conversation_history,
                max_tokens=20,
                temperature=0.7
            )

            # Extract the suggested search term from the response
            search_query = response.choices[0].message.content.strip() if response.choices else None
            self.logger.info(f"Generated search query: {search_query}")
            return search_query

        except Exception as e:
            self.logger.error(f"Error generating search query: {e}")
            return None

# ==================== Search Functions ====================

    async def search_youtube(self, query: Optional[str] = None, url: Optional[str] = None) -> YouTubeData:
        """Search Youtube and return the top video result or details of a specific video URL.
        Args:
            query (Optional[str]): The search query for Youtube.
            url (Optional[str]): The Youtube video URL.
        Returns:
            Optional[YouTubeData]: A tuple containing the title, author, description, and URL of the top video result, 
            or None if no results are found.
        """
        try:
            # Search by query
            if query:
                request = self.youtube_client.search().list(
                    part="snippet",
                    maxResults=1,
                    q=query,
                    type="video"
                )
                response = request.execute()
                if items := response.get("items"):
                    snippet = items[0].get("snippet", {})
                    video_id = items[0].get("id", {}).get("videoId")
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    title = snippet.get("title")
                    author = snippet.get("channelTitle")
                    description = snippet.get("description")
                    return title, author, description, url
                else:
                    self.logger.warning(f"No YouTube results found for query: {query}")
                    return None

            # Fetch video details by URL
            elif url:
                self.logger.debug(f"Fetching YouTube video details for URL: {url}")
                video_id = re.search(r"(?:v=|/)([A-Za-z0-9_-]{11})", url)
                if not video_id:
                    self.logger.error(f"Invalid YouTube URL format: {url}")
                    return None
                video_id = video_id.group(1)

                request = self.youtube_client.videos().list(part="snippet", id=video_id)
                response = request.execute()
                if items := response.get("items"):
                    snippet = items[0].get("snippet", {})
                    title = snippet.get("title")
                    author = snippet.get("channelTitle")
                    description = snippet.get("description")
                    return title, author, description, url
                else:
                    self.logger.warning(f"No YouTube results found for URL: {url}")
                    return None

            else:
                self.logger.error("No query or URL provided for YouTube search.")
                return None

        except Exception as e:
            self.logger.error(f"Error searching YouTube: {str(e)}")
            return None

    async def search_gif(self, query: Optional[str] = None, url: Optional[str] = None) -> GIFData:
        """Search Giphy and return the GIF title, description (if available), and URL.
        Args:
            query (str): The search query for Giphy.
            url (str): The GIF URL.
        Returns:
            tuple: The title, description, and URL of the top GIF result, or None if no results are found.
        """
        if not (query or url):
            self.logger.error("Must specify either query or url for search_gif")
            return None
        
        try:
            # Handle direct URL case
            if url and not query:
                title, description = await self.fetch_link_data(url)
                return title, description, url

            # Limit length to prevent 414 errors
            query = query[:50]
            encoded_query = urllib.parse.quote(query)
            giphy_url = f"https://api.giphy.com/v1/gifs/search?api_key={self.giphy_api_key}&q={encoded_query}&limit=1"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(giphy_url) as response:
                    if not response.status == 200:
                        self.logger.error(f"Error searching Giphy: HTTP {response.status}")
                        return None
                    
                    data = await response.json()
                    gif_data = data.get("data", [])

                    if not gif_data:
                        self.logger.error(f"No Giphy results found for query: {query}")
                        return None
                    
                    # Extract title, url, and tags
                    gif_entry = gif_data[0]
                    title = gif_entry.get("title")
                    url = gif_entry.get("url")
                    tags = gif_entry.get("tags", [])
                    description = ", ".join(tags) if tags else None

                    if title and url and description:
                        return title, description, url
                    elif title and url:
                        return title, "No Description Available", url
                    else:
                        self.logger.error(f"Missing title or URL for Giphy query: {query}")
                        return None
                    
        except Exception as e:
            self.logger.error(f"Error searching Giphy: {str(e)}")
            return None

    async def search_google(self, query: str, num_results=1) -> Optional[Tuple[str, str, str]]:
        """Search Google and return the top search result.
        Args:
            query (str): The search query for Google.
            num_results (int): The number of search results to return (default 1).
        Returns:
            list[dict]: The top search results from Google (up to num_results), or empty list if no results are found.
        """
        # Perform a Google search and return the top search result
        try:
            response = self.google_client.cse().list(q=query, cx=self.search_engine_id, num=num_results).execute()
            items = response.get("items", [])
            if items:
                title = items[0].get("title")
                snippet = items[0].get("snippet")
                link = items[0].get("link")
                return title, snippet, link
            else:
                self.logger.warning(f"No Google results found for query: {query}")
                return []
        except Exception as e:
            self.logger.error(f"Error searching Google: {str(e)}")
            return []

    async def fetch_link_data(self, url: str) -> Optional[Tuple[str, str]]:
        """Fetch the title and description of a webpage using BeautifulSoup. 
        Args:
            url (str): The URL of the webpage.
        Returns:
            Optional[Tuple[str, str]]: The title and description of the webpage, or None if an error occurs.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, "html.parser")

                        # Attempt to get the title and description
                        title = soup.title.string
                        description_meta = soup.find("meta", attrs={"name": "description"})
                        description = description_meta["content"] if description_meta else "No description available"
                        return title, description
                    else:
                        self.logger.warning(f"Error fetching link data: {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error fetching link data: {str(e)}")
            return None

# ==================== Utility Functions ====================

    def process_cache_for_openai(self, channel_id: int, custom_prompt: Optional[str]=None) -> ConversationHistory:
        """Process the recent messages in the specified channel's cache for OpenAI input.
        Args:
            channel_id (int): The ID of the channel to process.
            custom_prompt (Optional[str]): A custom system prompt to include at the start of the conversation history.
        Returns:
            ConversationHistory: A list of conversation entries in the format {"role": str, "content": str} (oldest messages first),
            with an optional system prompt as the first entry.
        """
        # Reverse recent messages to maintain chronological order
        recent_messages = list(self.recent_message_cache[channel_id])[:self.assistant_context_length]
        recent_messages.reverse()

        # Initialize an empty conversation history and variables for tracking role and content
        conversation_history = []
        current_role = None
        current_content = ""

        # Iterate over recent messages to combine consecutive messages by role
        for msg in recent_messages:
            role = msg["role"]
            content = f"{msg['displayname']}: {msg['content']}" if role == "user" else msg["content"]
            
            if role == current_role:
                # If the role is the same as the previous one, continue concatenating
                current_content += f"\n{content}"
            else:
                # When the role switches, save the accumulated message and reset
                if current_role is not None:
                    conversation_history.append({"role": current_role, "content": current_content.strip()})
                
                # Start a new message block for the new role
                current_role = role
                current_content = content

        # Append the final accumulated message
        if current_content:
            conversation_history.append({"role": current_role, "content": current_content.strip()})

        # Insert the system prompt at the start of the conversation history
        if custom_prompt:
            conversation_history.insert(0, {"role": "system", "content": custom_prompt})
        else:
            conversation_history.insert(0, {"role": "system", "content": self.system_prompt})


        return conversation_history

    def clear_cache(self, channel_id: int) -> bool:
        """Clear the recent message cache for the specified channel.
        Args:
            channel_id (int): The ID of the channel to clear the cache for.
        Returns:
            bool: True if the cache was cleared successfully, False otherwise.
        """
        self.recent_message_cache[channel_id].clear()
        if self.recent_message_cache[channel_id]:
            self.logger.error(f"Failed to clear cache for channel {channel_id}")
            return False
        else:
            self.logger.info(f"Cleared cache for channel {channel_id}")
            return True

    def show_cache(self, show_names: bool = False) -> Dict[Union[int, str], List[MessageEntry]]:
        """Displays the contents of the recent message cache for each allowed channel.
        Args:
            show_names (bool): Whether to display the display names of users (defaults to False, showing IDs).
        Returns:
            Dict[Union[int, str], List[MessageEntry]]: A dictionary of cache contents with either channel IDs or names as keys.
        """
        cache_contents = {}
        
        for channel_id, messages in self.recent_message_cache.items():
            # Retrieve the channel name if show_names is True
            if show_names:
                channel = self.bot.get_channel(channel_id)
                channel_key = channel.name if channel else f"Unknown Channel {channel_id}"
            else:
                channel_key = channel_id
            
            # Create a list of message summaries for each channel
            channel_cache = [
                {
                    "displayname": message["displayname"],
                    "role": message["role"],
                    "content": message["content"],
                    "timestamp": message["timestamp"]
                }
                for message in messages
            ]
            cache_contents[channel_key] = channel_cache
        
        return cache_contents

    # TODO: see if we should keep gregg limper attributes in the summarize description function
    async def summarize_description(self, description: str) -> str:
        """Generate a succinct summary of the provided description using GreggLimper.
        Args:
            description (str): The full description text to summarize.
        Returns:
            str: A concise summary of the description, or "No summary available" if an error occurs.
        """
        try:
            # Construct a prompt for generating a succinct summary
            system_prompt = (
                f"You are Gregg Limper\n\n"
                f"{self.gregg_limper_attributes}\n\n"
                "Your purpose is to provide a concise summary of text descriptions.\n\n"
                "Respond in a way that reflects Gregg Limper's personality and style.\n\n"
                "**Do not include mentions to Tenor or Giphy.**"
            )

            user_prompt = (
                f"Create a concise one-to-two-sentence summary for the following description:\n\n"
                f"{description}\n\n"
                "Ignore references to Tenor or Giphy.\n\n"
                "Summary:"
            )
            # Send the prompt to the OpenAI API for summarization
            response = self.client.chat.completions.create(
                model=self.fine_tuned_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=50,
                temperature=0.7
            )

            # Extract and return the summary if it exists
            summary = response.choices[0].message.content.strip() if response.choices else "No summary available"
            return summary

        except Exception as e:
            self.logger.error(f"Error summarizing description: {str(e)}")
            return "No summary available" 

    async def encode_image_base64(self, image_path: str) -> str:
        """Encode an image at the specified path to base64 format.
        Args:
            image_path (str): The path of the image to encode.
        Returns:
            str: The base64 encoded string of the image.
        """
        async with aiofiles.open(image_path, "rb") as image_file:
            return base64.b64encode(await image_file.read()).decode('utf-8')
        
    async def _process_youtube_link(self, query: Optional[str] = None, url: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Search YouTube for the given query or URL and format the response for a Discord message and cache entry.
    
        Args:
        query (Optional[str]): The search query for YouTube.
        url (Optional[str]): The YouTube video URL.
    
        Returns:
        Optional[Tuple[str, str]]: A tuple containing the formatted YouTube link and cache entry, or None if an error occurs.
        """
        
        if not (query or url):
            logging.error("Must specify either query or url for _process_youtube_link")
            return None
        
        # Process the url
        if url:
            result = await self.search_youtube(url=url)
            if result:
                title, author, description, url = result
            else:
                return None
            
            description = await self.summarize_description(description) if description else None

            if title and author and description and url:
                # Format for message and cache
                message_to_send = f"[{title}]({url})"
                message_to_cache = f"[YouTube ::: {title} ::: {author} ::: {description}]"
                
                return message_to_send, message_to_cache

            elif title and author and url:
                # Format for message and cache
                message_to_send = f"[{title}]({url})"
                message_to_cache = f"[YouTube ::: {title} ::: {author} ::: No description available]"
                
                return message_to_send, message_to_cache
            
            elif title and url:
                # Format for message and cache
                message_to_send = f"[{title}]({url})"
                message_to_cache = f"[YouTube ::: {title} ::: Unknown Author ::: No description available]"
                
                return message_to_send, message_to_cache
            
            else:
                logging.log_error("YouTube search returned no results.")
                return None

        # Process the query
        title, author, description, url = await self.search_youtube(query=query)
        description = await self.summarize_description(description) if description else None

        if title and author and description and url:
            # Format for message and cache
            message_to_send = f"[{title}]({url})"
            message_to_cache = f"[YouTube ::: {title} ::: {author} ::: {description}]"
            
            return message_to_send, message_to_cache
        
        elif title and author and url:
            # Format for message and cache
            message_to_send = f"[{title}]({url})"
            message_to_cache = f"[YouTube ::: {title} ::: {author} ::: No description available]"
            
            return message_to_send, message_to_cache
        
        elif title and url:
            # Format for message and cache
            message_to_send = f"[{title}]({url})"
            message_to_cache = f"[YouTube ::: {title} ::: Unknown Author ::: No description available]"
            
            return message_to_send, message_to_cache
        
        else:
            logging.log_error("YouTube search returned no results.")
            return None

    async def _process_gif_link(self, query: Optional[str] = None, url: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Search Giphy for the given query or URL and format the response for a Discord message and cache entry.
        Args:
            query (Optional[str]): The search query for GIF.
            url (Optional[str]): The GIF URL.
        Returns:
            Optional[Tuple[str, str]]: A tuple containing the formatted GIF link and cache entry, or None if an error occurs
        """
        if not (query or url):
            logging.error("Must specify either query or url for _process_gif_link")
            return None
        
        # Process the url
        if url:
            result = await self.search_gif(url=url)
            if result:
                title, description, url = result
            else:
                return None
            
            description = await self.summarize_description(description) if description else None

            if title and description and url:
                # Format for message and cache
                message_to_send = url
                message_to_cache = f"[GIF ::: {title} ::: {description}]"

                return message_to_send, message_to_cache
            
            elif title and url:
                # Format for message and cache
                message_to_send = url
                message_to_cache = f"[GIF ::: {title} ::: No description available]"

                return message_to_send, message_to_cache
            
            else:
                logging.log_error("GIF search returned no results.")
                return None

        # Process the query
        title, description, url = await self.search_gif(query=query)
        description = await self.summarize_description(description) if description else None

        if title and description and url:
            # Format for message and cache
            message_to_send = url
            message_to_cache = f"[GIF ::: {title} ::: {description}]"

            return message_to_send, message_to_cache
        
        elif title and url:
            # Format for message and cache
            message_to_send = url
            message_to_cache = f"[GIF ::: {title} ::: No description available]"

            return message_to_send, message_to_cache
        
        else:
            logging.log_error("GIF search returned no results.")
            return None
   
    async def _process_generic_link(self, query: Optional[str] = None, url: Optional[str] = None, num_results=1) -> Optional[Tuple[str, str]]:
        """Search Google for the given query or BeautifulSoup for the given URL, and format the response for a Discord message and cache entry.
        Args:
            query (Optional[str]): The search query for Google.
            url (Optional[str]): The URL to process.
            num_results (int): The number of search results to retrieve (default 1).
        Returns:
            Optional[Tuple[str, str]]: A tuple containing the formatted website link and cache entry, or None if an error occurs.
        """
        if not (query or url):
            logging.error("Must specify either query or url for _process_generic_link")
            return None
        # Process the url
        if url:
            result = await self.fetch_link_data(url)
            if result:
                title, description = result
            else:
                return None
            
            description = await self.summarize_description(description) if description else None

            if title and description:
                # Format for message and cache
                message_to_send = f"[{title}]({url})"
                message_to_cache = f"[Website ::: {title} ::: {description}]"

                return message_to_send, message_to_cache
            
            elif title:
                # Format for message and cache
                message_to_send = f"[{title}]({url})"
                message_to_cache = f"[Website ::: {title} ::: No description available]"

                return message_to_send, message_to_cache
            
            else:
                logging.log_error("Website search returned no results.")
                return None
        
        # Process the query
        results = await self.search_google(query, num_results=num_results)
        for result in results:
            title, snippet, url = result
            description = await self.summarize_description(snippet) if snippet else None
            if title and description and url:
                # Format for message and cache
                message_to_send = f"[{title}]({url})"
                message_to_cache = f"[Website ::: {title} ::: {description}]"

                return message_to_send, message_to_cache
            
            elif title and url:
                # Format for message and cache
                message_to_send = f"[{title}]({url})"
                message_to_cache = f"[Website ::: {title} ::: No description available]"

                return message_to_send, message_to_cache

            else:
                logging.log_error("Website search returned no results.")
                return None
