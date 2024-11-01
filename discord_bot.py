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

# Define type aliases for complex structures
MessageEntry = Dict[str, Union[str, int]]
ConversationHistory = List[MessageEntry]
OpenAIResponse = Dict[str, Optional[Union[str, Dict[str, Optional[str]]]]]
GreggAction = Dict[str, Optional[Union[str, Dict[str, Optional[str]]]]]
RecentMessageCache = Dict[int, Deque[MessageEntry]]
GIFData = Tuple[Optional[str], Optional[str], Optional[str]]
YouTubeData = Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]

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
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

    def run(self) -> None:
        self.bot.run(self.discord_api_token)
    
# ==================== Event Handlers ====================

    async def on_ready(self) -> None:
        """Event handler triggered when the bot is ready to begin processing events.
        Load and process recent chat history from the allowed channels into the channel's deque cache.
        """
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
        """Event handler triggered when a message is sent in a channel."""
        # =============== INCOMING MESSAGE HANDLING ===============
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

        # Process the message content for OpenAI
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

        # Check if the bot was mentioned in the message
        if  not self.bot.user in message.mentions:
            return

        # Grab recent messages from cache and process for OpenAI query
        conversation_history = self.process_message_for_openai(channel_id)

        # Send the conversation history to OpenAI for a response
        response = await self.query_gregg(conversation_history)

        # Check if OpenAI returned a response
        if not response:
            self.logger.error("OpenAI response is None. Skipping message processing.")
            return

        # Query the assistant's replied action (regular message, YouTube search, or GIF search)
        action_dir = await self.process_openai_response(response)

        # Process the action (saves it to cache) and send message to discord
        message_to_send = await self.process_gregg_action(message, action_dir)
        await message.channel.send(message_to_send)

    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
        """Event handler for when a reaction is added to a message. 
        Fetch recent messages for training data and save to the training data file.

        Args:
            reaction (discord.Reaction): The reaction object added to the message.
            user (discord.User): The user who added the reaction.
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
        """Convert a Discord message to a format suitable for OpenAI processing by processing mentions, links, and image content.
        1. Checked if message is a pin and format accordingly
        2. Process mentions in the message content.
        3. Replace links with titles for both user and bot messages.
        4. Process image content by downloading, encoding, and sending to OpenAI.
        5. Return the processed message content.

        Args:
            message (discord.Message): The message object from Discord.
        Returns:
            str: The processed message content.
        """
        # 1. Check if message is a pin and format accordingly
        # if message.pinned:
        #     message_content = f"{message.author.display_name} pinned a message: {message.content}"
        #     self.logger.debug(f"Pinned message processed: {message_content}")
        # else:
        message_content = message.content
        self.logger.debug(f"Original message content: {message_content}")

        # 2. Process mentions in the message content
        message_content_without_mentions = self.replace_mentions(message_content, message.guild)
        self.logger.debug(f"After mention processing: {message_content_without_mentions}")

        # 3. Replace links with titles for user ^ bot messages
        is_bot = message.author.id == self.bot.user.id
        message_content_without_links = await self.process_links(message_content_without_mentions, is_bot)
        self.logger.debug(f"After link processing: {message_content_without_links} from_bot: {is_bot}")

        # 4. Process image content by downloading, encoding, and sending to OpenAI
        image_description = await self.process_image_content(message)
        if image_description:
            message_content_without_links += f" [Image description: {image_description}]"
        
        self.logger.debug(f"Final processed content: {message_content_without_links}")
        # 5. Return the processed message content
        return message_content_without_links

    async def process_links(self, message_content: str, from_bot: bool = False) -> str:
        """Replace or reformat links in the message content with descriptions and content markers, using 
        separate logic for user-generated versus bot-generated links. Titles/descriptions are fetched for
        YouTube, GIF, and general links. If the message already contains a formatted link using ~|~, it is skipped.

        Args:
            message_content (str): The content of the message.
            from_bot (bool): Whether the message is from the bot.
        Returns:
            str: The content with links replaced with titles and markers.
        """
        # Skip reformatting if the message already contains a formatted link using ~|~
        if re.search(r"\[.*? ~\|~ .*?\]\((YT_Video|GIF|LINK)\)", message_content):
            return message_content

        if from_bot:
            # Patterns for bot-generated links
            youtube_pattern = re.compile(r"\[(?!.*?~\|~).*?]\(<?(https?://(?:www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+)>?\)")
            gif_pattern = re.compile(r"\[(?!.*?~\|~).*?]\(<?(https?://\S*\.(?:giphy|tenor)\.com/\S+)>?\)")
            link_pattern = re.compile(r"\[(?!.*?~\|~).*?]\(<?(https?://\S+)>?\)")
        else:
            # Patterns for user-generated links (just URLs)
            youtube_pattern = re.compile(r"(https?://(?:www\.)?(youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11}))[^\s>]*")
            gif_pattern = re.compile(r"(https?://(?:\S+\.)?(?:giphy|tenor)\.com/\S+)")
            link_pattern = re.compile(r"(https?://\S+)")

        def clean_url(url):
            # Parse and rebuild URL without query or fragment
            parsed_url = urllib.parse.urlparse(url)
            return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

        # Process YouTube links
        for match in youtube_pattern.finditer(message_content):
            url = clean_url(match.group(1))
            title, author, description, _ = await self.search_youtube(url=url)
            if title and author:
                description_display = description if description else "No description available"
                formatted_youtube = f"[{title} ~|~ {author} ~|~ {description_display}](YT_Video)"
                message_content = message_content.replace(match.group(0), formatted_youtube)
                return message_content

        # Process GIF links
        for match in gif_pattern.finditer(message_content):
            url = clean_url(match.group(1))
            title, description, _ = await self.search_gif(url=url)
            description_display = description if description else "No description available"
            formatted_gif = f"[{title} ~|~ {description_display}](GIF)"
            message_content = message_content.replace(match.group(0), formatted_gif)
            return message_content

        # Process general links
        for match in link_pattern.finditer(message_content):
            url = clean_url(match.group(1))
            title, description = await self.fetch_link_data(url)
            description_display = description if description else "No description available"
            formatted_link = f"[{title} ~|~ {description_display}](LINK)"
            message_content = message_content.replace(match.group(0), formatted_link)
            return message_content

        # If no match
        return message_content

    async def process_image_content(self, message: discord.Message) -> Optional[str]:
        """Process image attachments in a message by downloading, encoding, and sending to OpenAI.
        
        Args:
            message (discord.Message): The message object with image attachments.
        Returns:
            str: The description of the image content, or None if no image attachments are found or if an error occurs.
        """
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
                                self.logger.warning(f"Failed to download image {attachment.filename} - HTTP {resp.status}")
                                return None

                    # Encode the downloaded image to base64
                    base64_image = await self.encode_image_base64(image_path)
                    if not base64_image:
                        self.logger.warning("Failed to encode image to base64.")
                        return None

                    # Prepare and send the request to OpenAI for image analysis
                    response = await self.client.chat.completions.create(
                        model=self.image_detection_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "What is in this image? Provide a succinct description useful for someone who can't see it."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                    },
                                ],
                            }
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
 
    def replace_mentions(self, message: str, guild: discord.Guild) -> str:
        """Replace all user, role, and channel mentions in the message content with their display names, 
        and remove mentions of the bot itself.
        
        Args:
            message (str): The content of the message.
            guild (discord.Guild): The guild to fetch members, roles, and channels from.
            
        Returns:
            str: The content with mentions replaced by display names and the bot's mention removed.
        """
        bot_id = self.bot.user.id  # Store the bot's ID for comparison
        
        # Replace any mentions in the response with display names for users, names for roles, or names for channels
        def replace_mention(match):
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
        msg_content = re.sub(r"<@!?(\d+)>|<@&(\d+)>|<#(\d+)>", replace_mention, message)

        return msg_content

    def process_message_for_openai(self, channel_id: int) -> ConversationHistory:
        """Process the recent messages from the channel for OpenAI, combining consecutive messages by role.

        Args:
            channel_id (int): The ID of the channel to process.
        Returns:
            list: The conversation history for the channel. Format: [{"role": str, "content": str}] (oldest messages first). If a system prompt is included, it should be the first message.
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

        self.logger.debug(f"Conversation history: {conversation_history}")

        # Insert the system prompt at the start of the conversation history
        conversation_history.insert(0, {"role": "system", "content": self.system_prompt})


        return conversation_history

# ==================== OpenAI Response Processing ====================

    async def query_gregg(self, conversation_history: ConversationHistory) -> Optional[OpenAIResponse]:
        """Send the conversation history to Gregg Limper (our OpenAI model) for a response.
        
        Args:
            conversation_history (list): The conversation history for the channel. Format: [{"role": str, "content": str}] (oldest messages first). If a system prompt is included, it should be the first message.
        Returns:
            OpenAIResponse: The response object from OpenAI.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.fine_tuned_model,
                messages=conversation_history,
                max_tokens=450,
                temperature=0.85,
                functions=[
                    {
                        "name": "search_youtube_video",
                        "description": "Retrieve a relevant YouTube video based on a user's query.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "search_term": {
                                    "type": "string",
                                    "description": "Query to search for on YouTube"
                                }
                            },
                            "required": ["search_term"]
                        }
                    },
                    {
                        "name": "search_gif",
                        "description": "Retrieve a relevant GIF based on a user's query.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "search_term": {
                                    "type": "string",
                                    "description": "Query to search_gif"
                                }
                            },
                            "required": ["search_term"]
                        }
                    }
                ]
            )
            return response
        except Exception as e:
            self.logger.error(f"Error querying Gregg Limper: {str(e)}")
            return None

    async def process_openai_response(self, response: Dict) -> GreggAction:
        """Process the response from OpenAI (our assistant's response), handling potential function calls

        Args:
            response (dict): The response object from OpenAI.
        Returns:
            dict: The assistant's reply in a dictionary with the following format:
            {
                "message": "The assistant's reply",
                "YouTube": {"title": "Video Title", "author": "Video Author", "description": "Video Description", "url": "Video URL"},
                "GIF": {"title": "GIF Title", "description": "GIF Description", "url": "GIF URL"}    
            }
        """
        # Check if OpenAI responded with a function call (e.g., YouTube search, GIF search)
        if response.choices[0].finish_reason == "function_call":
            action = response.choices[0].message.function_call
            if action:
                if action.name == "search_youtube_video":
                    arguments = json.loads(action.arguments)
                    search_term = arguments["search_term"]
                    self.logger.info(f"Executing YouTube search for '{search_term}'")
                    title, author, description, url = await self.search_youtube(query=search_term)
                    return {
                        "message": None,
                        "YouTube": {"title": title, "author": author, "description": description, "url": url},
                        "GIF": None
                    }
                elif action.name == "search_gif":
                    arguments = json.loads(action.arguments)
                    search_term = arguments["search_term"]
                    self.logger.info(f"Executing GIF search for '{search_term}'")
                    title, description, url = await self.search_gif(query=search_term)
                    return {
                        "message": None,
                        "YouTube": None,
                        "GIF": {"title": title, "description": description, "url": url}
                    }
        
        # Otherwise, process the normal assistant reply
        assistant_reply = response.choices[0].message.content.strip()
        return {
            "message": assistant_reply,
            "YouTube": None,
            "GIF": None
        }

    async def process_gregg_action(self, target_message: discord.Message, action: GreggAction) -> str:
        """Process the response from OpenAI and send the appropriate message to Discord. Saves the action to the channel's deque cache.

        Args:
            target_message (discord.Message): The message object to reply to.
            action (dict): The action object from OpenAI.
        Returns:
            str: The message content to send to Discord.
        """

        channel_id = target_message.channel.id
        # Prepare variables for message to send and to cache
        message_to_send = ""
        message_to_cache = ""

        # Process general message
        if action["message"]:
            # Replace mentions in the message content and process hallucinated links
            message_to_send = self.replace_mentions(action["message"], target_message.guild)
            message_to_send, message_to_cache = await self.process_hallucinated_link_reply(message_to_send, target_message.channel.id)

        # Process YouTube action
        elif action.get("YouTube"):
            title, author, description, url = action["YouTube"].values()
            if title and url:
                message_to_send = f"[{title}](<{url}>)"
                message_to_cache = f"[{title} ~|~ {author or 'Unknown Author'} ~|~ {description or 'No description available'}](YT_Video)"
            else:
                message_to_send = "YouTube search returned no results."
                message_to_cache = "[Error fetching Youtube data](YT_Video)"

        # Process GIF action
        elif action.get("GIF"):
            title, description, url = action["GIF"].values()
            if url:
                message_to_send = url
                message_to_cache = f"[{title} ~|~ {description or 'No description available'}](GIF)"
            else:
                message_to_send = "GIF search returned no results."
                message_to_cache = "[GIF Link Unavailable](GIF)"

        # Cache the processed message and return the message to send
        message_entry = {
            "displayname": "Gregg Limper",
            "role": "assistant",
            "content": message_to_cache,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.recent_message_cache[channel_id].appendleft(message_entry)

        return message_to_send
  
    async def process_hallucinated_link_reply(self, message: str, channel_id: int) -> Tuple[Optional[str], Optional[str]]:
        """Replace hallucinated links in the message content with real data from searches.
        Uses regex to detect if the bot is using the [...](Marker) format or sending raw links.
        Args:
            message (str): The message content to process.
            channel_id (int): The ID of the channel the message is from.
        Returns:
            tuple[str, str]: The message content with hallucinated links replaced with real data [message_to_send, message_to_cache]
        """
        message_to_send = message
        message_to_cache = message

        # Log hallucinated message
        self.logger.debug(f"Hallucinated message: {message}")

        # Process hallucinated links with markers ([...](Marker))
        youtube_match = re.search(r"\[.*\]\(YT_Video\)", message)
        gif_match = re.search(r"\[.*\]\(GIF\)", message)
        link_match = re.search(r"\[.*\]\(LINK\)", message)

        if youtube_match:
            # Replace the hallucinated YouTube link with real data
            self.logger.info("Processing hallucinated YouTube link")
            content_parts = youtube_match.group(0).split("~|~")
            query = " ".join(part.strip("[]") for part in content_parts[:2])  # Extract title and author if available
            message_to_send, message_to_cache = await self._process_youtube_link(query=query)

        elif gif_match:
            # Replace the hallucinated GIF link with real data
            self.logger.info("Processing hallucinated GIF link")
            query = gif_match.group(0).split("~|~")[0][1:].strip()  # Extract title as query
            message_to_send, message_to_cache = await self._process_gif_link(query=query)

        elif link_match:
            # Replace the hallucinated generic link with real data
            self.logger.info("Processing hallucinated generic link")
            query = link_match.group(0).split("~|~")[0][1:].strip()  # Extract title as query
            message_to_send, message_to_cache = await self._process_generic_link(query=query)

        else:
            # Process raw links without markers
            self.logger.info("Processing raw link without markers")

            # Find all raw URLs in the message (https:// or http://)
            raw_url_pattern = re.compile(r"https?://[^\s]+")
            for raw_url_match in raw_url_pattern.finditer(message):
                raw_url = raw_url_match.group(0)

                # Generate a relevant search query based on the recent messages in the channel
                query = await self.request_search_query(channel_id)
                self.logger.debug(f"Generated query for raw link {raw_url}: {query}")

                # Process the link based on the domain
                if "youtube.com" in raw_url or "youtu.be" in raw_url:
                    youtube_send, youtube_cache = await self._process_youtube_link(query=query)
                    # Replace raw URL in both send and cache messages
                    message_to_send = message_to_send.replace(raw_url, youtube_send)
                    message_to_cache = message_to_cache.replace(raw_url, youtube_cache)
                    self.logger.debug(f"Processed YouTube link: {youtube_send}")

                elif "giphy.com" in raw_url or "tenor.com" in raw_url:
                    gif_send, gif_cache = await self._process_gif_link(query=query)
                    message_to_send = message_to_send.replace(raw_url, gif_send)
                    message_to_cache = message_to_cache.replace(raw_url, gif_cache)
                    self.logger.debug(f"Processed GIF link: {gif_send}")

                else:
                    generic_send, generic_cache = await self._process_generic_link(query=query)
                    message_to_send = message_to_send.replace(raw_url, generic_send)
                    message_to_cache = message_to_cache.replace(raw_url, generic_cache)
                    self.logger.debug(f"Processed generic link: {generic_send}")

            # Final debug logs for message content
            self.logger.debug(f"Final message to send: {message_to_send}")
            self.logger.debug(f"Final message to cache: {message_to_cache}")

        return message_to_send, message_to_cache

# ==================== Search Functions ====================

    async def fetch_link_data(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetch the title of a webpage using BeautifulSoup.
        
        Args:
            url (str): The URL of the webpage.
        Returns:
            tuple: The title and description of the webpage, or None if an error occurs.
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
                        return None, None
        except Exception as e:
            self.logger.error(f"Error fetching link data: {str(e)}")
            return None, None

    async def search_google(self, query: str, num_results=1) -> List[Dict[str, str]]:
        """Search Google and return the top search result.

        Args:
            query (str): The search query for Google.
        Returns:
            list[dict]: The top search results from Google (up to num_results), or empty list if no results are found.
        """
        # Perform a Google search and return the top search result
        try:
            response = self.google_client.cse().list(q=query, cx=self.search_engine_id, num=num_results).execute()
            results = [{"title": item.get('title'), "link": item.get('link'), "snippet": item.get('snippet', 'No description available')} for item in response.get("items", [])]
            if results:
                return results
            else:
                self.logger.warning(f"No Google results found for query: {query}")
                return []
        except Exception as e:
            self.logger.error(f"Error searching Google: {str(e)}")
            return []

    async def search_gif(self, query: Optional[str] = None, url: Optional[str] = None) -> GIFData:
        """Search Giphy and return the GIF title, description (if available), and URL.
        
        Args:
            query (str): The search query for Giphy.
            url (str): The GIF URL.

        Returns:
            tuple: The title, description, and URL of the top GIF result, or None if no results are found.
        """
        if not (query or url):
            self.logger.warning("Must specify either query or url for search_gif")
            return None, None, None
        try:
            # Search Giphy with query if provided
            if query:
                # Limit length to prevent 414 errors
                query = query[:100]
                encoded_query = urllib.parse.quote(query)
                giphy_url = f"https://api.giphy.com/v1/gifs/search?api_key={self.giphy_api_key}&q={encoded_query}&limit=1"
                async with aiohttp.ClientSession() as session:
                    async with session.get(giphy_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            gif_data = data.get("data", [])
                            if gif_data:
                                title = gif_data[0].get("title", "No title")
                                url = gif_data[0].get("url", None)
                                tags = gif_data[0].get("tags", [])
                                description = ", ".join(tags) if tags else "No tags available"
                                return title, description, url
                            else:
                                self.logger.warning(f"No Giphy results found for query: {query}")
                                return None, None, None
                        else:
                            self.logger.error(f"Error searching Giphy: HTTP {response.status}")
                            return None, None, None

            # Handle direct URL case if provided
            elif url:
                title, description = await self.fetch_link_data(url)
                return title, description, url

        except Exception as e:
            self.logger.error(f"Error searching Giphy: {str(e)}")
            return None, None, None

    async def search_youtube(self, query: Optional[str] = None, url: Optional[str] = None) -> YouTubeData:
        """Search Youtube and return the top video result.

        Args:
            query (str): The search query for Youtube.
            url (str): The Youtube video URL.
        Returns:
            tuple: The title, author, description, and url of the top Youtube video result, or None if no results are found.
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
                    # Summarize the description with Gregg for a more concise response
                    description = await self.summarize_description(snippet.get("description"))
                    return title, author, description, url
                else:
                    self.logger.warning(f"No YouTube results found for query: {query}")
                    return None, None, None, None

            # Fetch video details by URL
            elif url:
                video_id = re.search(r"(?:v=|/)([A-Za-z0-9_-]{11})", url)
                if not video_id:
                    self.logger.warning("Invalid YouTube URL format.")
                    return None, None, None, None
                video_id = video_id.group(1)

                request = self.youtube_client.videos().list(part="snippet", id=video_id)
                response = request.execute()
                if items := response.get("items"):
                    snippet = items[0].get("snippet", {})
                    title = snippet.get("title")
                    author = snippet.get("channelTitle")
                    # Summarize the description with Gregg for a more concise response
                    description = await self.summarize_description(snippet.get("description"))
                    return title, author, description, url
                else:
                    self.logger.warning(f"No YouTube results found for URL: {url}")
                    return None, None, None, None

            else:
                self.logger.warning("No query or URL provided for YouTube search.")
                return None, None, None, None

        except Exception as e:
            self.logger.error(f"Error searching YouTube: {str(e)}")
            return None, None, None, None

    async def request_search_query(self, channel_id: int) -> str:
        """Generate a search query based on the recent messages in the channel"""

        # Get the recent messages in reverse order to maintain chronological order
        recent_messages = list(self.recent_message_cache[channel_id])[:self.assistant_context_length]
        recent_messages.reverse()

        # Format the conversation history for OpenAI
        conversation_history = self.process_message_for_openai(channel_id)

        # Send to OpenAI to generate a query
        try:
            response = self.client.chat.completions.create(
                model=self.fine_tuned_model,
                messages=conversation_history,
                max_tokens=20,
                temperature=0.6
            )

            # Extract the suggested search term from the response
            search_query = response.choices[0].message.content.strip() if response.choices else "default search query"
            self.logger.debug(f"Generated search query: {search_query}")
            return search_query

        except Exception as e:
            self.logger.error(f"Error generating search query: {e}")
            return "default search query"

    async def summarize_description(self, description: str) -> str:
        """Generate a succinct summary of a long YouTube/other description using OpenAI.

        Args:
            description (str): The full description of a YouTube video.

        Returns:
            str: A succinct summary of the description.
        """
        try:
            # Construct a prompt for generating a succinct summary
            prompt = (
                "Create a concise, one- to two-sentence summary for the following YouTube video description:\n\n"
                f"{description}\n\n"
                "Summary:"
            )

            # Send the prompt to the OpenAI API for summarization
            response = self.client.chat.completions.create(
                model=self.fine_tuned_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.7
            )

            # Extract and return the summary if it exists
            summary = response.choices[0].message.content.strip() if response.choices else "No summary available"
            return summary

        except Exception as e:
            self.logger.error(f"Error summarizing description: {str(e)}")
            return "No summary available" 

# ==================== Utility Functions ====================

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
            show_names (bool): Whether to display the display names of users or not (defaults to IDs)

        Returns:
            dict: Cache contents with either channel IDs or names based on the show_names flag.
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

    async def encode_image_base64(self, image_path: str) -> str:
        """Encode an image at the specified path to base64.
        
        Args:
            image_path (str): The path to the image file.
        Returns:
            str: The base64 encoded image.
        """
        async with aiofiles.open(image_path, "rb") as image_file:
            return base64.b64encode(await image_file.read()).decode('utf-8')
        
    async def _process_youtube_link(self, query: str) -> Tuple[str, str]:
        """Search YouTube for the given query and format the response for the Discord message and cache entry.

        Args:
            query (str): The search query for YouTube.
        Returns:
            tuple: The formatted YouTube link and cache entry (message_to_send, message_to_cache).
        """
        title, author, description, url = await self.search_youtube(query=query)
        
        if title and author and url:
            # Clean up title, author, and description to remove newlines and extra spaces
            title = title.replace("\n", " ").strip()
            author = author.replace("\n", " ").strip()
            description_display = (description or "No description available").replace("\n", " ").strip()

            # Format for message and cache
            message_to_send = f"[{title}](<{url}>)"
            message_to_cache = f"[{title} ~|~ {author} ~|~ {description_display}](YT_Video)"
            
            return message_to_send, message_to_cache
        else:
            return "YouTube search returned no results.", "[Error fetching YouTube data](YT_Video)"

    async def _process_gif_link(self, query: str) -> Tuple[str, str]:
        """Search GIf for the given query and format the response for the discord message and cache entry.
        
        Args:
            query (str): The search query for GIF.
        Returns:
            tuple: The formatted YouTube link and cache entry (message_to_send, message_to_cache).
        """
        title, description, url = await self.search_gif(query=query)
        if url:
            description_display = description if description else "No description available"
            return url, f"[{title} ~|~ {description_display}](GIF)"
        else:
            return "GIF search returned no results.", "[GIF Link Unavailable](GIF)"

    async def _process_generic_link(self, query: str, num_results=1) -> Tuple[str, str]:
        """Search Google for the given query and format the response for the discord message and cache entry.
        
        Args:
            query (str): The search query for Google.
        Returns:
            tuple: The formatted YouTube link and cache entry (message_to_send, message_to_cache).
        """
        results = await self.search_google(query, num_results=num_results)
        for result in results:
            if result["title"] and result["link"]:
                description_display = result["snippet"] if result["snippet"] else "No description available"
                return f"[{result['title']}](<{result['link']}>)", f"[{result['title']} ~|~ {description_display}](LINK)"
        return "Google search returned no results.", "[Link Unavailable](LINK)"