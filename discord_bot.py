import os
import json
import asyncio
from datetime import datetime, timedelta, timezone
import discord
from discord.ext import commands
from openai import OpenAI
from privtoken import OPENAI_API_KEY, DISCORD_API_TOKEN

class GreggLimperBot:
    # Constants and Configurations
    FINE_TUNED_MODEL = "ft:gpt-4o-2024-08-06:personal:gregg-limper:AN9TcxoD"
    ALLOWED_CHANNEL_IDS = [1299973475959705631, 662531201097007121]
    HISTORY_DIR = "active_conversations"
    TRAINING_DATA_DIR = "new_training_data"
    TRAINING_DATA_FILE = os.path.join(TRAINING_DATA_DIR, "new_training_data.json")
    TIME_LIMIT = timedelta(minutes=30)
    MAX_HISTORY_LENGTH = 10
    SYSTEM_PROMPT = """ 
    You are Gregg Limper.
    ... (your full system prompt here)
    """
    
    def __init__(self):
        # Setup Directories
        os.makedirs(self.HISTORY_DIR, exist_ok=True)
        os.makedirs(self.TRAINING_DATA_DIR, exist_ok=True)
        
        # Initialize Discord Client and OpenAI API
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        self.bot = discord.Client(intents=intents)

        # Event Bindings
        self.bot.event(self.on_ready)
        self.bot.event(self.on_message)
        self.bot.event(self.on_reaction_add)
    
    # Utility Functions
    def get_history_file_path(self, channel_id):
        """Path to the JSON file for channel conversation history."""
        return os.path.join(self.HISTORY_DIR, f"{channel_id}.json")
    
    def save_conversation_history(self, channel_id, channel_name, conversation_history):
        """Saves conversation history, truncating if necessary."""
        if len(conversation_history) > self.MAX_HISTORY_LENGTH:
            conversation_history = conversation_history[-self.MAX_HISTORY_LENGTH:]
        
        data = {
            "channel_id": channel_id,
            "channel_name": channel_name,
            "messages": conversation_history
        }
        with open(self.get_history_file_path(channel_id), "w") as f:
            json.dump(data, f, indent=4)
    
    def save_to_training_data(self, messages, message_id):
        """Save new conversation entry to training data if it's unique."""
        data_entry = {"message_id": message_id, "messages": messages}
        
        # Load existing training data
        if os.path.exists(self.TRAINING_DATA_FILE):
            with open(self.TRAINING_DATA_FILE, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # Skip if the message ID already exists
        if any(entry.get("message_id") == message_id for entry in existing_data):
            print(f"Duplicate message ID {message_id} found; skipping.")
            return

        # Append new entry and save back to JSON
        existing_data.append(data_entry)
        with open(self.TRAINING_DATA_FILE, "w") as f:
            json.dump(existing_data, f, indent=4)
        print(f"Saved new training data: {data_entry['messages'][-1]['content']}")

    async def clear_conversation_history(self, channel_id, channel_name):
        """Clear conversation history without deleting the file."""
        file_path = self.get_history_file_path(channel_id)
        if os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump({"channel_id": channel_id, "channel_name": channel_name, "messages": []}, f, indent=4)
            return True
        return False

    async def periodic_cleanup(self):
        """Periodically clear old conversation histories."""
        while True:
            for filename in os.listdir(self.HISTORY_DIR):
                file_path = os.path.join(self.HISTORY_DIR, filename)
                with open(file_path, "r") as f:
                    data = json.load(f)
                current_time = datetime.now(timezone.utc)
                conversation_history = [
                    msg for msg in data.get("messages", [])
                    if current_time - datetime.fromisoformat(msg["timestamp"]) <= self.TIME_LIMIT
                ]
                self.save_conversation_history(data["channel_id"], data["channel_name"], conversation_history)
            await asyncio.sleep(3600)

    # Event Handlers
    async def on_ready(self):
        print(f'Logged in as {self.bot.user}')
        self.bot.loop.create_task(self.periodic_cleanup())
        
        # Load recent history on startup
        for channel_id in self.ALLOWED_CHANNEL_IDS:
            channel = self.bot.get_channel(channel_id)
            if channel:
                conversation_history = []
                async for message in channel.history(limit=self.MAX_HISTORY_LENGTH):
                    if datetime.now(timezone.utc) - message.created_at <= self.TIME_LIMIT:
                        conversation_history.append({
                            "role": "user" if message.author != self.bot.user else "assistant",
                            "content": message.content,
                            "timestamp": message.created_at.isoformat()
                        })
                conversation_history.reverse()
                self.save_conversation_history(channel_id, channel.name, conversation_history)

    async def on_message(self, message):
        if message.author == self.bot.user or message.channel.id not in self.ALLOWED_CHANNEL_IDS:
            return
        
        # Check for /lobotomy command
        if message.content.startswith(f"<@{self.bot.user.id}>") and message.content.strip().endswith("/lobotomy"):
            cleared = await self.clear_conversation_history(message.channel.id, message.channel.name)
            await message.channel.send("All gone <:brainlet:1300560937778155540>" if cleared else "No history found.")
            return

        # Process regular messages and save conversation
        conversation_history = []
        history_file = self.get_history_file_path(message.channel.id)
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                data = json.load(f)
                conversation_history = data.get("messages", [])
        
        conversation_history.append({
            "role": "user",
            "content": message.content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        self.save_conversation_history(message.channel.id, message.channel.name, conversation_history)

        # Check if bot is mentioned for OpenAI response
        if self.bot.user in message.mentions:
            messages_for_api = [{"role": "system", "content": self.SYSTEM_PROMPT}]
            messages_for_api.extend({"role": msg["role"], "content": msg["content"]} for msg in conversation_history)
            try:
                response = self.client.chat.completions.create(
                    model=self.FINE_TUNED_MODEL,
                    messages=messages_for_api,
                    max_tokens=450,
                    temperature=0.85,
                )
                assistant_reply = response.choices[0].message.content.strip()
                await message.channel.send(assistant_reply)
                conversation_history.append({
                    "role": "assistant",
                    "content": assistant_reply,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                self.save_conversation_history(message.channel.id, message.channel.name, conversation_history)
            except Exception as e:
                await message.channel.send(f"Error: {str(e)}")

    async def on_reaction_add(self, reaction, user):
        if user == self.bot.user or reaction.message.author != self.bot.user:
            return

        # Fetch recent messages for training data
        conversation_history = []
        async for msg in reaction.message.channel.history(limit=5, before=reaction.message.created_at):
            conversation_history.insert(0, {
                "role": "user" if msg.author != self.bot.user else "assistant",
                "content": msg.content,
                "timestamp": msg.created_at.isoformat()
            })
        
        conversation_history.append({
            "role": "assistant",
            "content": reaction.message.content,
            "timestamp": reaction.message.created_at.isoformat()
        })
        self.save_to_training_data(conversation_history, reaction.message.id)

    # Start the bot
    def run(self):
        self.bot.run(DISCORD_API_TOKEN)

# Initialize and run the bot
gregg_limper_bot = GreggLimperBot()
gregg_limper_bot.run()
