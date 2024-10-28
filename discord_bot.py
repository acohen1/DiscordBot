import os
import discord
from discord.ext import commands
import json
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from privtoken import OPENAI_API_KEY, DISCORD_API_TOKEN
import asyncio

FINE_TUNED_MODEL = "ft:gpt-4o-2024-08-06:personal:gregg-limper:AN9TcxoD"
ALLOWED_CHANNEL_IDS = [1299973475959705631, 662531201097007121]
HISTORY_DIR = "active_conversations"
TRAINING_DATA_DIR = "new_training_data"
TRAINING_DATA_FILE = os.path.join(TRAINING_DATA_DIR, "new_training_data.json")
TIME_LIMIT = timedelta(minutes=30)
MAX_HISTORY_LENGTH = 10
SYSTEM_PROMPT = """ 
You are Gregg Limper.

You will recieve a series of messages from different users. Respond to them as if you were in the shoes of Gregg Limper.
Gregg Limper has the following characteristics:
- You love watching DOOM speedruns, study all the WADs, and are subscribed to a Youtuber named "Brainfreezzzzz".
- You play melee. You are a Captain Falcon main. You are a fan of Mew2King.
- You watch pokimane and are a fan of her. You are a tier 3 sub. You also watch Emiru.
- You play counter strike 2 and know a lot about the game, including map callouts, smoke spots, and strategies.
- You are a thug.

Pull in information from other user messages if relevant to your response. Refrain from sending links or images.

Under no circumstance, should you prefix your message with "username says: " or "Gregg Limper says: ".
"""

# Create directories if they don't exist
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

# Initialize OpenAI and Discord clients
client = OpenAI(api_key=OPENAI_API_KEY)
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
discord_client = discord.Client(intents=intents)

def get_history_file_path(channel_id):
    """Get the file path for a channel's conversation history."""
    return os.path.join(HISTORY_DIR, f"{channel_id}.json")

def save_conversation_history(channel_id, channel_name, conversation_history):
    """Save conversation history for a specific channel, truncating if necessary."""
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]
    file_path = get_history_file_path(channel_id)
    data = {
        "channel_id": channel_id,
        "channel_name": channel_name,
        "messages": conversation_history
    }
    with open(file_path, "w") as f:
        json.dump(data, f, default=str, indent=4)

def save_to_training_data(messages, message_id):
    """Append a new entry to the training data JSON file if it doesn't already exist."""
    data_entry = {
        "message_id": message_id,  # Track unique message ID to prevent duplicates
        "messages": [
            {"role": msg["role"], "content": msg["content"], "timestamp": msg["timestamp"]}
            for msg in messages
        ]
    }
    # Load existing training data or create a new list
    if os.path.exists(TRAINING_DATA_FILE):
        with open(TRAINING_DATA_FILE, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Check if this message ID is already in the training data
    if any(entry.get("message_id") == message_id for entry in existing_data):
        print(f"Message ID {message_id} is already saved in training data. Skipping duplicate.")
        return  # Skip saving if the message ID already exists

    # Append the new entry and save back to JSON
    existing_data.append(data_entry)
    with open(TRAINING_DATA_FILE, "w") as f:
        json.dump(existing_data, f, indent=4)
    print(f"Saved new message to training data: {data_entry['messages'][-1]['content']}")

async def clear_conversation_history(channel_id, channel_name):
    """Clear the conversation history for a specific channel without deleting the file."""
    file_path = get_history_file_path(channel_id)
    if os.path.exists(file_path):
        # Write an empty JSON structure to clear out the content
        with open(file_path, "w") as f:
            json.dump({"channel_id": channel_id, "channel_name": channel_name, "messages": []}, f, indent=4)
        return True
    return False

async def periodic_cleanup():
    """Clear old conversation histories periodically."""
    while True:
        for filename in os.listdir(HISTORY_DIR):
            file_path = os.path.join(HISTORY_DIR, filename)
            with open(file_path, "r") as f:
                data = json.load(f)
                conversation_history = data.get("messages", [])
            current_time = datetime.now(timezone.utc)
            conversation_history = [msg for msg in conversation_history if current_time - datetime.fromisoformat(msg["timestamp"]) <= TIME_LIMIT]
            save_conversation_history(data["channel_id"], data["channel_name"], conversation_history)
        await asyncio.sleep(3600)

@discord_client.event
async def on_ready():
    print(f'We have logged in as {discord_client.user}')

    discord_client.loop.create_task(periodic_cleanup())

    for channel_id in ALLOWED_CHANNEL_IDS:
        channel = discord_client.get_channel(channel_id)
        if channel:
            print (f"Fetching message history for channel: {channel.name}")
            conversation_history = []
            async for message in channel.history(limit=MAX_HISTORY_LENGTH):
                if datetime.now(timezone.utc) - message.created_at <= TIME_LIMIT:
                    content_with_usernames = message.content
                    # Remove bot mention
                    content_with_usernames = content_with_usernames.replace(f"<@{discord_client.user.id}>", "").strip()
                    # Replace all other mentions with usernames
                    for user in message.mentions:
                        content_with_usernames = content_with_usernames.replace(f"<@{user.id}>", user.display_name)
                    # Add message to history
                    conversation_history.append({
                        "role": "user" if message.author != discord_client.user else "assistant",
                        "content": f"{message.author.name} says: {content_with_usernames}" if message.author != discord_client.user else content_with_usernames,
                        "timestamp": message.created_at.isoformat()
                    })
            # Save the fetched history to JSON
            conversation_history.reverse()
            save_conversation_history(channel_id, channel.name, conversation_history)

@discord_client.event
async def on_message(message):
    if message.author == discord_client.user:
        return
    if message.channel.id not in ALLOWED_CHANNEL_IDS:
        return
    
    # Check for @GreggLimper /lobotomy command to clear conversation history
    if (
        message.content.startswith(f"<@{discord_client.user.id}>") and 
        message.content.strip().endswith("/lobotomy")
    ):
        # Clear conversation history for the channel
        cleared = await clear_conversation_history(message.channel.id, message.channel.name)
        if cleared:
            await message.channel.send("All gone <:brainlet:1300560937778155540>")
            print(f"Conversation history cleared for channel: {message.channel.name}")
        else:
            print(f"No conversation history to clear for channel: {message.channel.name}")
        return

    channel_id = message.channel.id
    content_with_usernames = message.content.replace(f"<@{discord_client.user.id}>", "").strip()
    for user in message.mentions:
        content_with_usernames = content_with_usernames.replace(f"<@{user.id}>", user.display_name)

    conversation_history = []
    history_file = get_history_file_path(channel_id)
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            data = json.load(f)
            conversation_history = data.get("messages", [])
            channel_name = data.get("channel_name", str(channel_id))

    conversation_history.append({
        "role": "user",
        "content": f"{message.author.name} says: {content_with_usernames}",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    current_time = datetime.now(timezone.utc)
    conversation_history = [
        msg for msg in conversation_history if current_time - datetime.fromisoformat(msg["timestamp"]) <= TIME_LIMIT
    ]
    save_conversation_history(channel_id, channel_name, conversation_history)

    if discord_client.user in message.mentions:
        messages_for_api = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages_for_api.extend({"role": msg["role"], "content": msg["content"]} for msg in conversation_history)

        try:
            response = client.chat.completions.create(
                model=FINE_TUNED_MODEL,
                messages=messages_for_api,
                max_tokens=450,
                temperature=0.85,
            )
            assistant_reply = response.choices[0].message.content.strip()
            conversation_history.append({"role": "assistant", "content": assistant_reply, "timestamp": datetime.now(timezone.utc).isoformat()})
            save_conversation_history(channel_id, channel_name, conversation_history)
            await message.channel.send(assistant_reply)
        except Exception as e:
            await message.channel.send(f"Error: {str(e)}")

@discord_client.event
async def on_reaction_add(reaction, user):
    """Event handler for when a user reacts to a message."""
    if user == discord_client.user:
        return

    message = reaction.message
    if message.author == discord_client.user and message.channel.id in ALLOWED_CHANNEL_IDS:
        # Initialize conversation history list
        conversation_history = []

        # Fetch the last 5 messages before the reacted message, ordered newest first by default
        async for msg in message.channel.history(limit=5, before=message.created_at):
            conversation_history.insert(0, {  # Insert at the beginning to maintain chronological order
                "role": "user" if msg.author != discord_client.user else "assistant",
                "content": f"{msg.author.name} says: {msg.content}" if msg.author != discord_client.user else msg.content,
                "timestamp": msg.created_at.isoformat()
            })

        # Ensure the first message in the sequence is from a "user"
        while conversation_history and conversation_history[0]["role"] != "user":
            conversation_history.pop(0)  # Remove assistant messages from the beginning if present

        # Append the reacted message itself at the end
        conversation_history.append({
            "role": "assistant",
            "content": message.content,
            "timestamp": message.created_at.isoformat()
        })

        # Save the conversation history to training data only if it's unique
        save_to_training_data(conversation_history, message.id)
    
# Run the Discord bot
discord_client.run(DISCORD_API_TOKEN)
