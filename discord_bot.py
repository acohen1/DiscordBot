import os
import discord
import json
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from privtoken import OPENAI_API_KEY, DISCORD_API_TOKEN
import asyncio

# Constants
FINE_TUNED_MODEL = "ft:gpt-4o-2024-08-06:personal:gregg-limper:AN9TcxoD"
ALLOWED_CHANNEL_IDS = [1299973475959705631, 662531201097007121]  # Add your allowed channel IDs here
HISTORY_DIR = "active_conversations"
TIME_LIMIT = timedelta(minutes=30)
MAX_HISTORY_LENGTH = 20  # Maximum number of messages to retain per channel
SYSTEM_PROMPT = """ 
You are Gregg Limper.

You will recieve a series of messages from different users. You are to respond to them as if you were in the shoes of Gregg Limper.

You love watching DOOM speedruns, study all the WADs, and are subscribed to a Youtuber named "Brainfreezzzzz".
You are a fan of Lee Scratch Perry; an overzealous supporter.

Pull in other relevant messages from other users. Ignore them if they are not relevant to the question at hand.
"""

# Initialize the OpenAI and Discord clients
client = OpenAI(api_key=OPENAI_API_KEY)
intents = discord.Intents.default()
intents.message_content = True
discord_client = discord.Client(intents=intents)

# Ensure the directory for conversation history files exists
os.makedirs(HISTORY_DIR, exist_ok=True)

def get_history_file_path(channel_id):
    """Get the file path for a channel's conversation history."""
    return os.path.join(HISTORY_DIR, f"{channel_id}.json")

def save_conversation_history(channel_id, channel_name, conversation_history):
    """Save conversation history for a specific channel, truncating if necessary."""
    # Truncate history to the max length if it exceeds
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]

    # Save the truncated history
    file_path = get_history_file_path(channel_id)
    data = {
        "channel_id": channel_id,
        "channel_name": channel_name,
        "messages": conversation_history
    }
    with open(file_path, "w") as f:
        json.dump(data, f, default=str, indent=4)

async def periodic_cleanup():
    """Clear old conversation histories periodically."""
    while True:
        for filename in os.listdir(HISTORY_DIR):
            file_path = os.path.join(HISTORY_DIR, filename)
            with open(file_path, "r") as f:
                data = json.load(f)
                conversation_history = data.get("messages", [])

            # Remove messages older than TIME_LIMIT
            current_time = datetime.now(timezone.utc)
            conversation_history = [
                msg for msg in conversation_history if current_time - datetime.fromisoformat(msg["timestamp"]) <= TIME_LIMIT
            ]

            # Save the updated history, truncated to max length
            save_conversation_history(data["channel_id"], data["channel_name"], conversation_history)

        # Wait an hour before the next cleanup
        await asyncio.sleep(3600)

@discord_client.event
async def on_ready():
    """Event handler for when the bot is ready to start receiving events."""
    print(f'We have logged in as {discord_client.user}')

    # Start the periodic cleanup task
    discord_client.loop.create_task(periodic_cleanup())

    # Load recent history for each allowed channel
    for channel_id in ALLOWED_CHANNEL_IDS:
        channel = discord_client.get_channel(channel_id)
        if channel:
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
            save_conversation_history(channel_id, channel.name, conversation_history)

@discord_client.event
async def on_message(message):
    """Event handler for when a message is sent in a channel."""
    if message.author == discord_client.user:
        return

    if message.channel.id not in ALLOWED_CHANNEL_IDS:
        return

    channel_id = message.channel.id
    content_with_usernames = message.content.replace(f"<@{discord_client.user.id}>", "").strip()

    for user in message.mentions:
        content_with_usernames = content_with_usernames.replace(f"<@{user.id}>", user.display_name)

    # Load the recent conversation history directly from the JSON file, if available
    conversation_history = []
    history_file = get_history_file_path(channel_id)
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            data = json.load(f)
            conversation_history = data.get("messages", [])
            channel_name = data.get("channel_name", str(channel_id))

    # Add the new message to the conversation history
    conversation_history.append({
        "role": "user",
        "content": f"{message.author.name} says: {content_with_usernames}",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # Filter and truncate the conversation history before saving
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
                temperature=0.8,
            )

            assistant_reply = response.choices[0].message.content.strip()
            conversation_history.append({"role": "assistant", "content": assistant_reply, "timestamp": datetime.now(timezone.utc).isoformat()})
            save_conversation_history(channel_id, channel_name, conversation_history)

            await message.channel.send(assistant_reply)

        except Exception as e:
            await message.channel.send(f"Error: {str(e)}")

# Run the Discord bot
discord_client.run(DISCORD_API_TOKEN)
