import os
import discord
import json
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from privtoken import OPENAI_API_KEY, DISCORD_API_TOKEN

# Constants
FINE_TUNED_MODEL = "ft:gpt-4o-2024-08-06:personal:gregg-limper:AN9TcxoD"
ALLOWED_CHANNEL_IDS = [1299973475959705631, 662531201097007121]  # Add your allowed channel IDs here
HISTORY_DIR = "active_conversations"
TIME_LIMIT = timedelta(minutes=60)
SYSTEM_PROMPT = """
You are Gregg Limper, a knowledgeable and approachable assistant. Your role is to engage in thoughtful, insightful conversations, providing accurate, relevant, and well-articulated responses to user requests. 

- Strive for clarity and helpfulness in each response.
- Maintain a friendly, professional tone while adapting your language to suit the user's level of familiarity with the topic.
- If asked for complex information, provide a clear, concise explanation or breakdown.
- For follow-up questions, demonstrate continuity and awareness of prior messages to ensure cohesive conversation flow.
- Ask clarifying questions if user requests are ambiguous to ensure you provide the most useful information.
"""

# Create the directory for conversation history files and initialize the OpenAI, Discord clients
os.makedirs(HISTORY_DIR, exist_ok=True)
client = OpenAI(api_key=OPENAI_API_KEY)
intents = discord.Intents.default()
intents.message_content = True
discord_client = discord.Client(intents=intents)

def get_history_file_path(channel_id):
    """Get the file path for a channel's conversation history."""
    return os.path.join(HISTORY_DIR, f"{channel_id}.json")

def save_conversation_history(channel_id, channel_name, conversation_history):
    """Save conversation history for a specific channel with its name."""
    file_path = get_history_file_path(channel_id)
    data = {
        "channel_id": channel_id,
        "channel_name": channel_name,
        "messages": conversation_history
    }
    with open(file_path, "w") as f:
        json.dump(data, f, default=str, indent=4)

@discord_client.event
async def on_ready():
    """Event handler for when the bot is ready to start receiving events."""

    print(f'We have logged in as {discord_client.user}')

    # Fetch the channels where the bot should respond
    for channel_id in ALLOWED_CHANNEL_IDS:
        channel = discord_client.get_channel(channel_id)
        if channel:
            conversation_history = []
            async for message in channel.history(limit=100):
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

    # Ignore messages sent by the bot itself
    if message.author == discord_client.user:
        return

    # Track messages only if the channel ID is in the ALLOWED_CHANNEL_IDS list
    if message.channel.id not in ALLOWED_CHANNEL_IDS:
        return

    channel_id = message.channel.id
    content_with_usernames = message.content

    # Remove the bot's own mention from messages
    content_with_usernames = content_with_usernames.replace(f"<@{discord_client.user.id}>", "").strip()

    # Replace all other mentions with usernames
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

    # Filter messages to retain only those within the TIME_LIMIT
    current_time = datetime.now(timezone.utc)
    conversation_history[:] = [
        msg for msg in conversation_history if current_time - datetime.fromisoformat(msg["timestamp"]) <= TIME_LIMIT
    ]

    # Sort and save history for the current channel
    conversation_history.sort(key=lambda msg: msg["timestamp"])
    save_conversation_history(channel_id, channel_name, conversation_history)

    # Respond only if the bot is mentioned
    if discord_client.user in message.mentions:
        # Prepare the message content for the API call, removing the bot's mention
        content_with_usernames = message.content.replace(f"<@{discord_client.user.id}>", "").strip()
        for user in message.mentions:
            content_with_usernames = content_with_usernames.replace(f"<@{user.id}>", user.display_name)

        # Prepare the messages list for the API call, starting with the system prompt
        messages_for_api = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages_for_api.extend({"role": msg["role"], "content": msg["content"]} for msg in conversation_history)

        try:
            # Make the API call to OpenAI
            response = client.chat.completions.create(
                model=FINE_TUNED_MODEL,
                messages=messages_for_api,
                max_tokens=150,
                temperature=1.0,
            )

            # Append the assistant's reply to the conversation history and save
            assistant_reply = response.choices[0].message.content.strip()
            conversation_history.append({"role": "assistant", "content": assistant_reply, "timestamp": datetime.now(timezone.utc).isoformat()})
            save_conversation_history(channel_id, channel_name, conversation_history)

            await message.channel.send(assistant_reply)

        except Exception as e:
            await message.channel.send(f"Error: {str(e)}")

# Run the Discord bot
discord_client.run(DISCORD_API_TOKEN)
