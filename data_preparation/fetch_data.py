import discord
from discord.ext import commands
import asyncio
import json
from privtoken import DISCORD_API_TOKEN

TOKEN = DISCORD_API_TOKEN
CHANNEL_IDS = [662531201097007121]
SAVE_DIR = 'luh_crank_msg_history.json'

# Define intents
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content
intents.messages = True
intents.guilds = True

# Initialize bot
bot = commands.Bot(command_prefix='!', intents=intents)

# Event when the bot is ready
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    await fetch_messages()

async def fetch_messages():
    await bot.wait_until_ready()

    all_messages = []

    for channel_id in CHANNEL_IDS:
        try:
            channel = await bot.fetch_channel(channel_id)
        except discord.NotFound:
            print(f"Channel with ID {channel_id} not found.")
            continue
        except discord.Forbidden:
            print(f"No permissions to access channel {channel_id}.")
            continue
        except discord.HTTPException as e:
            print(f"HTTP exception occurred: {e}")
            continue

        print(f'Fetching messages from channel: {channel.name}')
        messages = []
        try:
            async for message in channel.history(limit=None, oldest_first=True):
                messages.append({
                    'author': str(message.author),
                    'content': message.content,
                    'timestamp': message.created_at.isoformat()
                })
            print(f'Fetched {len(messages)} messages from {channel.name}.')

            all_messages.extend(messages)

        except Exception as e:
            print(f'An error occurred: {e}')

    # Save messages to a JSON file
    with open(SAVE_DIR, 'w', encoding='utf-8') as f:
        json.dump(all_messages, f, ensure_ascii=False, indent=4)
    print('Message history saved to message_history.json')

    await bot.close()


bot.run(TOKEN)
