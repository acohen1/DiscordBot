import os
from dotenv import load_dotenv
import logging

load_dotenv()

DISCORD_API_TOKEN = os.getenv("DISCORD_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")

ASSISTANT_ID = os.getenv("ASSISTANT_ID")

COT_MODEL_ID = os.getenv("COT_MODEL_ID")
COT_MODEL_TEMP = 0.2
COT_MAX_ATTEMPTS = 3

MSG_MODEL_ID = os.getenv("MSG_MODEL_ID")
MSG_MODEL_TEMP = 0.8

IMG_MODEL_ID = os.getenv("IMG_MODEL_ID")
IMG_MODEL_TEMP = 0.5

CACHE_CONVERSATIONS_LEN = 100
CACHE_CONVERSATIONS_TIMELIMIT_MINS = 360
CACHE_MESSAGE_LEN = 1000

MAX_SEARCH_RESULTS = 5

if not DISCORD_API_TOKEN:
    raise ValueError("DISCORD_API_TOKEN is not set in .env")
if not OPENAI_API_KEY:
    raise ValueError("OPEN_AI_API_KEY is not set in .env")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env")
if not GIPHY_API_KEY:
    raise ValueError("GIPHY_API_KEY is not set in .env")
if not ASSISTANT_ID:
    raise ValueError("ASSISTANT_ID is not set in .env")

# Centralized logging configuration
def setup_logging(level=logging.INFO):
    """
    Configure logging for the entire application.
    Args:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )

    # Suppress DEBUG logs from external libraries
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("discord").setLevel(logging.WARNING)