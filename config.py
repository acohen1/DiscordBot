from privtoken import DISCORD_API_TOKEN, OPENAI_API_KEY, GOOGLE_API_KEY, GIPHY_API_KEY
from sys_prompt import MAIN_SYS_PROMPT, RULES, GREGG_LIMPER_ATTRIBUTES

CONFIG = {
    "FINE_TUNED_MODEL": "ft:gpt-4o-2024-08-06:personal:gregg-limper:AN9TcxoD",              # The OpenAI model ID for the fine-tuned model used for conversation.
    "IMAGE_DETECTION_MODEL": "ft:gpt-4o-2024-08-06:personal:gregg-limper:AN9TcxoD",         # The OpenAI model ID for the fine-tuned model used for image detection.
    "SEARCH_ENGINE_ID": "317eaf1b03dbe42dc",                                                # The Google Custom Search Engine ID for searching the web.
    "ALLOWED_CHANNEL_IDS": [1299973475959705631, 662531201097007121],                       # The list of channel IDs where the bot is allowed to operate.
    "TRAINING_DATA_DIR": "new_training_data",                                               # The directory to store the new training data file.
    "CACHE_INIT_TIME_LIMIT_MINUTES": 180,                                                   # The time limit for messages to be considered in the conversation history.
    "CACHED_HISTORY_LENGTH": 30,                                                            # The maximum number of messages to store in the recent messages cache
    "ASSISTANT_CONTEXT_LENGTH": 10,                                                         # The number of messages to send to the assistant for context.  
    "REACTION_HISTORY_LENGTH": 10,                                                          # The number of messages to fetch for training data when a reaction is added.

    "OPENAI_API_KEY": OPENAI_API_KEY,                                                       # The OpenAI API key.
    "DISCORD_API_TOKEN": DISCORD_API_TOKEN,                                                 # The Discord API token.
    "GOOGLE_API_KEY": GOOGLE_API_KEY,                                                       # The Google API key.
    "GIPHY_API_KEY": GIPHY_API_KEY,                                                         # The Giphy API key.

    "GREGG_LIMPER_ATTRIBUTES": GREGG_LIMPER_ATTRIBUTES,                                     # The characteristics of Gregg Limper.
    "SYSTEM_PROMPT": f"""               
    You are Gregg Limper.
    {GREGG_LIMPER_ATTRIBUTES}

    {MAIN_SYS_PROMPT}

    {RULES}

    **REMEMBER:** 
    - **Text Only**: Reply to the latest message with a plain text response; bring in other relevant context if needed.
    - **No Media Formatting**: Don’t include links or media content formats in your responses.  
    - **Engage in Gregg’s Unique Voice**: Stay in character, push boundaries, and challenge users with responses that bring Gregg Limper’s personality to life.
    """
}
