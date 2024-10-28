import json
import os
import time
from datetime import datetime, timedelta
from better_profanity import profanity
import openai
from privtoken import OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

MESSAGE_HISTORY_FILE = 'msg/luh_crank_msg_history.json'
FINE_TUNE_DATA_FILE = 'msg/fine_tune_data.jsonl'
FLAGGED_MESSAGES_FILE = 'msg/flagged_messages.jsonl'
ASSISTANT_USERNAMES = ['grumbo1', 'toe', 'mahik']
EXCLUDED_USERNAMES = ['yungdopesniffer', 'panpanpanpanpanpan', 'tijmen957', 'yerr9937']
CONVERSATION_GAP_MINUTES = 60
CUSTOM_PROFANITY_LIST = [""]

# Create directories if they don't exist
os.makedirs(os.path.dirname(FINE_TUNE_DATA_FILE), exist_ok=True)
os.makedirs(os.path.dirname(FLAGGED_MESSAGES_FILE), exist_ok=True)

# Initialize the profanity filter with default and custom words
profanity.load_censor_words()
profanity.add_censor_words(CUSTOM_PROFANITY_LIST)

# Load message history
with open(MESSAGE_HISTORY_FILE, 'r', encoding='utf-8') as f:
    messages = json.load(f)

BATCH_SIZE = 32
processed_count = 0
total_messages = len(messages)
filtered_messages = []
flagged_messages = []

# Function to process messages in batches for moderation with retry logic
def batch_moderate_content(batch, max_retries=7):
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.moderations.create(
                model="text-moderation-latest",
                input=batch
            )
            response_dict = response.to_dict()
            return response_dict["results"]
        except Exception as e:
            print(f"\nModeration API error: {e}")
            retry_count += 1
            print(f"Retrying... Attempt {retry_count}/{max_retries}")
            time.sleep(2 ** retry_count)  # Exponential backoff
    return [None] * len(batch)

# Process messages in batches for moderation
batch = []
for msg in messages:
    author = msg.get('author')
    content = msg.get('content', '').strip()
    timestamp = msg.get('timestamp')  # Keep the timestamp
    
    # Skip if message is missing critical info
    if not author or not content or not timestamp:
        continue

    # Filter out messages by excluded usernames or containing profanity
    if author in EXCLUDED_USERNAMES or profanity.contains_profanity(content):
        continue

    role = 'assistant' if author in ASSISTANT_USERNAMES else 'user'
    
    # Prepare the batch with stripped content, role, and timestamp
    batch.append({"content": content, "role": role, "timestamp": timestamp})
    
    if len(batch) >= BATCH_SIZE or (processed_count + len(batch)) >= total_messages:
        # Extract only the content for moderation
        moderation_input = [item["content"] for item in batch]
        results = batch_moderate_content(moderation_input)
        
        # Process each result
        for i, result in enumerate(results):
            if result:
                if not result['flagged']:
                    filtered_messages.append(batch[i])  # Append the full message (with content, role, and timestamp)
                else:
                    flagged_messages.append(batch[i])
            
        processed_count += len(batch)
        print(f"\rProcessed {processed_count}/{total_messages} messages...", end='', flush=True)
        
        batch = []  # Reset batch

# Ensure all filtered messages have the required keys before proceeding
for msg in filtered_messages:
    if 'role' not in msg:
        msg['role'] = 'user' if msg['author'] not in ASSISTANT_USERNAMES else 'assistant'

# Prepare data for fine-tuning in ChatGPT format
fine_tune_data = []
i = 0
while i < len(filtered_messages):
    msg = filtered_messages[i]
    if msg.get('role') == 'user':  # Only start with user-initiated conversations
        conversation = []
        assistant_in_convo = False
        last_assistant_index = None
        start_time = datetime.fromisoformat(msg['timestamp'].rstrip('Z'))
        end_time = start_time + timedelta(minutes=CONVERSATION_GAP_MINUTES)
        conversation.append(msg)
        i += 1

        while i < len(filtered_messages):
            next_msg = filtered_messages[i]
            next_time = datetime.fromisoformat(next_msg['timestamp'].rstrip('Z'))
            if next_time <= end_time:
                conversation.append(next_msg)
                if next_msg.get('role') == 'assistant':
                    assistant_in_convo = True
                    last_assistant_index = len(conversation) - 1
                i += 1
            else:
                break

        if assistant_in_convo and last_assistant_index is not None:
            conversation = conversation[:last_assistant_index + 1]
            formatted_conversation = [
                {"role": conv_msg['role'], "content": conv_msg['content'].strip()}
                for conv_msg in conversation if conv_msg['content'].strip()
            ]
            fine_tune_data.append({"messages": formatted_conversation})
    else:
        i += 1

# Save the fine-tuning data to a JSON Lines file in the required format
os.makedirs(os.path.dirname(FINE_TUNE_DATA_FILE), exist_ok=True)
with open(FINE_TUNE_DATA_FILE, 'w', encoding='utf-8') as f:
    for item in fine_tune_data:
        json.dump(item, f)
        f.write('\n')
