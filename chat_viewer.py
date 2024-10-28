from flask import Flask, render_template
from datetime import datetime, timezone
import pytz
import json
import os

HISTORY_DIR = "active_conversations" # Directory containing JSON files of conversation histories

app = Flask(__name__)

def format_timestamp(timestamp):
    """Convert UTC timestamp to EST and format it nicely."""
    utc_dt = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)
    est_dt = utc_dt.astimezone(pytz.timezone("America/New_York"))
    return est_dt.strftime("%b %d, %Y - %I:%M %p")

def load_all_conversation_histories():
    """Load all conversation histories from JSON files in the active_conversations directory."""
    conversations = {}
    if os.path.exists(HISTORY_DIR):
        for filename in os.listdir(HISTORY_DIR):
            file_path = os.path.join(HISTORY_DIR, filename)
            if os.path.isfile(file_path):
                with open(file_path, "r") as f:
                    try:
                        data = json.load(f)
                        # Extract channel name and format timestamps
                        channel_name = data.get("channel_name", filename)
                        messages = data.get("messages", [])
                        # Format timestamps and reverse order
                        formatted_messages = [
                            {**msg, "timestamp": format_timestamp(msg["timestamp"])}
                            for msg in messages[:-10]
                        ]
                        conversations[channel_name] = formatted_messages
                    except json.JSONDecodeError:
                        conversations[filename] = [{"error": "Unable to decode JSON content."}]
    return conversations

@app.route("/")
def view_history():
    conversations = load_all_conversation_histories()
    return render_template("index.html", conversations=conversations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
