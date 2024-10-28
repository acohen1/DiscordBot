from flask import Flask, render_template
import json
import os

HISTORY_DIR = "active_conversations" # Directory containing JSON files of conversation histories

app = Flask(__name__)

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
                        channel_name = data.get("channel_name", filename)             # Default to filename if channel name is missing
                        conversations[channel_name] = data.get("messages", [])[-10:]  # Load only the last 10 messages
                    except json.JSONDecodeError:
                        conversations[filename] = [{"error": "Unable to decode JSON content."}]
    return conversations

@app.route("/")
def view_history():
    conversations = load_all_conversation_histories()
    return render_template("index.html", conversations=conversations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
