from flask import Flask, render_template, jsonify
from discord_bot import GreggLimperBot
from datetime import datetime
import pytz
import threading
import logging

app = Flask(__name__)
gregg_limper_bot = GreggLimperBot()

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.WARNING)

# Custom filter to convert timestamps to EST
@app.template_filter('to_est_time')
def to_est_time(timestamp):
    utc_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))  # Parse ISO with UTC
    est = pytz.timezone("America/New_York")
    est_dt = utc_dt.astimezone(est)
    return est_dt.strftime("%Y-%m-%d %I:%M %p EST")  # Format time

@app.route('/')
def index():
    """Render the index.html template with the conversation history from the bot's cache."""
    conversations = gregg_limper_bot.show_cache(show_names=True)  # Get cache data with channel names
    return render_template('index.html', conversations=conversations)

@app.route('/show_cache', methods=['GET'])
def show_cache():
    """Endpoint to view the bot's message cache in JSON format."""
    return jsonify(gregg_limper_bot.show_cache(show_names=True))  # Return JSON of the cache with channel names

def start_discord_bot():
    """Run the Discord bot directly without an event loop wrapper."""
    gregg_limper_bot.run()

if __name__ == '__main__':
    # Start the bot in a separate thread
    bot_thread = threading.Thread(target=start_discord_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Start the Flask app
    app.run()
