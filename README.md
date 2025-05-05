# GreggLimper — Discord LLM Assistant 🤖🧠

A modular, LLM-powered Discord assistant built with OpenAI, structured message handling, and chain-of-thought reasoning.

This bot routes Discord messages through a flexible processing pipeline—handling commands, media queries, structured tasks, and prompt-chained replies using GPT. Built for extensibility and clarity.

---

## 🚀 Features

- 🤖 **LLM-powered replies** via OpenAI GPT
- 🧠 **Chain-of-thought pipelines** (`services/cot.py`)
- 🧩 **Plugin-style processors** for commands, YouTube, web queries, and image handling
- 🗃️ **Global cache** for dialogue state, context, and usage
- 🛠️ Central **event bus** for clean async message handling

---

## 🧱 Project Structure

| Folder | Purpose |
|--------|---------|
| `clients/` | Discord + OpenAI clients |
| `core/` | Logging, config, cache, event bus |
| `services/` | Main pipeline logic (e.g. chain-of-thought, DTGL broker) |
| `processors/` | Modular handlers for commands, media, etc. |
| `models/` | Thread model definitions |
| `main.py` | Entry point — launches bot |
| `tests/` | Jupyter-style test notebook |
| `.env` | Environment variables (OpenAI keys, bot token) |

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/gregglimper-bot
cd gregglimper-bot

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

Then, create a `.env` file with:

```
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_key
```

---

## ▶️ Running the Bot

```bash
python main.py
```

---

## 🧪 Extending the Bot

Each handler in `processors/` is an independent module. To add new capabilities:

1. Add a new `.py` file in `processors/` (e.g. `weather.py`)
2. Register it in the message routing logic
3. Use the event bus to hook into messages, reactions, or presence updates

---

## 📄 TODO / Future Work

- [ ] Add unit tests for processors
- [ ] Add dynamic plugin discovery
- [ ] Enable memory / context persistence beyond cache layer
- [ ] Integrate slash commands for structured invocation

---

## 📜 License

MIT for code.  
Requires OpenAI API key and Discord bot token to run.
