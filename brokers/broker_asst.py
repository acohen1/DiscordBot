import threading
from core.event_bus import RESPONSE_COMPLETE, emit_event
import logging
import openai

logger = logging.getLogger("AsstBroker")

class AssistantStreamHandler(openai.AsyncAssistantEventHandler):
    def __init__(self):
        super().__init__()

    async def on_event(self, event):
        if event.event == "thread.message.completed":
            message = event.data.content[0].text.value.strip()
            logger.info(f"Received message from assistant: {message}")
            emit_event(signal=RESPONSE_COMPLETE, message=message)