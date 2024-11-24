from pydispatch import dispatcher
import logging

logger = logging.getLogger("EventBus")

# Discord login events
ON_READY = "on_ready"                               # Called when discord client logs in (initialization)

# Discord reaction events
ON_REACTION_ADD = "on_reaction_add"                 # Called when a reaction is added: reaction, user

# Discord presence events
ON_PRESENCE_UPDATE = "on_presence_update"           # Called when a user's presence is updated: before, after

# Incoming message pipeline
ON_MESSAGE = "on_message"                           # Called when received message from Discord
AWAITING_RESPONSE = "awaiting_response"             # Called when incoming msg is processed and we are awaiting a response from the bot (inits CoT pipeline)
ON_RESPONSE_SENT = "on_response_completed"          # The bot finished a response stream and sent message(s) to the user


# Global Dispatcher for signals
def emit_event(signal, **kwargs):
    """Emit an event with a signal and associated data."""
    logger.info(f"Emitting event {signal}")
    dispatcher.send(signal=signal, **kwargs)
