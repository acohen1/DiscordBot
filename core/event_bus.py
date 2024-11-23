from pydispatch import dispatcher
import logging

logger = logging.getLogger("EventBus")

# Define signal names
ON_READY = "on_ready"                               # Called when discord client logs in (initialization)
ON_REACTION_ADD = "on_reaction_add"                 # Called when a reaction is added: reaction, user
ON_PRESENCE_UPDATE = "on_presence_update"           # Called when a user's presence is updated: before, after

# Incoming message pipeline
ON_MESSAGE = "on_message"                           # Called when received message from Discord
AWAITING_RESPONSE = "awaiting_response"             # Called when msg is processed and we are awaiting a response (inits CoT pipeline)
RESPONSE_COMPLETE = "response_complete"             # Called when response is ready to be sent to user (string)



ON_RESPONSE_COMPLETED = "on_response_completed"     # Assistant response is ready: response



# Global Dispatcher for signals
def emit_event(signal, **kwargs):
    """Emit an event with a signal and associated data."""
    logger.info(f"Emitting event {signal}")
    dispatcher.send(signal=signal, **kwargs)
