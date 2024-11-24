from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Optional, Iterator, List
import logging

logger = logging.getLogger('GLThread')

class GLMessage:
    """GLMessage represents a message in a conversation."""
    def __init__(
        self,
        role: str,  # 'user' or 'assistant'
        content: str,
        timestamp: datetime,
        message_id: int,
        target_message_id: Optional[int] = None
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.message_id = message_id
        self.target_message_id = target_message_id

    def __str__(self):
        return f"{self.timestamp} - {self.message_id} - {self.role if self.role=="user" else "asst"} - {self.content[:250].strip()}"
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp
    
    def __eq__(self, other):
        return self.message_id == other.message_id

    def set_role(self, role: str):
        if role not in ['user', 'assistant']:
            raise ValueError("role must be 'user' or 'assistant'")
        self.role = role

    def set_content(self, content: str):
        self.content = content

    def set_timestamp(self, timestamp: datetime):
        # Ensure timestamp is in UTC
        if timestamp.tzinfo is not None:
            raise ValueError("timestamp must be in UTC")
        self.timestamp = timestamp

    def set_target_message_id(self, target_message_id: int):
        if target_message_id == self.message_id:
            raise ValueError("target_message_id must be different from message_id")
        self.target_message_id = target_message_id

class GLConversation:
    def __init__(
        self,
        message_history: Optional[Deque[GLMessage]] = None,
        max_history_length: Optional[int] = None
    ):
        """GLConversation represents a conversation between a user and an assistant."""
        self.message_history: Deque[GLMessage] = message_history or deque(maxlen=max_history_length)

    def add_message(self, message: GLMessage) -> bool:
        try:
            self.message_history.append(message)
            logger.debug(f"Message {message.message_id} added to conversation.")
            return True
        except Exception as e:
            logger.error(f"Failed to add message {message.message_id}: {e}")
            return False

    def get_messages(self) -> List[GLMessage]:
        return list(self.message_history)
    
    def delete_message_by_id(self, message_id: int) -> bool:
        found = any(msg.message_id == message_id for msg in self.message_history)
        if not found:
            logger.debug(f"Message {message_id} not found in conversation.")
            return False
        self.message_history = deque(
            (msg for msg in self.message_history if msg.message_id != message_id),
            maxlen=self.message_history.maxlen
        )
        logger.debug(f"Deleted message {message_id} from conversation.")
        return True

    def delete_message_by_timestamp(self, timestamp: datetime, tolerance_ms: int = 100) -> bool:
        tol_ms = timedelta(milliseconds=tolerance_ms)
        found = any(abs(msg.timestamp - timestamp) <= tol_ms for msg in self.message_history)
        if not found:
            logger.debug(f"No message with timestamp {timestamp} within tolerance found.")
            return False
        self.message_history = deque(
            (msg for msg in self.message_history if abs(msg.timestamp - timestamp) > tol_ms),
            maxlen=self.message_history.maxlen
        )
        logger.debug(f"Deleted message with timestamp {timestamp} from conversation.")
        return True

    def delete_all_messages(self) -> bool:
        if not self.message_history:
            logger.debug("Conversation is already empty.")
            return False
        self.message_history.clear()
        logger.debug("All messages deleted from conversation.")
        return True

    def __str__(self) -> str:
        return '\n'.join(str(msg) for msg in self.message_history)

    def __iter__(self) -> Iterator[GLMessage]:
        return iter(self.message_history)

class GLThread:
    def __init__(
        self,
        discord_user_id: int,
        conversation: Optional[GLConversation] = None,
        max_history_length: int = 100,
    ):
        """GLThread holds a conversation thread between a user and an assistant, along with metadata about the thread and the conversation itself.
        Args:
            discord_channel_id (int): The Discord channel ID where the conversation is taking place.
            discord_user_id (int): The Discord user ID of the user participating in the conversation.
            assistant_thread_id (str): The unique ID of the thread.
            conversation (GLConversation, optional): The conversation object containing the message history. Defaults to None.
            max_history_length (int, optional): The maximum number of messages to store in the conversation history. Defaults to 100.
        """
        self.discord_user_id = discord_user_id
        self.conversation = conversation or GLConversation(max_history_length=max_history_length)

    def add_message(self, message: GLMessage) -> bool:
        """Add a message to the conversation."""
        try:
            result = self.conversation.add_message(message)
            if result:
                logger.debug(f"Added message {message.message_id} to conversation.")
            else:
                logger.error(f"Failed to add message {message.message_id} to conversation.")
            return result
        except Exception as e:
            logger.error(f"Exception when adding message {message.message_id}: {e}")
            return False

    def delete_message_by_id(self, message_id: int) -> bool:
        """Delete a message in the conversation by its unique ID."""
        result = self.conversation.delete_message_by_id(message_id)
        if result:
            logger.debug(f"Deleted message {message_id} from conversation.")
        else:
            logger.debug(f"Message {message_id} not found in conversation.")
        return result

    def delete_message_by_timestamp(self, timestamp: datetime, tolerance_ms: int = 100) -> bool:
        """Delete a message in the conversation by its timestamp, with a given tolerance."""
        result = self.conversation.delete_message_by_timestamp(timestamp, tolerance_ms)
        if result:
            logger.debug(f"Deleted message with timestamp {timestamp} from conversation.")
        else:
            logger.debug(f"No message with timestamp {timestamp} within tolerance found in conversation.")
        return result

    def clear_conversation(self) -> bool:
        """Clear all messages in the conversation."""
        result = self.conversation.delete_all_messages()
        if result:
            logger.debug("Cleared all messages in the conversation.")
        else:
            logger.debug("Conversation was already empty.")
        return result

    def get_conversation_messages(self) -> List[GLMessage]:
        """Get all messages in the conversation."""
        return self.conversation.get_messages()

    def __str__(self) -> str:
        """String representation of the thread."""
        return (
            f"Discord User ID: {self.discord_user_id}\n"
            f"Conversation:\n{self.conversation}"
        )

    def __iter__(self) -> Iterator['GLMessage']:
        """Make the thread iterable over its conversation messages."""
        return iter(self.conversation)
    
class CombinedGLThread:
    def __init__(self, threads: list[GLThread]):
        """A CombinedGLThread temporarily merges multiple GLThreads for context.
        Args:
            threads (list[GLThread]): A list of GLThreads to combine.
        """
        if not threads:
            raise ValueError("At least one GLThread must be provided.")

        # Store threads by user ID
        self.threads_by_user = {
            thread.discord_user_id: thread for thread in threads
        }

        # Create a combined conversation
        self.conversation = GLConversation(
            max_history_length=max(thread.conversation.message_history.maxlen for thread in threads)
        )

        # Combine messages from all threads
        all_messages = []
        for thread in threads:
            all_messages.extend(thread.get_conversation_messages())

        # Sort messages by timestamp
        all_messages.sort(key=lambda msg: msg.timestamp)

        # Add sorted messages to the combined conversation
        for message in all_messages:
            self.conversation.add_message(message)

    def get_combined_conversation_messages(self) -> List[GLMessage]:
        """
        Retrieve the messages in the combined conversation.
        Returns:
            List[GLMessage]: The list of GLMessages in the combined conversation.
        """
        return self.conversation.get_messages()

    def get_original_threads(self) -> list[GLThread]:
        """
        Retrieve the original threads that were merged.
        Returns:
            list[GLThread]: The list of original GLThreads.
        """
        return list(self.threads_by_user.values())

    def get_user_ids(self) -> list[int]:
        """
        Retrieve the user IDs involved in this combined thread.
        Returns:
            list[int]: List of user IDs in the combined thread.
        """
        return list(self.threads_by_user.keys())

    def __str__(self) -> str:
        """String representation of the combined thread."""
        user_threads = "\n".join(
            f"User ID: {user_id}, Assistant Thread ID: {thread.assistant_thread_id}"
            for user_id, thread in self.threads_by_user.items()
        )
        return (
            f"Combined Thread:\n"
            f"{user_threads}\n"
            f"Conversation:\n{self.conversation}"
        )