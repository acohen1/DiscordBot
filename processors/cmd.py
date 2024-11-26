import threading
import logging
from clients.openai_client import OpenAIClient
import discord

logger = logging.getLogger("CommandProcessor")

class CommandProcessor:
    """Handles bot commands."""
    _instance = None
    _lock = threading.Lock()  # Lock object to ensure thread safety

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of CommandProcessor is created."""
        if cls._instance is None:
            with cls._lock:  # Lock this section to prevent race conditions
                if cls._instance is None:  # Double-check locking
                    cls._instance = super(CommandProcessor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the CommandProcessor. Prevent reinitialization."""
        if not hasattr(self, "initialized"):
            self.openai_client = OpenAIClient.get_instance()
            self.initialized = True

    async def process_commands(self, message: discord.Message, threads: dict) -> bool:
        """
        Detect and handle commands in the message.

        Args:
            message (discord.Message): The message to process.
            threads (dict): The cache of GLThreads.
        Returns:
            bool: True if the message contains a command, else False.
        """
        command, args = self._is_command(message)
        if command:
            logger.info(f"Detected command: {command} with args: {args}")
            # Map commands to their respective handlers
            command_handlers = {
                "lobotomy": self._handle_lobotomy,
                # Add new commands here
            }

            # Dispatch the command if it has a handler
            if command in command_handlers:
                await command_handlers[command](message, threads, *args)
                return True
            else:
                logger.warning(f"Unknown command: {command}")
                await message.channel.send("I don't recognize that command. <:confused:975696064478847016>")
                return False
                
    
    def _is_command(self, message: discord.Message):
        """
        Check if the message contains a command.

        Args:
            message (discord.Message): The message to check.

        Returns:
            Tuple[str, List[str]]: The command name and arguments if a command is detected, else (None, None).
        """
        bot_id = message.guild.me.id
        if message.content.startswith(f"<@{bot_id}>"):
            # Extract command and arguments
            command_body = message.content[len(f"<@{bot_id}>"):].strip()
            if command_body.startswith("/"):
                parts = command_body[1:].split()
                command = parts[0].lower()
                args = parts[1:]  # Remaining parts as arguments
                return command, args
        return None, None

    async def _handle_lobotomy(self, message: discord.Message, threads: dict, *args):
        """
        Handle the /lobotomy command to clear a user's conversation.

        Args:
            message (discord.Message): The message containing the command.
            threads (dict): The cache of GLThreads.
        """
        try:
            user_id = message.author.id
            if user_id not in threads:
                await message.channel.send("You don't have an active thread to lobotomize.")
                return

            logger.info(f"Received /lobotomy command from {message.author.name} in {message.channel.name}.")

            # Check for --all argument flag; clear conversations for all users
            if args and args[0] == "--all":
                for user_id in threads.keys():
                    threads[user_id].clear_conversation()
                logger.info("Cleared all threads.")
                await message.channel.send("**ALL** gone <:brainlet:1300560937778155540>")
                return


            # Clear the conversation
            is_cleared = threads[user_id].clear_conversation()
            if is_cleared:
                logger.info(f"Cleared thread for {message.author.name}.")
                await message.channel.send("All gone <:brainlet:1300560937778155540>")
            else:
                await message.channel.send("I can't do that right now. <:GreggLimper:975696064478847016>")
        except Exception as e:
            logger.error(f"Error handling /lobotomy command for {message.author.name}: {e}")
            await message.channel.send("Something went wrong while performing the lobotomy.")
