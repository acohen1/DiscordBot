import openai
import threading
from core.config import OPENAI_API_KEY, SYS_PROMPT, COT_MODEL_ID, MSG_MODEL_ID, IMG_MODEL_ID, COT_MODEL_TEMP, MSG_MODEL_TEMP, IMG_MODEL_TEMP
from models.threads import GLMessage, GLThread
import logging
from typing import Optional, List, Dict
import asyncio

logger = logging.getLogger('AsyncOpenAI')

class OpenAIClient:
    _instance = None
    _lock = threading.Lock()  # Lock object to ensure thread safety

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of OpenAIClient is created, even in a multithreaded context."""
        if cls._instance is None:
            with cls._lock:  # Lock this section to prevent race conditions
                if cls._instance is None:  # Double-check locking
                    cls._instance = super(OpenAIClient, cls).__new__(cls)
                    cls._instance._initialized = False  # Set initialization flag
        return cls._instance

    async def async_init(self):
        """Asynchronous initialization for the OpenAIClient."""
        if not self._initialized:  # Check if initialization is needed
            async with asyncio.Lock():  # Ensure thread-safe async initialization
                if not self._initialized:  # Double-check inside the lock
                    self.client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

                    # Set model attributes
                    self.chain_of_thought_model_id = COT_MODEL_ID
                    self.chain_of_thought_temp = COT_MODEL_TEMP

                    self.message_model_id = MSG_MODEL_ID
                    self.message_model_temp = MSG_MODEL_TEMP

                    self.image_model_id = IMG_MODEL_ID
                    self.image_model_temp = IMG_MODEL_TEMP

                    self._initialized = True  # Mark instance as initialized

    @classmethod
    async def create(cls):
        """Factory method to asynchronously initialize the singleton."""
        instance = cls()  # Calls __new__, ensuring only one instance exists
        await instance.async_init()  # Perform async initialization
        return instance

    @classmethod
    def get_instance(cls):
        """Non-async method to get the singleton instance."""
        instance = cls()
        if not instance._initialized:
            raise RuntimeError(
                "OpenAIClient must be initialized asynchronously using `await OpenAIClient.create()` before accessing it."
            )
        return instance
        
    async def image_describer(self, base64_str: str) -> str:
        """Given a base64 encoded image, request a description from OpenAI."""
        try:                    
            # Prepare and send the request to OpenAI for image analysis
            system_prompt = (
                "Your purpose is to provide a description of the image content embeded in the message.\n\n"
                "Provide a succinct description useful for someone who can't see it. "
                "Include any relevant text or context in the image, but try to keep it concise."
            )
            user_prompt = f"What is in this image? Provide a succinct description useful for someone who can't see it."

            response = await self.client.chat.completions.create(
                model=self.image_model_id,
                messages=[
                    { "role" : "system", "content" : system_prompt },
                    { 
                        "role" : "user", 
                        "content" : [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=self.image_model_temp
            )
            
            # Retrieve and return the result from OpenAI
            result = response.choices[0].message.content if response.choices else "No description available"
            logger.debug(f"Image description: {result}")
            return result
        except Exception as e:
            logger.error(f"Error processing image content: {str(e)}")
            return "No description available"

    async def text_summarizer(self, description: str) -> str:
        try:
            system_prompt = (
                "Your purpose is to provide a concise, succint summary of text descriptions."
            )

            user_prompt = (
                f"Create a concise, succint, one-to-two-sentence summary for the following description:\n\n"
                f"{description}\n\n"
                "Summary:"
            )
            response = await self.client.chat.completions.create(
                model=self.message_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,
                temperature=self.message_model_temp
            )
            summary = response.choices[0].message.content.strip() if response.choices else "No summary available"
            return summary
        except Exception as e:
            logger.error(f"Error summarizing description: {str(e)}")
            return "No summary available"
        
    async def link_summarizer(self, url: str) -> str:
        try:
            system_prompt = (
                "Your purpose is to describe the content of a webpage based on its URL.\n\n"
                "Extract any details you can from the names, titles, and descriptions in the URL.\n\n"
                "Provide a concise, succint summary of the content that would be useful for someone who can't access the page."
            )

            user_prompt = (
                f"Please describe the content of the webpage at the following URL: {url}\n\n"
                "Description:"
            )
            response = await self.client.chat.completions.create(
                model=self.message_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,
                temperature=self.message_model_temp
            )
            summary = response.choices[0].message.content.strip() if response.choices else "No summary available"
            return summary
        except Exception as e:
            logger.error(f"Error summarizing description: {str(e)}")
            return "No summary available"
        
    async def determine_content_type(self, OAI_messages: List[Dict]) -> Optional[str]:
        """Given a list of OpenAI messages, determine the content type the assistant should respond with."""
        system_prompt = (
            "Based on the most recent message, reply with one word that best describes the type of response that would be most relevant and helpful: 'message', 'GIF', 'YouTube', or 'Website'.\n\n"
            "Rules:\n"
            "1. If the user explicitly requests a GIF (e.g., 'send me a GIF', 'respond with a GIF', or 'can you find a funny GIF about this'), always respond with 'GIF'.\n"
            "2. If the context suggests a reaction (e.g., something funny, shocking, or emotional), or if a GIF would add a playful or expressive touch to the conversation, respond with 'GIF'.\n"
            "3. If the user explicitly mentions or responds to a Website, YouTube, or GIF, reply with 'message' to keep the conversation going.\n"
            "4. If the user asks about current trends, popular culture, or something that benefits from links to explore further (like memes, articles, or viral topics), respond with 'Website'.\n"
            "5. If the user mentions tutorials, music videos, or anything best explained or shared visually through video, respond with 'YouTube'.\n"
            "6. For personal stories, advice, explanations, or recommendations, respond with 'message'.\n"
            "6. For casual back-and-forth conversations, questions, or general replies that don’t clearly fit another type, respond with 'message'.\n\n"
            "Important:\n"
            "- Do not provide any additional text or explanations.\n"
            "- **ONLY REPLY WITH ONE OF THE FOLLOWING WORDS:** message, GIF, YouTube, or Website."
        )

        affix_prompt = (
            "Now determine the content type of your response: message, GIF, YouTube, or Website."
        )
        # Prefix the messages with the system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            *OAI_messages,
            {"role": "user", "content": affix_prompt}
        ]

        # Send the messages to OpenAI for processing
        try:
            response = await self.client.chat.completions.create(
                model=self.chain_of_thought_model_id,
                messages=messages,
                max_tokens=10,
                temperature=self.chain_of_thought_temp
            )
            content_type = response.choices[0].message.content.strip().lower()

            if content_type in ["message", "gif", "youtube", "website"]:
                return content_type
            else:
                logger.error(f"Invalid content type '{content_type}")
                return None
        
        except Exception as e:
            logger.error(f"Error determining content type: {e}")
            return None

    async def generate_message_response(self, OAI_messages: List[Dict]) -> Optional[str]:
        prefix_prompt = SYS_PROMPT

        # affix_prompt = (
        #     "Now generate a response based on the most recent messages."
        # )

        messages = [
            {"role": "system", "content": prefix_prompt},
            *OAI_messages,
        ]

        # Send to OpenAI for a response
        try:
            response = await self.client.chat.completions.create(
                model=self.message_model_id,
                messages=messages,
                max_tokens=300,
                temperature=self.message_model_temp
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error processing message response: {e}")
            return None

    async def generate_search_query(self, content_type: str, OAI_messages: List[Dict]) -> Optional[str]:
        """Given a content type, request a search query from OpenAI based on the provided messages.
        Args:
            content_type (str): The content type to search for: 'message', 'gif', 'youtube', or 'website'.
            OAI_messages (List[Dict]): The list of messages to provide to OpenAI.
        Returns:
            Optional[str]: The search query response from OpenAI or None if an error occurs.
        """
        if content_type == "gif":
            search_type = "GIF"
        elif content_type == "youtube":
            search_type = "YouTube video"
        elif content_type == "website":
            search_type = "website"

        prefix_prompt = (
            "Your purpose is to generate a search query based on the most recent messages.\n"
            "Use the context of the conversation to determine the most relevant search query.\n"
            "Limit the query length to ensure clarity and relevance in a response.\n"
            "Do not directly copy the user's message, but use it to generate a relevant search query.\n"
            f"You are searching for a {search_type} based on the most recent messages."
            "Reply only with the search query, do not include any additional text, links, media, or explanations.\n"
        )

        affix_prompt = (
            f"Now generate a search query for a {search_type} based on the most recent messages."
        )

        messages = [
            {"role": "system", "content" : prefix_prompt},
            *OAI_messages,
            {"role": "user", "content": affix_prompt}
        ]

        # Send to OpenAI for a response
        try:
            response = await self.client.chat.completions.create(
                model=self.chain_of_thought_model_id,
                messages=messages,
                max_tokens=50,
                temperature=self.message_model_temp
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error processing search query: {e}")
            return None

    async def is_followup_required(self, OAI_messages: List[Dict]) -> bool:
        """Given a list of OpenAI messages, determine if a follow-up response is required."""
        prefix_prompt = (
            "Your purpose is to determine if a follow-up response is required based on the most recent messages.\n"
            "Use the context of the conversation to determine if a follow-up is necessary.\n"
            "Does the assistant's response require a follow-up message from the user?\n"
            "Reply only with 'yes' or 'no' to indicate if a follow-up response is required."
        )

        affix_prompt = (
            "Now determine if a follow-up response is required based on the most recent messages.\n"
            "Only reply with 'yes' or 'no'."
        )

        messages = [
            {"role": "system", "content": prefix_prompt},
            *OAI_messages,
            {"role": "user", "content": affix_prompt}
        ]

        # Send to OpenAI for a response
        try:
            response = await self.client.chat.completions.create(
                model=self.chain_of_thought_model_id,
                messages=messages,
                max_tokens=10,
                temperature=self.chain_of_thought_temp
            )
            content = response.choices[0].message.content.strip().lower()
            if content in ["yes", "no"]:
                return content == "yes"
            else:
                logger.error(f"Invalid response content: {content}. Defaulting to 'no'.")
                return False
        except Exception as e:
            logger.error(f"Error determining follow-up requirement: {e}")
            return False


    # async def select_most_relevant_media(self, query: str, media_descriptions: List[str], OAI_messages: List[Dict]) -> int:
    #     """Given a search query and a list of media descriptions, select the index of the most relevant media description based on the recent conversation.
    #     Args:
    #         query (str): The search query used to find the media.
    #         media_descriptions (List[str]): The list of media descriptions to choose from.
    #         OAI_messages (List[Dict]): The list of messages to provide to OpenAI
    #     Returns:
    #         int: The index of the most relevant media description, or 0 if an error occurs.
    #     """
    #     prefix_prompt = (
    #         "Your purpose is to select the most relevant media from a list of descriptions.\n"
    #         "Use the provided search query and the context of the conversation to determine the most relevant media.\n"
    #         "Reply only with the number corresponding to the index of the selected media description.\n"
    #         "Do not include any additional text in your response."
    #     )

    #     affix_prompt = (
    #         "Now select the most relevant media from the list of descriptions.\n"
    #         f"The query is: {query}\n"
    #         f"The descriptions are:"
    #     )
    #     for i, description in enumerate(media_descriptions):
    #         affix_prompt += f"\n{i+1}. {description}"

    #     messages = [
    #         {"role": "system", "content": prefix_prompt},
    #         *OAI_messages,
    #         {"role": "user", "content": affix_prompt},
    #     ]

    #     # Send to OpenAI for a response
    #     try:
    #         response = await self.client.chat.completions.create(
    #             model=self.chain_of_thought_model_id,
    #             messages=messages,
    #             max_tokens=10,
    #             temperature=self.chain_of_thought_temp
    #         )
    #         content = response.choices[0].message.content.strip()

    #         # Validate and convert to index
    #         if content.isdigit():
    #             index = int(content)
    #             if 1 <= index <= len(media_descriptions):  # Ensure it's within the valid range
    #                 return index - 1
    #         logger.error(f"Invalid response content: {content}")
    #     except Exception as e:
    #         logger.error(f"Error selecting most relevant media: {e}")

    #     # Default fallback
    #     return 0