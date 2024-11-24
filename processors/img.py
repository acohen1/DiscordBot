from core.config import IMG_MODEL_ID
import threading
from clients.openai_client import OpenAIClient
import discord
import base64
import aiofiles
import tempfile
import os
import aiohttp
import logging
from PIL import Image

logger = logging.getLogger("ImageProcessor")

class ImageProcessor:
    _instance = None
    _lock = threading.Lock()    # Lock object to ensure thread safety

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of YouTubeProcessor is created, even in a multithreaded context."""
        if cls._instance is None:
            with cls._lock:    # Lock this section to prevent race conditions
                if cls._instance is None:   # Double-check inside the lock
                    cls._instance = super(ImageProcessor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the YouTubeProcessor. Prevent reinitialization by checking an attribute."""
        if not hasattr(self, "initialized"):
            # Other init methods....
            self.openai_client = OpenAIClient.get_instance()
            self.image_model_id = IMG_MODEL_ID
            self.initialized = True    # Mark this instance as initialized

    async def describe_image(self, image_url: str, is_gif=False) -> str:
        """Process an image from its URL and return a description of the content. If the URL is a GIF, extract the first frame and process.
        Args:
            image_url (str): The URL of the image to process.
            is_gif (bool): Whether the image is a GIF.
        Returns:
            str: The description of the image content.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "image.jpg") if not is_gif else os.path.join(temp_dir, "image.gif")
            
            # Download the image/GIF
            logger.debug(f"Downloading {'GIF' if is_gif else 'image'} from {image_url}")
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as response:
                        if response.status != 200:
                            raise Exception(f"Failed to download image {response.status}")
                        async with aiofiles.open(image_path, 'wb') as f:
                            await f.write(await response.read())
            except Exception as e:
                logger.error(f"Error downloading image: {str(e)}")
                return "No Description Available"
            
            # If we downloaded a GIF, extract the first frame
            if is_gif:
                logger.debug("Extracting first frame from GIF...")
                try:
                    with Image.open(image_path) as gif:
                        image_path = os.path.join(temp_dir, "image.jpg")
                        gif.seek(0)
                        gif.convert('RGB').save(image_path, "JPEG")
                except Exception as e:
                    logger.error(f"Error extracting first frame from GIF: {str(e)}")
                    return "No Description Available"
            
            # Encode the image to base64
            async with aiofiles.open(image_path, "rb") as image_file:
                base64_str = base64.b64encode(await image_file.read()).decode('utf-8')

                # Call Image Describer from OpenAI client and return result
                description =  await self.openai_client.image_describer(base64_str)
                return description if description else "No Description Available"



            