"""
Interface module for interacting with Grok's image generation capabilities.
"""

import os
from typing import List, Optional, Tuple, Union

import requests
from PIL import Image
import io
import json
import time
from openai import OpenAI


class GrokImageGenerator:
    """Interface for Grok's image generation capabilities using OpenAI client."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.x.ai/v1",
        model: str = "grok-2-image-1212",
    ):
        """
        Initialize the Grok image generator interface.
        
        Args:
            api_key: Grok API key (defaults to GROK_API_KEY environment variable)
            base_url: Base URL for Grok API
            model: Name of the Grok model to use
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI API key must be provided or set as XAI_API_KEY environment variable")
        
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
    
    def generate_image(
        self,
        prompt: str,
        num_images: int = 1,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate images using Grok's image generation model via OpenAI client.
        """
        params = {
            "model": self.model,
            "prompt": prompt,
            "n": num_images,
            "response_format": "url",
        }
        params.update(kwargs)
        # Rate limiting: ensure no more than 5 requests per second
        time.sleep(0.2)  # 1/5 second delay
        response = self.client.images.generate(**params)
        images = []
        for data in response.data:
            url = data.url
            img_resp = requests.get(url)
            img_resp.raise_for_status()
            image = Image.open(io.BytesIO(img_resp.content))
            images.append(image)
        return images
    
    def generate_variations(
        self,
        image: Union[str, Image.Image],
        num_variations: int = 1,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate variations of an existing image.
        
        Args:
            image: Path to image file or PIL Image
            num_variations: Number of variations to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            List of generated PIL Images
        """
        endpoint = f"{self.base_url}/images/variations"
        
        # Convert image to bytes if needed
        if isinstance(image, str):
            with open(image, "rb") as f:
                image_bytes = f.read()
        else:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        
        # Prepare request
        files = {
            "image": ("image.png", image_bytes, "image/png")
        }
        data = {
            "n": num_variations,
            **kwargs
        }
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    data=data
                )
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                images = []
                
                for image_data in result["data"]:
                    # Convert base64 to image
                    image_bytes = image_data["b64_json"]
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)
                
                return images
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to generate variations after {self.max_retries} attempts: {str(e)}")
                time.sleep(self.retry_delay)
    
    def get_available_styles(self) -> List[str]:
        """
        Get list of available image generation styles.
        
        Returns:
            List of style names
        """
        endpoint = f"{self.base_url}/images/styles"
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers
            )
            response.raise_for_status()
            
            result = response.json()
            return result["styles"]
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get available styles: {str(e)}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the image generation model.
        
        Returns:
            Dictionary containing model information
        """
        endpoint = f"{self.base_url}/images/model"
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get model info: {str(e)}") 