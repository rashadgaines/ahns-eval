"""OpenAI model implementation for the evaluation framework."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.completion import Completion
from tiktoken import encoding_for_model

from eval_framework.models.base import BaseModel
from eval_framework.core.registry import register_model

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for OpenAI API requests."""
    
    def __init__(self, requests_per_minute: int):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit token."""
        async with self.lock:
            now = datetime.now()
            # Remove requests older than 1 minute
            self.requests = [req for req in self.requests if now - req < timedelta(minutes=1)]
            
            if len(self.requests) >= self.requests_per_minute:
                # Wait until oldest request expires
                wait_time = (self.requests[0] + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            self.requests.append(now)

class OpenAIModel(BaseModel[str, str]):
    """OpenAI model implementation.
    
    This class provides an interface to OpenAI's language models with support for:
    - Different model types (GPT-3.5, GPT-4, etc.)
    - Rate limiting
    - Token counting
    - Error handling and retries
    - Async operation
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        requests_per_minute: int = 60,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ):
        """Initialize OpenAI model.
        
        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: OpenAI organization ID (defaults to OPENAI_ORGANIZATION env var)
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            requests_per_minute: Maximum requests per minute for rate limiting
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        # Initialize tokenizer
        try:
            self.tokenizer = encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for newer models
            self.tokenizer = encoding_for_model("cl100k_base")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.model_name,
            "type": "openai",
            "parameters": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **self.kwargs
            }
        }
    
    def validate_input(self, input_data: str) -> Optional[str]:
        """Validate input data.
        
        Args:
            input_data: Input text to validate
            
        Returns:
            None if valid, error message if invalid
        """
        if not isinstance(input_data, str):
            return "Input must be a string"
        if not input_data.strip():
            return "Input cannot be empty"
        return None
    
    def count_tokens(self, text: str) -> int:
        """Count number of tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    async def generate(self, input_data: str, **kwargs) -> str:
        """Generate text from input.
        
        Args:
            input_data: Input text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            ValueError: If input is invalid
            openai.APIError: If API request fails
        """
        # Validate input
        if error := self.validate_input(input_data):
            raise ValueError(error)
        
        # Merge parameters
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.kwargs,
            **kwargs
        }
        
        # Add messages for chat models
        if self.model_name.startswith(("gpt-3.5", "gpt-4")):
            params["messages"] = [{"role": "user", "content": input_data}]
        
        # Retry loop
        for attempt in range(self.max_retries):
            try:
                # Acquire rate limit token
                await self.rate_limiter.acquire()
                
                # Make API request
                if self.model_name.startswith(("gpt-3.5", "gpt-4")):
                    response = await self.client.chat.completions.create(**params)
                    return response.choices[0].message.content
                else:
                    response = await self.client.completions.create(
                        prompt=input_data,
                        **params
                    )
                    return response.choices[0].text
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(
                    f"Request attempt {attempt + 1} failed: {str(e)}. "
                    f"Retrying in {self.retry_delay} seconds..."
                )
                await asyncio.sleep(self.retry_delay)
    
    async def batch_generate(self, batch_inputs: List[str], **kwargs) -> List[str]:
        """Generate text for a batch of inputs.
        
        Args:
            batch_inputs: List of input texts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
            
        Raises:
            ValueError: If any input is invalid
            openai.APIError: If API request fails
        """
        # Validate inputs
        for input_data in batch_inputs:
            if error := self.validate_input(input_data):
                raise ValueError(f"Invalid input: {error}")
        
        # Process inputs concurrently with rate limiting
        async def process_input(input_data: str) -> str:
            return await self.generate(input_data, **kwargs)
        
        tasks = [process_input(input_data) for input_data in batch_inputs]
        return await asyncio.gather(*tasks)

# Register the model
register_model("openai")(OpenAIModel) 