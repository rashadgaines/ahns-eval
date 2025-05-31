"""Meta Llama model implementation for the evaluation framework."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from eval_framework.models.base import BaseModel
from eval_framework.core.registry import register_model

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for local model inference."""
    
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

class LlamaModel(BaseModel[str, str]):
    """Meta Llama model implementation.
    
    This class provides an interface to Meta's Llama models with support for:
    - Different model types (Llama 2, Llama 3, etc.)
    - Local inference with GPU acceleration
    - Rate limiting for resource management
    - Error handling
    - Async operation
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        requests_per_minute: int = 60,
        temperature: float = 0.7,
        max_new_tokens: Optional[int] = None,
        **kwargs: Any
    ):
        """Initialize Llama model.
        
        Args:
            model_name: Name or path of the Llama model to use
            device: Device to run the model on ("cuda" or "cpu")
            max_retries: Maximum number of retries for failed inference
            retry_delay: Delay between retries in seconds
            requests_per_minute: Maximum requests per minute for rate limiting
            temperature: Sampling temperature (0-1)
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.device = device
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.kwargs = kwargs
        
        # Load model and tokenizer
        logger.info(f"Loading model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Configure generation
        self.generation_config = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.model_name,
            "type": "llama",
            "device": self.device,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
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
    
    async def generate(self, input_data: str, **kwargs) -> str:
        """Generate text from input.
        
        Args:
            input_data: Input text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            ValueError: If input is invalid
            Exception: If inference fails
        """
        # Validate input
        if error := self.validate_input(input_data):
            raise ValueError(error)
        
        # Merge parameters
        generation_config = GenerationConfig(
            **self.generation_config.to_dict(),
            **kwargs
        )
        
        # Retry loop
        for attempt in range(self.max_retries):
            try:
                # Acquire rate limit token
                await self.rate_limiter.acquire()
                
                # Tokenize input
                inputs = self.tokenizer(input_data, return_tensors="pt").to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                
                # Decode output
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(
                    f"Inference attempt {attempt + 1} failed: {str(e)}. "
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
            Exception: If inference fails
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
register_model("llama")(LlamaModel) 