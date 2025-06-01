"""Grok model implementation for the evaluation framework."""

import os
import time
from typing import List, Optional, Dict, Any
import asyncio
import json
from openai import AsyncOpenAI
from eval_framework.core.base import BaseModel
from eval_framework.core.registry import register_model

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int):
        """Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum number of API calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.calls = []
        
    async def acquire(self):
        """Acquire permission to make an API call."""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        
        # If we've hit the limit, wait until we can make another call
        if len(self.calls) >= self.calls_per_minute:
            wait_time = 60 - (now - self.calls[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.calls = self.calls[1:]
            
        self.calls.append(now)

@register_model("grok")
class GrokModel(BaseModel[str, str]):
    """Grok model implementation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        batch_size: int = 1,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model_type: str = "chat",
        system_prompt: str = "You are a helpful AI assistant.",
        **kwargs
    ):
        """Initialize the Grok model.
        
        Args:
            api_key: API key for Grok. If not provided, will try to get from GROK_API_KEY env var
            batch_size: Number of prompts to process in parallel
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            model_type: Type of model to use (chat or completion)
            system_prompt: System prompt to use for chat models
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        if not self.api_key:
            raise ValueError("Grok API key must be provided or set in GROK_API_KEY environment variable")
            
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model_type = model_type
        self.system_prompt = system_prompt
        
        # Initialize Grok client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )
        
        # Initialize rate limiter (adjust based on your API tier)
        self.rate_limiter = RateLimiter(calls_per_minute=60)
        
    async def predict(self, inputs: str) -> str:
        """Generate a response for a single prompt.
        
        Args:
            inputs: The input prompt text
            
        Returns:
            The generated response text
        """
        await self.rate_limiter.acquire()
        
        try:
            if self.model_type == "chat":
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": inputs}
                ]
                
                response = await self.client.chat.completions.create(
                    model="grok-3-latest",
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                
                # Process any tool calls if present
                if response.choices[0].message.tool_calls:
                    messages.append(response.choices[0].message)
                    
                    # Process tool calls (in this case, we'll just return a placeholder)
                    for tool_call in response.choices[0].message.tool_calls:
                        messages.append({
                            "role": "tool",
                            "content": json.dumps({"status": "not_implemented", "message": "Tool calls not implemented in this version"}),
                            "tool_call_id": tool_call.id
                        })
                    
                    # Get final response after tool calls
                    response = await self.client.chat.completions.create(
                        model="grok-3-latest",
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p
                    )
                
                return response.choices[0].message.content
            else:
                response = await self.client.completions.create(
                    model="grok-3-latest",
                    prompt=inputs,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                return response.choices[0].text
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    async def batch_predict(self, inputs: List[str]) -> List[str]:
        """Generate responses for multiple prompts.
        
        Args:
            inputs: List of input prompt texts
            
        Returns:
            List of generated response texts
        """
        # Process prompts in batches
        responses = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            batch_responses = await asyncio.gather(
                *[self.predict(prompt) for prompt in batch]
            )
            responses.extend(batch_responses)
        return responses 