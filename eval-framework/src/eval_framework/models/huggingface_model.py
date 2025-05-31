"""Hugging Face Transformers model implementation for the evaluation framework."""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig
)

from eval_framework.models.base import BaseModel

logger = logging.getLogger(__name__)

class HuggingFaceModel(BaseModel[str, str]):
    """Hugging Face Transformers model implementation.
    
    This class provides an interface to Hugging Face Transformers models with support for:
    - Local model loading and caching
    - GPU acceleration
    - Batch inference
    - Various model types (causal, seq2seq)
    - Generation configuration
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: str = "causal",  # or "seq2seq"
        device: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.7,
        max_new_tokens: Optional[int] = None,
        batch_size: int = 1,
        **kwargs: Any
    ):
        """Initialize Hugging Face model.
        
        Args:
            model_name: Name or path of the model to load
            model_type: Type of model ("causal" or "seq2seq")
            device: Device to run model on ("cuda", "cpu", etc.)
            max_retries: Maximum number of retries for failed inference
            retry_delay: Delay between retries in seconds
            temperature: Sampling temperature (0-1)
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for inference
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.kwargs = kwargs
        
        # Initialize model and tokenizer
        logger.info(f"Loading model {model_name} on {self.device}...")
        self._load_model()
        
        # Configure generation
        self.generation_config = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    def _load_model(self) -> None:
        """Load model and tokenizer.
        
        Raises:
            ValueError: If model loading fails
        """
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model based on type
            if self.model_type == "causal":
                model_class = AutoModelForCausalLM
            elif self.model_type == "seq2seq":
                model_class = AutoModelForSeq2SeqLM
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Load model with appropriate device and dtype
            self.model = model_class.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Move model to device if not using device_map
            if self.device != "cuda" or not hasattr(self.model, "device_map"):
                self.model = self.model.to(self.device)
            
            # Set model to eval mode
            self.model.eval()
            
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.model_name,
            "type": f"huggingface_{self.model_type}",
            "device": self.device,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "batch_size": self.batch_size,
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
            Exception: If generation fails
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
                # Tokenize input
                inputs = self.tokenizer(
                    input_data,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
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
                    f"Generation attempt {attempt + 1} failed: {str(e)}. "
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
            Exception: If generation fails
        """
        # Validate inputs
        for input_data in batch_inputs:
            if error := self.validate_input(input_data):
                raise ValueError(f"Invalid input: {error}")
        
        # Process in batches
        results = []
        for i in range(0, len(batch_inputs), self.batch_size):
            batch = batch_inputs[i:i + self.batch_size]
            
            # Retry loop
            for attempt in range(self.max_retries):
                try:
                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    
                    # Generate
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=self.generation_config
                        )
                    
                    # Decode outputs
                    batch_results = [
                        self.tokenizer.decode(output, skip_special_tokens=True)
                        for output in outputs
                    ]
                    results.extend(batch_results)
                    break
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(
                        f"Batch generation attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {self.retry_delay} seconds..."
                    )
                    await asyncio.sleep(self.retry_delay)
        
        return results 