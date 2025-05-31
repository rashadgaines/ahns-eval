"""Dataset preprocessing pipeline implementation."""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class SamplingStrategy(Enum):
    """Sampling strategies for dataset preprocessing."""
    RANDOM = "random"
    SEQUENTIAL = "sequential"
    STRATIFIED = "stratified"
    WEIGHTED = "weighted"

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    # Text preprocessing
    max_length: Optional[int] = None
    truncation: bool = True
    padding: bool = True
    lowercase: bool = True
    remove_punctuation: bool = False
    remove_stopwords: bool = False
    
    # Image preprocessing
    image_size: Tuple[int, int] = (224, 224)
    image_channels: int = 3
    normalize: bool = True
    augment: bool = False
    
    # Sampling
    sampling_strategy: SamplingStrategy = SamplingStrategy.RANDOM
    sampling_seed: Optional[int] = None
    sampling_weights: Optional[List[float]] = None
    
    # Validation
    validate_text: bool = True
    validate_image: bool = True
    min_text_length: int = 1
    max_text_length: Optional[int] = None
    allowed_image_formats: List[str] = None
    
    def __post_init__(self):
        """Set default values for optional fields."""
        if self.allowed_image_formats is None:
            self.allowed_image_formats = ['.jpg', '.jpeg', '.png', '.bmp']

class BasePreprocessor(ABC):
    """Base class for data preprocessors."""
    
    def __init__(self, config: PreprocessingConfig):
        """Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess data.
        
        Args:
            data: Input data
            
        Returns:
            Preprocessed data
        """
        pass

class TextPreprocessor(BasePreprocessor):
    """Text data preprocessor."""
    
    def __init__(
        self,
        config: PreprocessingConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """Initialize text preprocessor.
        
        Args:
            config: Preprocessing configuration
            tokenizer: Optional tokenizer for tokenization
        """
        super().__init__(config)
        self.tokenizer = tokenizer
    
    def preprocess(self, text: str) -> Union[str, Dict[str, torch.Tensor]]:
        """Preprocess text data.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text or tokenized tensors
        """
        # Basic text cleaning
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_punctuation:
            import string
            text = text.translate(str.maketrans("", "", string.punctuation))
        
        if self.config.remove_stopwords:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text)
            text = ' '.join([w for w in word_tokens if w not in stop_words])
        
        # Tokenization if tokenizer is provided
        if self.tokenizer is not None:
            return self.tokenizer(
                text,
                max_length=self.config.max_length,
                truncation=self.config.truncation,
                padding=self.config.padding,
                return_tensors="pt"
            )
        
        return text

class ImagePreprocessor(BasePreprocessor):
    """Image data preprocessor."""
    
    def __init__(self, config: PreprocessingConfig):
        """Initialize image preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        super().__init__(config)
        
        # Basic transforms
        transforms_list = [
            transforms.Resize(self.config.image_size),
            transforms.ToTensor()
        ]
        
        # Add normalization
        if self.config.normalize:
            transforms_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        # Add augmentation
        if self.config.augment:
            transforms_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                )
            ])
        
        self.transform = transforms.Compose(transforms_list)
    
    def preprocess(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Preprocess image data.
        
        Args:
            image: Input image (path or PIL Image)
            
        Returns:
            Preprocessed image tensor
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Apply transforms
        return self.transform(image)

class DataValidator:
    """Data validation utility."""
    
    def __init__(self, config: PreprocessingConfig):
        """Initialize data validator.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
    
    def validate_text(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate text data.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(text, str):
            return False, "Text must be a string"
        
        if not text.strip():
            return False, "Text cannot be empty"
        
        if self.config.validate_text:
            length = len(text)
            if length < self.config.min_text_length:
                return False, f"Text too short (min: {self.config.min_text_length})"
            if (self.config.max_text_length is not None and
                length > self.config.max_text_length):
                return False, f"Text too long (max: {self.config.max_text_length})"
        
        return True, None
    
    def validate_image(
        self,
        image: Union[str, Path, Image.Image]
    ) -> Tuple[bool, Optional[str]]:
        """Validate image data.
        
        Args:
            image: Image to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.config.validate_image:
            # Check file format if path is provided
            if isinstance(image, (str, Path)):
                path = Path(image)
                if not path.exists():
                    return False, f"Image file not found: {path}"
                if path.suffix.lower() not in self.config.allowed_image_formats:
                    return False, f"Unsupported image format: {path.suffix}"
            
            # Check image dimensions
            try:
                if isinstance(image, (str, Path)):
                    img = Image.open(image)
                else:
                    img = image
                
                if img.mode != 'RGB':
                    return False, f"Image must be RGB (got {img.mode})"
                
                if img.size != self.config.image_size:
                    return False, (
                        f"Image size mismatch (expected {self.config.image_size}, "
                        f"got {img.size})"
                    )
                
            except Exception as e:
                return False, f"Error validating image: {str(e)}"
        
        return True, None

class Sampler:
    """Dataset sampling utility."""
    
    def __init__(
        self,
        config: PreprocessingConfig,
        data: List[Any],
        labels: Optional[List[Any]] = None
    ):
        """Initialize sampler.
        
        Args:
            config: Preprocessing configuration
            data: Dataset to sample from
            labels: Optional labels for stratified sampling
        """
        self.config = config
        self.data = data
        self.labels = labels
        
        # Set random seed
        if self.config.sampling_seed is not None:
            random.seed(self.config.sampling_seed)
            np.random.seed(self.config.sampling_seed)
    
    def sample(self, size: Union[int, float]) -> List[Any]:
        """Sample from dataset.
        
        Args:
            size: Number of samples or fraction of dataset
            
        Returns:
            Sampled data
        """
        # Calculate sample size
        if isinstance(size, float):
            if not 0 < size <= 1:
                raise ValueError("Fraction must be between 0 and 1")
            size = int(len(self.data) * size)
        
        # Apply sampling strategy
        if self.config.sampling_strategy == SamplingStrategy.RANDOM:
            return random.sample(self.data, size)
        
        elif self.config.sampling_strategy == SamplingStrategy.SEQUENTIAL:
            return self.data[:size]
        
        elif self.config.sampling_strategy == SamplingStrategy.STRATIFIED:
            if self.labels is None:
                raise ValueError("Labels required for stratified sampling")
            return self._stratified_sample(size)
        
        elif self.config.sampling_strategy == SamplingStrategy.WEIGHTED:
            if self.config.sampling_weights is None:
                raise ValueError("Weights required for weighted sampling")
            return self._weighted_sample(size)
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.config.sampling_strategy}")
    
    def _stratified_sample(self, size: int) -> List[Any]:
        """Perform stratified sampling.
        
        Args:
            size: Number of samples
            
        Returns:
            Stratified sample
        """
        # Calculate samples per class
        unique_labels = set(self.labels)
        samples_per_class = size // len(unique_labels)
        
        # Sample from each class
        sampled_indices = []
        for label in unique_labels:
            class_indices = [i for i, l in enumerate(self.labels) if l == label]
            sampled_indices.extend(random.sample(class_indices, samples_per_class))
        
        # Add remaining samples randomly
        remaining = size - len(sampled_indices)
        if remaining > 0:
            remaining_indices = set(range(len(self.data))) - set(sampled_indices)
            sampled_indices.extend(random.sample(list(remaining_indices), remaining))
        
        return [self.data[i] for i in sampled_indices]
    
    def _weighted_sample(self, size: int) -> List[Any]:
        """Perform weighted sampling.
        
        Args:
            size: Number of samples
            
        Returns:
            Weighted sample
        """
        if len(self.config.sampling_weights) != len(self.data):
            raise ValueError("Number of weights must match dataset size")
        
        # Normalize weights
        weights = np.array(self.config.sampling_weights)
        weights = weights / weights.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(self.data),
            size=size,
            replace=False,
            p=weights
        )
        
        return [self.data[i] for i in indices] 