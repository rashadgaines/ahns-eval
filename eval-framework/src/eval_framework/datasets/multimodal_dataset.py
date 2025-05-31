"""Multimodal dataset implementation for the evaluation framework."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from eval_framework.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

class MultimodalDataset(BaseDataset[Dict[str, Any]]):
    """Multimodal dataset implementation.
    
    This class provides functionality for loading and validating datasets
    containing both images and text. It supports various image formats and
    text encodings.
    """
    
    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        image_dir: Optional[Union[str, Path]] = None,
        image_size: Tuple[int, int] = (224, 224),
        image_channels: int = 3,
        text_encoding: str = 'utf-8',
        **kwargs: Any
    ):
        """Initialize multimodal dataset.
        
        Args:
            path: Path to dataset file (JSON/CSV)
            image_dir: Directory containing images
            image_size: Target image size (height, width)
            image_channels: Number of image channels
            text_encoding: Text encoding
            **kwargs: Additional dataset parameters
        """
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_size = image_size
        self.image_channels = image_channels
        self.text_encoding = text_encoding
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        super().__init__(path=path, **kwargs)
    
    def load(self) -> None:
        """Load multimodal dataset.
        
        Raises:
            FileNotFoundError: If dataset file or image directory doesn't exist
            ValueError: If data format is invalid
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        
        if self.image_dir and not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Load data based on file extension
        if self.path.suffix == '.json':
            self._load_json()
        elif self.path.suffix == '.csv':
            self._load_csv()
        else:
            raise ValueError(f"Unsupported file format: {self.path.suffix}")
        
        self.validate()
    
    def _load_json(self) -> None:
        """Load JSON data."""
        import json
        with open(self.path, 'r', encoding=self.text_encoding) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            self._data = data
        elif isinstance(data, dict) and 'data' in data:
            self._data = data['data']
        else:
            raise ValueError("Invalid JSON format")
    
    def _load_csv(self) -> None:
        """Load CSV data."""
        import pandas as pd
        df = pd.read_csv(self.path, encoding=self.text_encoding)
        self._data = df.to_dict('records')
    
    def validate(self) -> bool:
        """Validate dataset integrity.
        
        Returns:
            True if dataset is valid
            
        Raises:
            ValueError: If validation fails
        """
        # Call parent validation
        super().validate()
        
        # Validate data structure
        required_fields = {'image_path', 'text'}
        for i, item in enumerate(self._data):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {i} must be a dictionary")
            
            # Check required fields
            missing_fields = required_fields - set(item.keys())
            if missing_fields:
                raise ValueError(
                    f"Item at index {i} missing required fields: {missing_fields}"
                )
            
            # Validate image path
            if self.image_dir:
                image_path = self.image_dir / item['image_path']
                if not image_path.exists():
                    raise ValueError(f"Image not found: {image_path}")
            
            # Validate text
            if not isinstance(item['text'], str):
                raise ValueError(f"Text at index {i} must be a string")
        
        return True
    
    def load_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
            
        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If image is invalid
        """
        if self.image_dir:
            image_path = self.image_dir / image_path
        
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get dataset item.
        
        Args:
            index: Item index
            
        Returns:
            Dictionary containing image tensor and text
            
        Raises:
            ValueError: If dataset is not loaded
            IndexError: If index is out of range
        """
        item = super().__getitem__(index)
        
        # Load and preprocess image
        image = self.load_image(item['image_path'])
        
        return {
            'image': image,
            'text': item['text']
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information.
        
        Returns:
            Dictionary containing dataset information
        """
        info = super().get_info()
        info.update({
            "format": "multimodal",
            "image_dir": str(self.image_dir) if self.image_dir else None,
            "image_size": self.image_size,
            "image_channels": self.image_channels,
            "text_encoding": self.text_encoding
        })
        return info 