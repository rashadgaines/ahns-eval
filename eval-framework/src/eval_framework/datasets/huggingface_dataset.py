"""Hugging Face dataset implementation for the evaluation framework."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset, Dataset, DatasetDict
from datasets.features import Features

from eval_framework.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

class HuggingFaceDataset(BaseDataset[Dict[str, Any]]):
    """Hugging Face dataset implementation.
    
    This class provides functionality for loading and validating datasets
    from the Hugging Face Hub. It supports various dataset formats and
    configurations.
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        config_name: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ):
        """Initialize Hugging Face dataset.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            split: Dataset split to load (e.g., 'train', 'test')
            config_name: Dataset configuration name
            cache_dir: Directory to cache downloaded datasets
            **kwargs: Additional dataset parameters
        """
        self.dataset_name = dataset_name
        self.split = split
        self.config_name = config_name
        self.cache_dir = str(cache_dir) if cache_dir else None
        self._dataset: Optional[Dataset] = None
        super().__init__(**kwargs)
    
    def load(self) -> None:
        """Load dataset from Hugging Face Hub.
        
        Raises:
            ValueError: If dataset loading fails
        """
        try:
            # Load dataset
            dataset = load_dataset(
                self.dataset_name,
                name=self.config_name,
                split=self.split,
                cache_dir=self.cache_dir
            )
            
            # Handle different dataset types
            if isinstance(dataset, DatasetDict):
                if self.split is None:
                    raise ValueError(
                        "Dataset has multiple splits. Please specify a split."
                    )
                dataset = dataset[self.split]
            
            self._dataset = dataset
            
            # Convert to list of dictionaries
            self._data = dataset.to_list()
            
            self.validate()
            
        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")
    
    def validate(self) -> bool:
        """Validate dataset integrity.
        
        Returns:
            True if dataset is valid
            
        Raises:
            ValueError: If validation fails
        """
        # Call parent validation
        super().validate()
        
        # Validate dataset structure
        if not isinstance(self._dataset, Dataset):
            raise ValueError("Invalid dataset type")
        
        # Check features
        if not isinstance(self._dataset.features, Features):
            raise ValueError("Invalid dataset features")
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information.
        
        Returns:
            Dictionary containing dataset information
        """
        info = super().get_info()
        info.update({
            "format": "huggingface",
            "dataset_name": self.dataset_name,
            "split": self.split,
            "config_name": self.config_name,
            "cache_dir": self.cache_dir,
            "features": list(self._dataset.features.keys()) if self._dataset else []
        })
        return info
    
    def get_features(self) -> Features:
        """Get dataset features.
        
        Returns:
            Dataset features
            
        Raises:
            ValueError: If dataset is not loaded
        """
        if self._dataset is None:
            raise ValueError("Dataset not loaded")
        return self._dataset.features 