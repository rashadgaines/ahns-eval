"""JSON dataset implementation for the evaluation framework."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from eval_framework.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

class JsonDataset(BaseDataset[Dict[str, Any]]):
    """JSON dataset implementation.
    
    This class provides functionality for loading and validating JSON datasets.
    It supports both single JSON objects and arrays of objects.
    
    Example JSON formats:
    - Array of objects: [{"text": "example1"}, {"text": "example2"}]
    - Single object: {"data": [{"text": "example1"}, {"text": "example2"}]}
    """
    
    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        data_key: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize JSON dataset.
        
        Args:
            path: Path to JSON file
            data_key: Key to extract data from JSON object (if not array)
            **kwargs: Additional dataset parameters
        """
        self.data_key = data_key
        super().__init__(path=path, **kwargs)
    
    def load(self) -> None:
        """Load JSON data from file.
        
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is invalid
            ValueError: If data format is invalid
        """
        if not self.path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.path}")
        
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                self._data = data
            elif isinstance(data, dict):
                if self.data_key is None:
                    raise ValueError(
                        "JSON is an object but no data_key specified. "
                        "Either provide data_key or use array format."
                    )
                if self.data_key not in data:
                    raise ValueError(f"Data key '{self.data_key}' not found in JSON")
                if not isinstance(data[self.data_key], list):
                    raise ValueError(f"Data at key '{self.data_key}' must be a list")
                self._data = data[self.data_key]
            else:
                raise ValueError("JSON must be either an array or an object")
            
            self.validate()
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
    
    def validate(self) -> bool:
        """Validate JSON dataset integrity.
        
        Returns:
            True if dataset is valid
            
        Raises:
            ValueError: If validation fails
        """
        # Call parent validation
        super().validate()
        
        # Validate JSON structure
        for i, item in enumerate(self._data):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {i} must be a dictionary")
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information.
        
        Returns:
            Dictionary containing dataset information
        """
        info = super().get_info()
        info.update({
            "format": "json",
            "data_key": self.data_key,
            "fields": list(self._data[0].keys()) if self._data else []
        })
        return info 