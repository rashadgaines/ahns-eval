"""CSV dataset implementation for the evaluation framework."""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from eval_framework.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

class CsvDataset(BaseDataset[Dict[str, Any]]):
    """CSV dataset implementation.
    
    This class provides functionality for loading and validating CSV datasets.
    It uses pandas for efficient CSV handling and supports various CSV formats.
    """
    
    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        delimiter: str = ',',
        encoding: str = 'utf-8',
        **kwargs: Any
    ):
        """Initialize CSV dataset.
        
        Args:
            path: Path to CSV file
            delimiter: CSV delimiter character
            encoding: File encoding
            **kwargs: Additional dataset parameters
        """
        self.delimiter = delimiter
        self.encoding = encoding
        super().__init__(path=path, **kwargs)
    
    def load(self) -> None:
        """Load CSV data from file.
        
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            pd.errors.EmptyDataError: If CSV file is empty
            ValueError: If data format is invalid
        """
        if not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")
        
        try:
            # Read CSV with pandas
            df = pd.read_csv(
                self.path,
                delimiter=self.delimiter,
                encoding=self.encoding,
                dtype=str  # Read all columns as strings initially
            )
            
            # Convert to list of dictionaries
            self._data = df.to_dict('records')
            
            self.validate()
            
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")
    
    def validate(self) -> bool:
        """Validate CSV dataset integrity.
        
        Returns:
            True if dataset is valid
            
        Raises:
            ValueError: If validation fails
        """
        # Call parent validation
        super().validate()
        
        # Validate CSV structure
        if not self._data:
            raise ValueError("CSV dataset is empty")
        
        # Check all records have same fields
        fields = set(self._data[0].keys())
        for i, record in enumerate(self._data[1:], 1):
            if set(record.keys()) != fields:
                raise ValueError(
                    f"Record at index {i} has different fields than first record"
                )
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information.
        
        Returns:
            Dictionary containing dataset information
        """
        info = super().get_info()
        info.update({
            "format": "csv",
            "delimiter": self.delimiter,
            "encoding": self.encoding,
            "fields": list(self._data[0].keys()) if self._data else []
        })
        return info 