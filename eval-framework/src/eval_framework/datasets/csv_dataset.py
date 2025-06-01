"""CSV dataset implementation for the evaluation framework."""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import pandas as pd

from eval_framework.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

class CsvDataset(BaseDataset[str]):
    """Dataset implementation for CSV files."""

    def __init__(
        self,
        path: Optional[Path] = None,
        input_column: str = "input",
        target_column: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ):
        """Initialize the dataset.
        
        Args:
            path: Path to the CSV file
            input_column: Name of the input column
            target_column: Optional name of the target column
            data: Optional pre-loaded DataFrame
        """
        self.path = path
        self.input_column = input_column
        self.target_column = target_column
        self._data = data
        self._loaded = data is not None

    async def load(self) -> None:
        """Load the dataset data."""
        if self._loaded:
            return

        if self._data is not None:
            self._validate_data(self._data)
            self._loaded = True
            return

        if self.path is None:
            raise ValueError("Either path or data must be provided")

        if not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")

        try:
            self._data = pd.read_csv(self.path)
            self._validate_data(self._data)
            self._loaded = True
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate the dataset data.
        
        Args:
            data: The DataFrame to validate
            
        Raises:
            ValueError: If the data is invalid
        """
        if data.empty:
            raise ValueError("Dataset is empty")
            
        if self.input_column not in data.columns:
            raise ValueError(f"Input column '{self.input_column}' not found in dataset")
            
        if self.target_column and self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

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
        if not self._data.empty:
            # Check all records have same fields
            fields = set(self._data.columns)
            for i, record in enumerate(self._data.iloc[1:], 1):
                if set(record.keys()) != fields:
                    raise ValueError(
                        f"Record at index {i} has different fields than first record"
                    )
        
        # Check input column exists
        if self.input_column not in self._data.columns:
            raise ValueError(f"Input column '{self.input_column}' not found in CSV")
        
        return True
    
    async def get_all(self) -> tuple[List[str], List[None]]:
        """Get all data in the dataset.
        
        Returns:
            Tuple of (inputs, targets) for all data.
        """
        if not self._loaded:
            await self.load()
            
        if self._data is None or self._data.empty:
            raise ValueError("Dataset not loaded")
            
        # Extract input text from each record
        items = self._data[self.input_column].tolist()
        return items, []  # Return empty labels list
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information.
        
        Returns:
            Dictionary containing dataset information
        """
        info = super().get_info()
        info.update({
            "format": "csv",
            "input_column": self.input_column,
            "target_column": self.target_column,
            "fields": list(self._data.columns) if not self._data.empty else []
        })
        return info 