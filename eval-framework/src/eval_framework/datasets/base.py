"""Base dataset interface for the evaluation framework."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar, Union
import random
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')

class BaseDataset(Generic[T], ABC):
    """Base dataset interface for the evaluation framework.
    
    This abstract base class defines the interface for all datasets in the framework.
    It provides common functionality for:
    - Loading and validating data
    - Iterating over dataset items
    - Sampling subsets of data
    - Data integrity checks
    
    Type Parameters:
        T: The type of items in the dataset
    """
    
    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        **kwargs: Any
    ):
        """Initialize dataset.
        
        Args:
            path: Path to dataset file or directory
            seed: Random seed for reproducibility
            **kwargs: Additional dataset-specific parameters
        """
        self.path = Path(path) if path else None
        self.seed = seed
        self.kwargs = kwargs
        self._data: Optional[List[T]] = None
        self._rng = random.Random(seed)
        
        # Load data if path is provided
        if path:
            self.load()
    
    @abstractmethod
    def load(self) -> None:
        """Load dataset from source.
        
        This method should be implemented by subclasses to load data from
        the specified path or other source. The loaded data should be stored
        in self._data.
        
        Raises:
            FileNotFoundError: If dataset file/directory doesn't exist
            ValueError: If data format is invalid
        """
        pass
    
    def validate(self) -> bool:
        """Validate dataset integrity.
        
        This method checks if the dataset is properly loaded and contains
        valid data. Subclasses should override this method to implement
        dataset-specific validation logic.
        
        Returns:
            True if dataset is valid, False otherwise
            
        Raises:
            ValueError: If validation fails with specific error message
        """
        if self._data is None:
            raise ValueError("Dataset not loaded")
        
        if not isinstance(self._data, list):
            raise ValueError("Dataset must be a list")
        
        if not self._data:
            raise ValueError("Dataset is empty")
        
        return True
    
    def sample(self, size: Optional[int] = None, fraction: Optional[float] = None) -> List[T]:
        """Sample a subset of the dataset.
        
        Args:
            size: Number of items to sample (mutually exclusive with fraction)
            fraction: Fraction of dataset to sample (0-1, mutually exclusive with size)
            
        Returns:
            List of sampled items
            
        Raises:
            ValueError: If dataset is not loaded or sampling parameters are invalid
        """
        if self._data is None:
            raise ValueError("Dataset not loaded")
        
        if size is not None and fraction is not None:
            raise ValueError("Cannot specify both size and fraction")
        
        if size is not None:
            if not isinstance(size, int) or size < 0:
                raise ValueError("Size must be a non-negative integer")
            if size > len(self._data):
                raise ValueError("Sample size cannot be larger than dataset")
            return self._rng.sample(self._data, size)
        
        if fraction is not None:
            if not isinstance(fraction, float) or not 0 < fraction <= 1:
                raise ValueError("Fraction must be a float between 0 and 1")
            size = max(1, int(len(self._data) * fraction))
            return self._rng.sample(self._data, size)
        
        raise ValueError("Must specify either size or fraction")
    
    def __iter__(self) -> Iterator[T]:
        """Iterate over dataset items.
        
        Returns:
            Iterator over dataset items
            
        Raises:
            ValueError: If dataset is not loaded
        """
        if self._data is None:
            raise ValueError("Dataset not loaded")
        return iter(self._data)
    
    def __len__(self) -> int:
        """Get dataset size.
        
        Returns:
            Number of items in dataset
            
        Raises:
            ValueError: If dataset is not loaded
        """
        if self._data is None:
            raise ValueError("Dataset not loaded")
        return len(self._data)
    
    def __getitem__(self, index: int) -> T:
        """Get item at index.
        
        Args:
            index: Item index
            
        Returns:
            Item at index
            
        Raises:
            ValueError: If dataset is not loaded
            IndexError: If index is out of range
        """
        if self._data is None:
            raise ValueError("Dataset not loaded")
        return self._data[index]
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            "path": str(self.path) if self.path else None,
            "size": len(self) if self._data is not None else None,
            "seed": self.seed,
            **self.kwargs
        } 