"""Data processing and transformation utilities."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel

T = TypeVar("T", bound=Union[BaseModel, dict])


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Data to save
        path: Path to save the file
        indent: Number of spaces for indentation
        
    Raises:
        OSError: If the file cannot be written
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def to_dict(obj: Union[BaseModel, dict]) -> Dict[str, Any]:
    """Convert an object to a dictionary.
    
    Args:
        obj: Object to convert (Pydantic model or dict)
        
    Returns:
        Dictionary representation of the object
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj


def from_dict(data: Dict[str, Any], model_class: type[T]) -> T:
    """Convert a dictionary to a model instance.
    
    Args:
        data: Dictionary to convert
        model_class: Pydantic model class
        
    Returns:
        Model instance
    """
    return model_class.model_validate(data)


def batch_data(
    data: List[Any], batch_size: int, drop_last: bool = False
) -> List[List[Any]]:
    """Split data into batches.
    
    Args:
        data: List of data items
        batch_size: Size of each batch
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        List of batches
    """
    if drop_last:
        return [
            data[i : i + batch_size]
            for i in range(0, len(data) - batch_size + 1, batch_size)
        ]
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a nested list.
    
    Args:
        nested_list: List of lists
        
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def chunk_dataframe(
    df: pd.DataFrame, chunk_size: int
) -> List[pd.DataFrame]:
    """Split a DataFrame into chunks.
    
    Args:
        df: DataFrame to split
        chunk_size: Size of each chunk
        
    Returns:
        List of DataFrame chunks
    """
    return [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]


def normalize_array(
    arr: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """Normalize a numpy array to [0, 1] or [min_val, max_val].
    
    Args:
        arr: Array to normalize
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
        
    Returns:
        Normalized array
    """
    arr_min = arr.min()
    arr_max = arr.max()
    
    if min_val is None:
        min_val = 0.0
    if max_val is None:
        max_val = 1.0
        
    return (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val


def pad_sequence(
    sequence: List[Any],
    max_length: int,
    padding_value: Any = 0,
    padding_side: str = "right",
) -> List[Any]:
    """Pad a sequence to a fixed length.
    
    Args:
        sequence: Sequence to pad
        max_length: Target length
        padding_value: Value to use for padding
        padding_side: Where to add padding ("left" or "right")
        
    Returns:
        Padded sequence
    """
    if len(sequence) >= max_length:
        return sequence[:max_length]
        
    padding = [padding_value] * (max_length - len(sequence))
    if padding_side == "left":
        return padding + sequence
    return sequence + padding


def truncate_sequence(
    sequence: List[Any], max_length: int, truncate_side: str = "right"
) -> List[Any]:
    """Truncate a sequence to a maximum length.
    
    Args:
        sequence: Sequence to truncate
        max_length: Maximum length
        truncate_side: Which side to truncate from ("left" or "right")
        
    Returns:
        Truncated sequence
    """
    if len(sequence) <= max_length:
        return sequence
        
    if truncate_side == "left":
        return sequence[-max_length:]
    return sequence[:max_length]


def merge_dicts(
    dicts: List[Dict[str, Any]], merge_lists: bool = True
) -> Dict[str, Any]:
    """Merge multiple dictionaries.
    
    Args:
        dicts: List of dictionaries to merge
        merge_lists: Whether to merge lists instead of overwriting
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        for key, value in d.items():
            if (
                key in result
                and isinstance(result[key], list)
                and isinstance(value, list)
                and merge_lists
            ):
                result[key].extend(value)
            else:
                result[key] = value
    return result 