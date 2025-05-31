"""Configuration models for the evaluation framework."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import GenerateJsonSchema
import yaml


class MetricConfig(BaseModel):
    """Configuration for a single metric.
    
    Attributes:
        name: Name of the metric
        type: Type/class of the metric
        params: Optional parameters for the metric
        weight: Optional weight for weighted averaging (default: 1.0)
    """

    name: str = Field(..., description="Name of the metric")
    type: str = Field(..., description="Type/class of the metric")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional parameters for the metric",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for weighted averaging",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate that the metric type is valid.
        
        Args:
            v: The metric type to validate.
            
        Returns:
            The validated metric type.
            
        Raises:
            ValueError: If the metric type is invalid.
        """
        # TODO: Add actual validation against available metric types
        if not v or not v.strip():
            raise ValueError("Metric type cannot be empty")
        return v.strip()


class DatasetConfig(BaseModel):
    """Configuration for a dataset.
    
    Attributes:
        name: Name of the dataset
        type: Type/class of the dataset
        path: Path to the dataset file or directory
        params: Optional parameters for dataset loading
        batch_size: Batch size for evaluation (default: 32)
        shuffle: Whether to shuffle the dataset (default: True)
    """

    name: str = Field(..., description="Name of the dataset")
    type: str = Field(..., description="Type/class of the dataset")
    path: Union[str, Path] = Field(..., description="Path to the dataset")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional parameters for dataset loading",
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Batch size for evaluation",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle the dataset",
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Union[str, Path]) -> Path:
        """Validate that the dataset path exists.
        
        Args:
            v: The path to validate.
            
        Returns:
            The validated Path object.
            
        Raises:
            ValueError: If the path doesn't exist.
        """
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Dataset path does not exist: {path}")
        return path


class ModelConfig(BaseModel):
    """Configuration for a model.
    
    Attributes:
        name: Name of the model
        type: Type/class of the model
        path: Path to the model file or directory
        params: Optional parameters for model initialization
        device: Device to run the model on (default: "cpu")
    """

    name: str = Field(..., description="Name of the model")
    type: str = Field(..., description="Type/class of the model")
    path: Optional[Union[str, Path]] = Field(
        None,
        description="Path to the model file or directory",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional parameters for model initialization",
    )
    device: str = Field(
        default="cpu",
        description="Device to run the model on",
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate that the device is valid.
        
        Args:
            v: The device to validate.
            
        Returns:
            The validated device.
            
        Raises:
            ValueError: If the device is invalid.
        """
        valid_devices = {"cpu", "cuda", "mps"}
        if v.lower() not in valid_devices:
            raise ValueError(f"Invalid device: {v}. Must be one of {valid_devices}")
        return v.lower()


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation run.
    
    Attributes:
        name: Name of the evaluation run
        model: Model configuration
        dataset: Dataset configuration
        metrics: List of metric configurations
        output_dir: Directory to save results (default: "results")
        seed: Random seed for reproducibility (default: 42)
        num_workers: Number of worker processes (default: 1)
        verbose: Whether to print progress (default: True)
    """

    name: str = Field(..., description="Name of the evaluation run")
    model: ModelConfig = Field(..., description="Model configuration")
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
    metrics: List[MetricConfig] = Field(..., description="List of metric configurations")
    output_dir: Union[str, Path] = Field(
        default="results",
        description="Directory to save results",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    num_workers: int = Field(
        default=1,
        ge=1,
        description="Number of worker processes",
    )
    verbose: bool = Field(
        default=True,
        description="Whether to print progress",
    )

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Union[str, Path]) -> Path:
        """Create output directory if it doesn't exist.
        
        Args:
            v: The output directory path.
            
        Returns:
            The validated Path object.
        """
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EvaluationConfig":
        """Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML file.
            
        Returns:
            The loaded configuration.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            yaml.YAMLError: If the YAML is invalid.
            ValidationError: If the configuration is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls.model_validate(config_dict)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "EvaluationConfig":
        """Load configuration from a JSON file.
        
        Args:
            path: Path to the JSON file.
            
        Returns:
            The loaded configuration.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the JSON is invalid.
            ValidationError: If the configuration is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        return cls.model_validate_json(path.read_text())

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.
        
        Args:
            path: Path to save the YAML file.
        """
        path = Path(path)
        config_dict = self.model_dump()
        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file.
        
        Args:
            path: Path to save the JSON file.
        """
        path = Path(path)
        path.write_text(self.model_dump_json(indent=2)) 