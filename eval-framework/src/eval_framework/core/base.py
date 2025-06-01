"""Core base classes for the evaluation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

# Type variables for generic classes
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
PredictionT = TypeVar("PredictionT")
TargetT = TypeVar("TargetT")
MetricValueT = TypeVar("MetricValueT", bound=Union[float, int, bool, str])


class BaseModel(ABC, Generic[InputT, OutputT]):
    """Abstract base class for AI models.
    
    This class defines the interface that all model implementations must follow.
    It uses generics to specify input and output types.
    """

    @abstractmethod
    async def predict(self, inputs: InputT) -> OutputT:
        """Generate predictions for the given inputs.
        
        Args:
            inputs: The input data to generate predictions for.
            
        Returns:
            The model's predictions.
        """
        pass

    @abstractmethod
    async def batch_predict(self, inputs: List[InputT]) -> List[OutputT]:
        """Generate predictions for a batch of inputs.
        
        Args:
            inputs: A list of input data to generate predictions for.
            
        Returns:
            A list of the model's predictions.
        """
        pass


class BaseEvaluator(ABC, Generic[InputT, OutputT, TargetT]):
    """Abstract base class for evaluators.
    
    Evaluators are responsible for running the evaluation process,
    managing metrics, and producing evaluation results.
    """

    def __init__(self, metrics: List["BaseMetric"]):
        """Initialize the evaluator.
        
        Args:
            metrics: List of metrics to use for evaluation.
        """
        self.metrics = metrics

    @abstractmethod
    async def evaluate(
        self,
        model: BaseModel[InputT, OutputT],
        dataset: "BaseDataset[InputT, TargetT]",
    ) -> "EvalResult":
        """Run the evaluation process.
        
        Args:
            model: The model to evaluate.
            dataset: The dataset to evaluate on.
            
        Returns:
            The evaluation results.
        """
        pass


class BaseMetric(ABC, Generic[OutputT, TargetT, MetricValueT]):
    """Abstract base class for evaluation metrics.
    
    Metrics are responsible for computing specific evaluation scores
    based on model predictions and ground truth.
    """

    name: str = Field(..., description="Name of the metric")
    description: str = Field(..., description="Description of what the metric measures")

    @abstractmethod
    async def compute(
        self, predictions: OutputT, targets: TargetT
    ) -> MetricValueT:
        """Compute the metric value.
        
        Args:
            predictions: The model's predictions.
            targets: The ground truth targets.
            
        Returns:
            The computed metric value.
        """
        pass

    @abstractmethod
    async def batch_compute(
        self, predictions: List[OutputT], targets: List[TargetT]
    ) -> MetricValueT:
        """Compute the metric value for a batch of predictions.
        
        Args:
            predictions: List of model predictions.
            targets: List of ground truth targets.
            
        Returns:
            The computed metric value for the batch.
        """
        pass


class BaseDataset(ABC, Generic[InputT, TargetT]):
    """Abstract base class for datasets.
    
    Datasets are responsible for loading and providing access to
    evaluation data.
    """

    @abstractmethod
    async def load(self) -> None:
        """Load the dataset data."""
        pass

    @abstractmethod
    async def get_batch(
        self, batch_size: int, start_idx: Optional[int] = None
    ) -> tuple[List[InputT], List[TargetT]]:
        """Get a batch of data.
        
        Args:
            batch_size: Size of the batch to return.
            start_idx: Optional starting index for the batch.
            
        Returns:
            Tuple of (inputs, targets) for the batch.
        """
        pass

    @abstractmethod
    async def get_all(self) -> tuple[List[InputT], List[TargetT]]:
        """Get all data in the dataset.
        
        Returns:
            Tuple of (inputs, targets) for all data.
        """
        pass


class EvalResult(PydanticBaseModel):
    """Structure for evaluation results.
    
    This class provides a standardized way to store and access
    evaluation results, including individual metric scores and
    aggregated statistics.
    """

    model_name: str = Field(..., description="Name of the evaluated model")
    dataset_name: str = Field(..., description="Name of the dataset used")
    metrics: Dict[str, Any] = Field(
        ..., description="Dictionary of metric names to their computed values"
    )
    predictions: List[Any] = Field(
        default_factory=list,
        description="List of model predictions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the evaluation run",
    )
    timestamp: str = Field(..., description="Timestamp of the evaluation")

    def get_metric(self, metric_name: str) -> Any:
        """Get the value for a specific metric.
        
        Args:
            metric_name: Name of the metric to retrieve.
            
        Returns:
            The value of the requested metric.
            
        Raises:
            KeyError: If the metric name is not found.
        """
        return self.metrics[metric_name]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the results to a dictionary.
        
        Returns:
            Dictionary representation of the results.
        """
        return self.model_dump()

    def to_json(self) -> str:
        """Convert the results to a JSON string.
        
        Returns:
            JSON string representation of the results.
        """
        return self.model_dump_json() 