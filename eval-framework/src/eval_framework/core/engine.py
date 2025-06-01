"""Evaluation engine for running model evaluations."""

import asyncio
import logging
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type

import psutil
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from eval_framework.core.base import (
    BaseDataset,
    BaseEvaluator,
    BaseMetric,
    BaseModel,
    EvalResult,
)
from eval_framework.core.config import EvaluationConfig, ModelConfig, DatasetConfig, MetricConfig
from eval_framework.core.results import ResultManager
from eval_framework.core.registry import get_model_class, get_dataset_class, get_evaluator_class

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class EvaluationProgress:
    """Progress tracking for evaluation runs.
    
    Attributes:
        total_samples: Total number of samples to evaluate
        processed_samples: Number of samples processed
        current_batch: Current batch number
        total_batches: Total number of batches
        metrics: Current metric values
        errors: List of errors encountered
    """

    total_samples: int
    processed_samples: int = 0
    current_batch: int = 0
    total_batches: int = 0
    metrics: Dict[str, float] = None
    errors: List[Tuple[str, Exception]] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.metrics is None:
            self.metrics = {}
        if self.errors is None:
            self.errors = []

    def update(
        self,
        processed: int,
        batch: int,
        metrics: Optional[Dict[str, float]] = None,
        error: Optional[Tuple[str, Exception]] = None,
    ) -> None:
        """Update progress information.
        
        Args:
            processed: Number of samples processed
            batch: Current batch number
            metrics: Optional updated metric values
            error: Optional error information
        """
        self.processed_samples = processed
        self.current_batch = batch
        if metrics:
            self.metrics.update(metrics)
        if error:
            self.errors.append(error)


class EvaluationEngine:
    """Evaluation engine for running model evaluations.
    
    This class provides functionality for:
    - Loading and validating evaluation configuration
    - Running evaluations with specified models and datasets
    - Collecting and aggregating evaluation results
    - Handling errors and retries
    """

    def __init__(self, config: EvaluationConfig):
        """Initialize evaluation engine.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self._validate_config()
        self.result_manager = None
        self.max_retries = 3
        self.retry_delay = 1.0
        self._executor = ThreadPoolExecutor(max_workers=config.num_workers)

    def _validate_config(self) -> None:
        """Validate evaluation configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, EvaluationConfig):
            raise ValueError("Config must be an EvaluationConfig instance")
        
        if not self.config.model:
            raise ValueError("Model configuration is required")
        
        if not self.config.dataset:
            raise ValueError("Dataset configuration is required")
        
        if not self.config.metrics:
            raise ValueError("At least one metric is required")

    async def evaluate(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        evaluator: BaseEvaluator
    ) -> Any:
        """Run evaluation.
        
        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
            evaluator: Evaluator to use
            
        Returns:
            Evaluation results
            
        Raises:
            ValueError: If evaluation fails
        """
        try:
            # Get total samples
            items, _ = await dataset.get_all()
            total_samples = len(items)
            
            # Run evaluation
            results = await evaluator.evaluate(model, dataset)
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise ValueError(f"Evaluation failed: {str(e)}")

    def _get_model_class(self, config: ModelConfig) -> Type[BaseModel]:
        """Get model class from registry.
        
        Args:
            config: Model configuration
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model class not found
        """
        model_class = get_model_class(config.type)
        if not model_class:
            raise ValueError(f"Model type '{config.type}' not found in registry")
        return model_class

    def _get_dataset_class(self, config: DatasetConfig) -> Type[BaseDataset]:
        """Get dataset class from registry.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Dataset class
            
        Raises:
            ValueError: If dataset class not found
        """
        dataset_class = get_dataset_class(config.type)
        if not dataset_class:
            raise ValueError(f"Dataset type '{config.type}' not found in registry")
        return dataset_class

    def _get_evaluator_class(self, config: MetricConfig) -> Type[BaseEvaluator]:
        """Get evaluator class from registry.
        
        Args:
            config: Metric configuration
            
        Returns:
            Evaluator class
            
        Raises:
            ValueError: If evaluator class not found
        """
        evaluator_class = get_evaluator_class(config.type)
        if not evaluator_class:
            raise ValueError(f"Evaluator type '{config.type}' not found in registry")
        return evaluator_class

    async def _run_single_evaluation(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        evaluator: BaseEvaluator,
        progress: EvaluationProgress,
    ) -> EvalResult:
        """Run a single evaluation.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            evaluator: The evaluator to use
            progress: Progress tracking object
            
        Returns:
            The evaluation results
            
        Raises:
            Exception: If evaluation fails after all retries
        """
        for attempt in range(self.max_retries):
            try:
                result = await evaluator.evaluate(model, dataset)
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(
                    f"Evaluation attempt {attempt + 1} failed: {str(e)}. "
                    f"Retrying in {self.retry_delay} seconds..."
                )
                await asyncio.sleep(self.retry_delay)

    async def _process_batch(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        evaluator: BaseEvaluator,
        batch_size: int,
        start_idx: int,
        progress: EvaluationProgress,
    ) -> Optional[EvalResult]:
        """Process a single batch of data.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            evaluator: The evaluator to use
            batch_size: Size of the batch
            start_idx: Starting index for the batch
            progress: Progress tracking object
            
        Returns:
            Optional evaluation results for the batch
        """
        try:
            # Get batch data
            inputs, targets = await dataset.get_batch(batch_size, start_idx)
            
            # Run evaluation
            result = await self._run_single_evaluation(
                model, dataset, evaluator, progress
            )
            
            # Update progress
            progress.update(
                processed=start_idx + len(inputs),
                batch=start_idx // batch_size + 1,
                metrics=result.metrics,
            )
            
            return result
        except Exception as e:
            error_msg = f"Error processing batch starting at {start_idx}"
            logger.error(f"{error_msg}: {str(e)}")
            progress.update(
                processed=start_idx,
                batch=start_idx // batch_size + 1,
                error=(error_msg, e),
            )
            return None

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information.
        
        Returns:
            Dictionary containing system information
        """
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
        }

    async def evaluate_batch(
        self,
        models: List[BaseModel],
        datasets: List[BaseDataset],
        evaluator: BaseEvaluator,
        save_results: bool = True,
    ) -> List[EvalResult]:
        """Run evaluations for multiple model-dataset pairs.
        
        Args:
            models: List of models to evaluate
            datasets: List of datasets to evaluate on
            evaluator: The evaluator to use
            save_results: Whether to save results
            
        Returns:
            List of evaluation results
            
        Raises:
            ValueError: If models and datasets lists have different lengths
        """
        if len(models) != len(datasets):
            raise ValueError("Number of models must match number of datasets")

        results = []
        for model, dataset in zip(models, datasets):
            result = await self.evaluate(
                model, dataset, evaluator, save_results=save_results
            )
            results.append(result)

        return results

    def __del__(self):
        """Clean up resources."""
        self._executor.shutdown(wait=False) 