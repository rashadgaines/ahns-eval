"""Evaluation engine for running model evaluations."""

import asyncio
import logging
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
from eval_framework.core.config import EvaluationConfig
from eval_framework.core.results import ResultManager

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
    """Engine for running model evaluations.
    
    This class provides functionality for running evaluations, including
    single evaluations, batch processing, progress tracking, and error handling.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        result_manager: Optional[ResultManager] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the evaluation engine.
        
        Args:
            config: Evaluation configuration
            result_manager: Optional result manager for saving results
            max_retries: Maximum number of retries for failed evaluations
            retry_delay: Delay between retries in seconds
        """
        self.config = config
        self.result_manager = result_manager
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._executor = ThreadPoolExecutor(max_workers=config.num_workers)

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

    async def evaluate(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        evaluator: BaseEvaluator,
        save_results: bool = True,
    ) -> EvalResult:
        """Run a complete evaluation.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            evaluator: The evaluator to use
            save_results: Whether to save results
            
        Returns:
            The evaluation results
        """
        # Initialize progress tracking
        total_samples = len(await dataset.get_all()[0])
        progress = EvaluationProgress(
            total_samples=total_samples,
            total_batches=(total_samples + self.config.dataset.batch_size - 1)
            // self.config.dataset.batch_size,
        )

        # Create progress display
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress_bar:
            # Create progress task
            task = progress_bar.add_task(
                "[cyan]Evaluating...",
                total=progress.total_samples,
            )

            # Process batches
            batch_results = []
            for start_idx in range(0, total_samples, self.config.dataset.batch_size):
                result = await self._process_batch(
                    model,
                    dataset,
                    evaluator,
                    self.config.dataset.batch_size,
                    start_idx,
                    progress,
                )
                if result:
                    batch_results.append(result)
                
                # Update progress bar
                progress_bar.update(
                    task,
                    completed=progress.processed_samples,
                    description=f"[cyan]Evaluating... (Batch {progress.current_batch}/{progress.total_batches})",
                )

        # Combine batch results
        final_result = self._combine_results(batch_results)

        # Save results if requested
        if save_results and self.result_manager:
            self.result_manager.save_results(
                result=final_result,
                model_name=self.config.model.name,
                dataset_name=self.config.dataset.name,
                config=self.config.model_dump(),
                system_info=self._get_system_info(),
            )

        return final_result

    def _combine_results(self, results: List[EvalResult]) -> EvalResult:
        """Combine results from multiple batches.
        
        Args:
            results: List of batch results
            
        Returns:
            Combined evaluation results
        """
        if not results:
            raise ValueError("No results to combine")

        # Combine metrics
        combined_metrics = {}
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in combined_metrics:
                    combined_metrics[metric_name] = []
                combined_metrics[metric_name].append(value)

        # Average metrics
        final_metrics = {
            name: sum(values) / len(values)
            for name, values in combined_metrics.items()
        }

        # Create final result
        return EvalResult(
            model_name=results[0].model_name,
            dataset_name=results[0].dataset_name,
            metrics=final_metrics,
            metadata=results[0].metadata,
            timestamp=datetime.utcnow().isoformat(),
        )

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