"""Core package for the evaluation framework."""

from eval_framework.core.base import (
    BaseDataset,
    BaseEvaluator,
    BaseMetric,
    BaseModel,
    EvalResult,
    InputT,
    OutputT,
    TargetT,
    MetricValueT,
)

__all__ = [
    "BaseModel",
    "BaseEvaluator",
    "BaseMetric",
    "BaseDataset",
    "EvalResult",
    "InputT",
    "OutputT",
    "TargetT",
    "MetricValueT",
]
