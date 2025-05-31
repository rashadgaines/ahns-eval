"""Model comparison implementation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ComparisonConfig:
    """Configuration for model comparison."""
    # Statistical test configuration
    significance_level: float = 0.05
    test_type: str = "wilcoxon"  # wilcoxon, ttest, mcnemar
    
    # Comparison metrics
    primary_metrics: List[str] = ("accuracy", "f1")
    secondary_metrics: List[str] = ("precision", "recall")
    
    # Analysis configuration
    min_samples: int = 30
    confidence_level: float = 0.95

class ModelComparer:
    """Model comparison implementation.
    
    This class provides tools for comparing model performance and conducting
    statistical significance tests.
    """
    
    def __init__(
        self,
        config: Optional[ComparisonConfig] = None,
        **kwargs: Any
    ):
        """Initialize model comparer.
        
        Args:
            config: Comparison configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or ComparisonConfig(**kwargs)
    
    def compare_models(
        self,
        model1_predictions: List[Any],
        model2_predictions: List[Any],
        references: List[Any],
        metrics: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Compare two models' performance.
        
        Args:
            model1_predictions: First model's predictions
            model2_predictions: Second model's predictions
            references: Ground truth references
            metrics: Evaluation metrics for both models
            
        Returns:
            Dictionary containing comparison results
        """
        # Compute performance differences
        differences = self._compute_differences(
            model1_predictions,
            model2_predictions,
            references,
            metrics
        )
        
        # Perform statistical tests
        significance = self._compute_significance(
            model1_predictions,
            model2_predictions,
            references,
            metrics
        )
        
        # Analyze error patterns
        error_analysis = self._analyze_errors(
            model1_predictions,
            model2_predictions,
            references
        )
        
        # Compute confidence intervals
        confidence_intervals = self._compute_confidence_intervals(
            differences,
            metrics
        )
        
        return {
            "differences": differences,
            "significance": significance,
            "error_analysis": error_analysis,
            "confidence_intervals": confidence_intervals
        }
    
    def _compute_differences(
        self,
        model1_predictions: List[Any],
        model2_predictions: List[Any],
        references: List[Any],
        metrics: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Compute performance differences between models.
        
        Args:
            model1_predictions: First model's predictions
            model2_predictions: Second model's predictions
            references: Ground truth references
            metrics: Evaluation metrics for both models
            
        Returns:
            Dictionary containing performance differences
        """
        differences = {}
        
        # Compute differences for each metric
        for metric in self.config.primary_metrics + self.config.secondary_metrics:
            if metric in metrics["model1"][0]:
                model1_scores = [m[metric] for m in metrics["model1"]]
                model2_scores = [m[metric] for m in metrics["model2"]]
                
                differences[metric] = {
                    "mean_difference": np.mean(model2_scores) - np.mean(model1_scores),
                    "std_difference": np.std(np.array(model2_scores) - np.array(model1_scores)),
                    "relative_improvement": (
                        (np.mean(model2_scores) - np.mean(model1_scores)) /
                        np.mean(model1_scores) * 100
                    )
                }
        
        return differences
    
    def _compute_significance(
        self,
        model1_predictions: List[Any],
        model2_predictions: List[Any],
        references: List[Any],
        metrics: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Compute statistical significance of differences.
        
        Args:
            model1_predictions: First model's predictions
            model2_predictions: Second model's predictions
            references: Ground truth references
            metrics: Evaluation metrics for both models
            
        Returns:
            Dictionary containing significance test results
        """
        significance = {}
        
        if self.config.test_type == "wilcoxon":
            # Compute signed differences for each metric
            for metric in self.config.primary_metrics:
                if metric in metrics["model1"][0]:
                    model1_scores = [m[metric] for m in metrics["model1"]]
                    model2_scores = [m[metric] for m in metrics["model2"]]
                    
                    statistic, p_value = stats.wilcoxon(
                        model2_scores,
                        model1_scores
                    )
                    
                    significance[metric] = {
                        "test": "wilcoxon",
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant": p_value < self.config.significance_level
                    }
        
        elif self.config.test_type == "ttest":
            # Compute t-test for each metric
            for metric in self.config.primary_metrics:
                if metric in metrics["model1"][0]:
                    model1_scores = [m[metric] for m in metrics["model1"]]
                    model2_scores = [m[metric] for m in metrics["model2"]]
                    
                    statistic, p_value = stats.ttest_rel(
                        model2_scores,
                        model1_scores
                    )
                    
                    significance[metric] = {
                        "test": "ttest",
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant": p_value < self.config.significance_level
                    }
        
        elif self.config.test_type == "mcnemar":
            # Compute McNemar's test
            b = sum(1 for r, p1, p2 in zip(references, model1_predictions, model2_predictions)
                   if p1 != r and p2 == r)
            c = sum(1 for r, p1, p2 in zip(references, model1_predictions, model2_predictions)
                   if p1 == r and p2 != r)
            
            statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
            p_value = 1 - stats.chi2.cdf(statistic, 1)
            
            significance["mcnemar"] = {
                "test": "mcnemar",
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < self.config.significance_level,
                "contingency_table": {
                    "b": b,  # model1 wrong, model2 right
                    "c": c   # model1 right, model2 wrong
                }
            }
        
        return significance
    
    def _analyze_errors(
        self,
        model1_predictions: List[Any],
        model2_predictions: List[Any],
        references: List[Any]
    ) -> Dict[str, Any]:
        """Analyze error patterns between models.
        
        Args:
            model1_predictions: First model's predictions
            model2_predictions: Second model's predictions
            references: Ground truth references
            
        Returns:
            Dictionary containing error analysis results
        """
        # Compute confusion matrices
        cm1 = confusion_matrix(references, model1_predictions)
        cm2 = confusion_matrix(references, model2_predictions)
        
        # Analyze error patterns
        error_patterns = defaultdict(int)
        for r, p1, p2 in zip(references, model1_predictions, model2_predictions):
            if p1 != r and p2 != r:
                error_patterns["both_wrong"] += 1
            elif p1 != r and p2 == r:
                error_patterns["model1_wrong"] += 1
            elif p1 == r and p2 != r:
                error_patterns["model2_wrong"] += 1
            else:
                error_patterns["both_right"] += 1
        
        # Compute error agreement
        total = len(references)
        error_agreement = {
            "both_wrong": error_patterns["both_wrong"] / total,
            "model1_wrong": error_patterns["model1_wrong"] / total,
            "model2_wrong": error_patterns["model2_wrong"] / total,
            "both_right": error_patterns["both_right"] / total
        }
        
        return {
            "confusion_matrices": {
                "model1": cm1.tolist(),
                "model2": cm2.tolist()
            },
            "error_patterns": dict(error_patterns),
            "error_agreement": error_agreement
        }
    
    def _compute_confidence_intervals(
        self,
        differences: Dict[str, Any],
        metrics: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Compute confidence intervals for performance differences.
        
        Args:
            differences: Performance differences
            metrics: Evaluation metrics for both models
            
        Returns:
            Dictionary containing confidence intervals
        """
        confidence_intervals = {}
        
        for metric in self.config.primary_metrics:
            if metric in differences:
                model1_scores = [m[metric] for m in metrics["model1"]]
                model2_scores = [m[metric] for m in metrics["model2"]]
                
                # Compute differences
                score_differences = np.array(model2_scores) - np.array(model1_scores)
                
                # Compute confidence interval
                ci = stats.t.interval(
                    self.config.confidence_level,
                    len(score_differences) - 1,
                    loc=np.mean(score_differences),
                    scale=stats.sem(score_differences)
                )
                
                confidence_intervals[metric] = {
                    "lower": ci[0],
                    "upper": ci[1],
                    "mean": np.mean(score_differences),
                    "std": np.std(score_differences)
                }
        
        return confidence_intervals 