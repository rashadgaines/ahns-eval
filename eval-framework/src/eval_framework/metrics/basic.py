"""Basic evaluation metrics implementation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef
)

logger = logging.getLogger(__name__)

@dataclass
class MetricConfig:
    """Configuration for metric computation."""
    average: str = "weighted"  # weighted, macro, micro, binary
    zero_division: float = 0.0
    beta: float = 1.0  # F-beta score
    labels: Optional[List[Any]] = None
    sample_weight: Optional[List[float]] = None
    significance_level: float = 0.05

class BasicMetrics:
    """Core evaluation metrics implementation.
    
    This class provides implementations of basic evaluation metrics including
    accuracy, precision/recall/F1, confusion matrix, and statistical significance
    tests.
    """
    
    def __init__(
        self,
        config: Optional[MetricConfig] = None,
        **kwargs: Any
    ):
        """Initialize metrics calculator.
        
        Args:
            config: Metric configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or MetricConfig(**kwargs)
    
    def compute_accuracy(
        self,
        y_true: List[Any],
        y_pred: List[Any]
    ) -> float:
        """Compute accuracy score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score
        """
        return accuracy_score(
            y_true,
            y_pred,
            sample_weight=self.config.sample_weight
        )
    
    def compute_precision_recall_f1(
        self,
        y_true: List[Any],
        y_pred: List[Any]
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute precision, recall, and F1 scores.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing precision, recall, and F1 scores
        """
        # Compute scores
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=self.config.average,
            beta=self.config.beta,
            zero_division=self.config.zero_division,
            labels=self.config.labels,
            sample_weight=self.config.sample_weight
        )
        
        # Get per-class scores if not binary
        if self.config.average != "binary":
            per_class = precision_recall_fscore_support(
                y_true,
                y_pred,
                average=None,
                beta=self.config.beta,
                zero_division=self.config.zero_division,
                labels=self.config.labels,
                sample_weight=self.config.sample_weight
            )
            
            per_class_scores = {
                label: {
                    "precision": p,
                    "recall": r,
                    "f1": f,
                    "support": s
                }
                for label, p, r, f, s in zip(
                    self.config.labels or sorted(set(y_true)),
                    *per_class
                )
            }
        else:
            per_class_scores = None
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "per_class": per_class_scores
        }
    
    def compute_confusion_matrix(
        self,
        y_true: List[Any],
        y_pred: List[Any]
    ) -> Dict[str, Any]:
        """Compute confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing confusion matrix and related metrics
        """
        # Compute confusion matrix
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=self.config.labels,
            sample_weight=self.config.sample_weight
        )
        
        # Get labels
        labels = self.config.labels or sorted(set(y_true))
        
        # Compute per-class metrics
        per_class = {}
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            per_class[label] = {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            }
        
        return {
            "matrix": cm,
            "labels": labels,
            "per_class": per_class
        }
    
    def compute_matthews_correlation(
        self,
        y_true: List[Any],
        y_pred: List[Any]
    ) -> float:
        """Compute Matthews correlation coefficient.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Matthews correlation coefficient
        """
        return matthews_corrcoef(
            y_true,
            y_pred,
            sample_weight=self.config.sample_weight
        )
    
    def compute_statistical_significance(
        self,
        y_true: List[Any],
        y_pred1: List[Any],
        y_pred2: List[Any],
        test: str = "mcnemar"
    ) -> Dict[str, Any]:
        """Compute statistical significance between two predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred1: First set of predictions
            y_pred2: Second set of predictions
            test: Statistical test to use (mcnemar, wilcoxon, ttest)
            
        Returns:
            Dictionary containing test results
        """
        if test == "mcnemar":
            # Compute contingency table
            b = sum(1 for t, p1, p2 in zip(y_true, y_pred1, y_pred2)
                   if p1 != t and p2 == t)
            c = sum(1 for t, p1, p2 in zip(y_true, y_pred1, y_pred2)
                   if p1 == t and p2 != t)
            
            # Compute chi-square statistic
            chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
            p_value = 1 - stats.chi2.cdf(chi2, 1)
            
        elif test == "wilcoxon":
            # Compute signed differences
            diff1 = [1 if p == t else 0 for p, t in zip(y_pred1, y_true)]
            diff2 = [1 if p == t else 0 for p, t in zip(y_pred2, y_true)]
            differences = [d1 - d2 for d1, d2 in zip(diff1, diff2)]
            
            # Compute test statistic
            statistic, p_value = stats.wilcoxon(differences)
            
        elif test == "ttest":
            # Compute accuracy differences
            diff1 = [1 if p == t else 0 for p, t in zip(y_pred1, y_true)]
            diff2 = [1 if p == t else 0 for p, t in zip(y_pred2, y_true)]
            
            # Compute test statistic
            statistic, p_value = stats.ttest_rel(diff1, diff2)
            
        else:
            raise ValueError(f"Unknown statistical test: {test}")
        
        # Determine significance
        is_significant = p_value < self.config.significance_level
        
        return {
            "test": test,
            "statistic": statistic if test != "mcnemar" else chi2,
            "p_value": p_value,
            "significant": is_significant,
            "significance_level": self.config.significance_level
        }
    
    def compute_all_metrics(
        self,
        y_true: List[Any],
        y_pred: List[Any]
    ) -> Dict[str, Any]:
        """Compute all metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing all metrics
        """
        return {
            "accuracy": self.compute_accuracy(y_true, y_pred),
            "precision_recall_f1": self.compute_precision_recall_f1(y_true, y_pred),
            "confusion_matrix": self.compute_confusion_matrix(y_true, y_pred),
            "matthews_correlation": self.compute_matthews_correlation(y_true, y_pred)
        }
    
    def compare_predictions(
        self,
        y_true: List[Any],
        y_pred1: List[Any],
        y_pred2: List[Any]
    ) -> Dict[str, Any]:
        """Compare two sets of predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred1: First set of predictions
            y_pred2: Second set of predictions
            
        Returns:
            Dictionary containing comparison results
        """
        # Compute metrics for both predictions
        metrics1 = self.compute_all_metrics(y_true, y_pred1)
        metrics2 = self.compute_all_metrics(y_true, y_pred2)
        
        # Compute statistical significance
        significance = {
            test: self.compute_statistical_significance(
                y_true,
                y_pred1,
                y_pred2,
                test=test
            )
            for test in ["mcnemar", "wilcoxon", "ttest"]
        }
        
        return {
            "metrics1": metrics1,
            "metrics2": metrics2,
            "significance": significance,
            "improvement": {
                metric: metrics2[metric] - metrics1[metric]
                for metric in ["accuracy", "matthews_correlation"]
            }
        } 