"""Visualization implementation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    # Style configuration
    style: str = "seaborn"
    context: str = "notebook"
    palette: str = "colorblind"
    
    # Figure configuration
    dpi: int = 100
    figsize: Tuple[int, int] = (10, 6)
    
    # Font configuration
    font_family: str = "sans-serif"
    font_size: int = 12
    
    # Color configuration
    colors: List[str] = None
    
    def __post_init__(self):
        """Set default colors if not provided."""
        if self.colors is None:
            self.colors = sns.color_palette(self.palette)

class Visualizer:
    """Visualization implementation.
    
    This class provides tools for creating various charts and plots to visualize
    evaluation results.
    """
    
    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        **kwargs: Any
    ):
        """Initialize visualizer.
        
        Args:
            config: Visualization configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or VisualizationConfig(**kwargs)
        
        # Set style
        plt.style.use(self.config.style)
        sns.set_context(self.config.context)
        sns.set_palette(self.config.palette)
        
        # Set font
        plt.rcParams["font.family"] = self.config.font_family
        plt.rcParams["font.size"] = self.config.font_size
    
    def plot_metrics(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Model Metrics",
        **kwargs: Any
    ) -> Figure:
        """Plot multiple metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            title: Plot title
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot each metric
        for i, (metric, values) in enumerate(metrics.items()):
            ax.plot(
                values,
                label=metric,
                color=self.config.colors[i % len(self.config.colors)],
                **kwargs
            )
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix",
        **kwargs: Any
    ) -> Figure:
        """Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            labels: Class labels
            title: Plot title
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot confusion matrix
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            **kwargs
        )
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        
        return fig
    
    def plot_error_distribution(
        self,
        errors: List[Dict[str, Any]],
        title: str = "Error Distribution",
        **kwargs: Any
    ) -> Figure:
        """Plot error distribution.
        
        Args:
            errors: List of error dictionaries
            title: Plot title
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib figure
        """
        # Extract error types and counts
        error_types = [e["error_type"] for e in errors]
        counts = Counter(error_types)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot bar chart
        bars = ax.bar(
            counts.keys(),
            counts.values(),
            color=self.config.colors[:len(counts)]
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f"{height}",
                ha="center",
                va="bottom"
            )
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Error Type")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        
        return fig
    
    def plot_metric_comparison(
        self,
        model1_metrics: Dict[str, float],
        model2_metrics: Dict[str, float],
        title: str = "Model Comparison",
        **kwargs: Any
    ) -> Figure:
        """Plot metric comparison between models.
        
        Args:
            model1_metrics: First model's metrics
            model2_metrics: Second model's metrics
            title: Plot title
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        metrics = list(model1_metrics.keys())
        model1_values = [model1_metrics[m] for m in metrics]
        model2_values = [model2_metrics[m] for m in metrics]
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Set width of bars
        bar_width = 0.35
        
        # Set positions of bars
        r1 = np.arange(len(metrics))
        r2 = [x + bar_width for x in r1]
        
        # Plot bars
        ax.bar(
            r1,
            model1_values,
            width=bar_width,
            label="Model 1",
            color=self.config.colors[0]
        )
        ax.bar(
            r2,
            model2_values,
            width=bar_width,
            label="Model 2",
            color=self.config.colors[1]
        )
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Score")
        ax.set_xticks([r + bar_width/2 for r in range(len(metrics))])
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        
        return fig
    
    def plot_confidence_intervals(
        self,
        intervals: Dict[str, Dict[str, float]],
        title: str = "Confidence Intervals",
        **kwargs: Any
    ) -> Figure:
        """Plot confidence intervals.
        
        Args:
            intervals: Dictionary of metric names to interval data
            title: Plot title
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        metrics = list(intervals.keys())
        means = [intervals[m]["mean"] for m in metrics]
        lower = [intervals[m]["lower"] for m in metrics]
        upper = [intervals[m]["upper"] for m in metrics]
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot confidence intervals
        ax.errorbar(
            metrics,
            means,
            yerr=[np.array(means) - np.array(lower), np.array(upper) - np.array(means)],
            fmt="o",
            capsize=5,
            color=self.config.colors[0],
            **kwargs
        )
        
        # Add zero line
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Difference")
        plt.xticks(rotation=45)
        
        return fig
    
    def plot_error_clusters(
        self,
        clusters: Dict[str, List[Dict[str, Any]]],
        title: str = "Error Clusters",
        **kwargs: Any
    ) -> Figure:
        """Plot error clusters.
        
        Args:
            clusters: Dictionary of cluster labels to error cases
            title: Plot title
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        cluster_sizes = [len(cases) for cases in clusters.values()]
        cluster_labels = list(clusters.keys())
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot pie chart
        ax.pie(
            cluster_sizes,
            labels=cluster_labels,
            autopct="%1.1f%%",
            colors=self.config.colors[:len(cluster_sizes)],
            **kwargs
        )
        
        # Customize plot
        ax.set_title(title)
        
        return fig
    
    def plot_learning_curves(
        self,
        train_scores: List[float],
        val_scores: List[float],
        title: str = "Learning Curves",
        **kwargs: Any
    ) -> Figure:
        """Plot learning curves.
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            title: Plot title
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib figure
        """
        # Create plot
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot curves
        x = range(1, len(train_scores) + 1)
        ax.plot(
            x,
            train_scores,
            label="Training",
            color=self.config.colors[0],
            **kwargs
        )
        ax.plot(
            x,
            val_scores,
            label="Validation",
            color=self.config.colors[1],
            **kwargs
        )
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def save_figure(
        self,
        fig: Figure,
        filename: str,
        **kwargs: Any
    ) -> None:
        """Save figure to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            **kwargs: Additional saving parameters
        """
        try:
            fig.savefig(
                filename,
                dpi=self.config.dpi,
                bbox_inches="tight",
                **kwargs
            )
            plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to save figure: {str(e)}")
            raise 