"""JSON report generation."""

import logging
from typing import Any, Dict, List, Optional
import json
import os
from .base import BaseReporter, ReportConfig

logger = logging.getLogger(__name__)

class JSONReporter(BaseReporter):
    """JSON report generation.
    
    This class generates structured JSON summaries of evaluation results.
    """
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate JSON report.
        
        Args:
            results: Evaluation results
            output_path: Output file path
            
        Returns:
            Path to generated report
        """
        # Prepare results
        report_data = self._prepare_results(results)
        
        # Get output path
        output_path = self._get_output_path(output_path, ".json")
        self._ensure_output_dir(output_path)
        
        try:
            # Write to file
            with open(output_path, "w") as f:
                json.dump(
                    report_data,
                    f,
                    indent=self.config.json_indent,
                    default=self._json_serializer
                )
            
            logger.info(f"Generated JSON report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {str(e)}")
            raise
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable object
        """
        # Handle numpy types
        if hasattr(obj, "tolist"):
            return obj.tolist()
        
        # Handle datetime
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        
        # Handle matplotlib figures
        if hasattr(obj, "savefig"):
            return {
                "type": "matplotlib.figure.Figure",
                "size": obj.get_size_inches().tolist()
            }
        
        # Handle other non-serializable objects
        return str(obj)
    
    def _prepare_metrics(
        self,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare metrics for JSON serialization.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Processed metrics
        """
        processed = {}
        
        for metric, value in metrics.items():
            # Handle nested metrics
            if isinstance(value, dict):
                processed[metric] = self._prepare_metrics(value)
            else:
                # Convert to float if possible
                try:
                    processed[metric] = float(value)
                except (TypeError, ValueError):
                    processed[metric] = value
        
        return processed
    
    def _prepare_plots(
        self,
        plots: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare plots for JSON serialization.
        
        Args:
            plots: Dictionary of plots
            
        Returns:
            Processed plots
        """
        processed = {}
        
        for name, fig in plots.items():
            # Store plot metadata
            processed[name] = {
                "type": type(fig).__name__,
                "size": fig.get_size_inches().tolist() if hasattr(fig, "get_size_inches") else None,
                "axes": len(fig.axes) if hasattr(fig, "axes") else None
            }
        
        return processed
    
    def _prepare_error_analysis(
        self,
        error_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare error analysis for JSON serialization.
        
        Args:
            error_analysis: Error analysis results
            
        Returns:
            Processed error analysis
        """
        processed = {}
        
        # Process error distribution
        if "error_distribution" in error_analysis:
            processed["error_distribution"] = self._prepare_metrics(
                error_analysis["error_distribution"]
            )
        
        # Process error patterns
        if "error_patterns" in error_analysis:
            processed["error_patterns"] = dict(error_analysis["error_patterns"])
        
        # Process error clusters
        if "error_clusters" in error_analysis:
            processed["error_clusters"] = {
                cluster_id: {
                    "size": len(cluster["cases"]),
                    "error_types": dict(cluster["error_types"]),
                    "representative_case": cluster["representative_case"]
                }
                for cluster_id, cluster in error_analysis["error_clusters"].items()
            }
        
        return processed
    
    def _prepare_model_comparison(
        self,
        comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare model comparison for JSON serialization.
        
        Args:
            comparison: Model comparison results
            
        Returns:
            Processed model comparison
        """
        processed = {}
        
        # Process differences
        if "differences" in comparison:
            processed["differences"] = self._prepare_metrics(
                comparison["differences"]
            )
        
        # Process significance
        if "significance" in comparison:
            processed["significance"] = {
                metric: {
                    "test": stats["test"],
                    "statistic": float(stats["statistic"]),
                    "p_value": float(stats["p_value"]),
                    "significant": bool(stats["significant"])
                }
                for metric, stats in comparison["significance"].items()
            }
        
        # Process error analysis
        if "error_analysis" in comparison:
            processed["error_analysis"] = self._prepare_error_analysis(
                comparison["error_analysis"]
            )
        
        return processed 