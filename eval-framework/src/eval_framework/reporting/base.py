"""Base reporting implementation."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ReportConfig:
    """Configuration for report generation."""
    # General configuration
    title: str = "Evaluation Report"
    description: str = ""
    timestamp: bool = True
    
    # Output configuration
    output_dir: str = "reports"
    filename_prefix: str = "eval_report"
    
    # Content configuration
    include_metrics: bool = True
    include_plots: bool = True
    include_error_analysis: bool = True
    include_model_comparison: bool = True
    
    # Format-specific configuration
    html_template: Optional[str] = None
    latex_template: Optional[str] = None
    json_indent: int = 2
    pdf_dpi: int = 300

class BaseReporter(ABC):
    """Base reporter implementation.
    
    This class defines the interface for all report generators.
    """
    
    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        **kwargs: Any
    ):
        """Initialize reporter.
        
        Args:
            config: Report configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or ReportConfig(**kwargs)
    
    @abstractmethod
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate report.
        
        Args:
            results: Evaluation results
            output_path: Output file path
            
        Returns:
            Path to generated report
        """
        pass
    
    def _prepare_results(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare results for reporting.
        
        Args:
            results: Raw evaluation results
            
        Returns:
            Processed results
        """
        # Add metadata
        report_data = {
            "metadata": {
                "title": self.config.title,
                "description": self.config.description,
                "timestamp": datetime.now().isoformat() if self.config.timestamp else None
            },
            "results": results
        }
        
        # Filter content based on configuration
        if not self.config.include_metrics:
            report_data["results"].pop("metrics", None)
        
        if not self.config.include_plots:
            report_data["results"].pop("plots", None)
        
        if not self.config.include_error_analysis:
            report_data["results"].pop("error_analysis", None)
        
        if not self.config.include_model_comparison:
            report_data["results"].pop("model_comparison", None)
        
        return report_data
    
    def _ensure_output_dir(self, output_path: str) -> None:
        """Ensure output directory exists.
        
        Args:
            output_path: Output file path
        """
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _get_output_path(
        self,
        output_path: Optional[str] = None,
        extension: str = ""
    ) -> str:
        """Get output file path.
        
        Args:
            output_path: Custom output path
            extension: File extension
            
        Returns:
            Output file path
        """
        if output_path:
            return output_path
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.filename_prefix}_{timestamp}{extension}"
        return os.path.join(self.config.output_dir, filename) 