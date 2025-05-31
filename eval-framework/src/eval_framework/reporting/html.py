"""HTML report generation."""

import logging
from typing import Any, Dict, List, Optional
import json
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
import plotly.graph_objects as go
import plotly.io as pio
from .base import BaseReporter, ReportConfig

logger = logging.getLogger(__name__)

class HTMLReporter(BaseReporter):
    """HTML report generation.
    
    This class generates interactive HTML reports with plots and tables.
    """
    
    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        **kwargs: Any
    ):
        """Initialize HTML reporter.
        
        Args:
            config: Report configuration
            **kwargs: Additional configuration parameters
        """
        super().__init__(config, **kwargs)
        
        # Initialize Jinja2 environment
        template_dir = self.config.html_template or os.path.join(
            os.path.dirname(__file__),
            "templates"
        )
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"])
        )
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate HTML report.
        
        Args:
            results: Evaluation results
            output_path: Output file path
            
        Returns:
            Path to generated report
        """
        # Prepare results
        report_data = self._prepare_results(results)
        
        # Convert plots to Plotly
        if "plots" in report_data["results"]:
            report_data["results"]["plots"] = self._convert_plots_to_plotly(
                report_data["results"]["plots"]
            )
        
        # Get output path
        output_path = self._get_output_path(output_path, ".html")
        self._ensure_output_dir(output_path)
        
        try:
            # Load template
            template = self.env.get_template("report.html")
            
            # Render template
            html_content = template.render(
                data=report_data,
                plots=report_data["results"].get("plots", {}),
                metrics=report_data["results"].get("metrics", {}),
                error_analysis=report_data["results"].get("error_analysis", {}),
                model_comparison=report_data["results"].get("model_comparison", {})
            )
            
            # Write to file
            with open(output_path, "w") as f:
                f.write(html_content)
            
            logger.info(f"Generated HTML report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {str(e)}")
            raise
    
    def _convert_plots_to_plotly(
        self,
        plots: Dict[str, Any]
    ) -> Dict[str, str]:
        """Convert matplotlib plots to Plotly.
        
        Args:
            plots: Dictionary of plot names to matplotlib figures
            
        Returns:
            Dictionary of plot names to Plotly JSON
        """
        plotly_plots = {}
        
        for name, fig in plots.items():
            try:
                # Convert to Plotly
                plotly_fig = go.Figure(fig)
                
                # Convert to JSON
                plotly_json = pio.to_json(plotly_fig)
                plotly_plots[name] = plotly_json
                
            except Exception as e:
                logger.warning(f"Failed to convert plot {name}: {str(e)}")
                continue
        
        return plotly_plots
    
    def _format_metrics_table(
        self,
        metrics: Dict[str, Any]
    ) -> str:
        """Format metrics as HTML table.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            HTML table string
        """
        html = ["<table class='metrics-table'>"]
        
        # Add header
        html.append("<thead><tr>")
        html.append("<th>Metric</th>")
        html.append("<th>Value</th>")
        html.append("</tr></thead>")
        
        # Add rows
        html.append("<tbody>")
        for metric, value in metrics.items():
            html.append("<tr>")
            html.append(f"<td>{metric}</td>")
            html.append(f"<td>{value:.4f}</td>")
            html.append("</tr>")
        html.append("</tbody>")
        
        html.append("</table>")
        return "\n".join(html)
    
    def _format_error_analysis(
        self,
        error_analysis: Dict[str, Any]
    ) -> str:
        """Format error analysis as HTML.
        
        Args:
            error_analysis: Error analysis results
            
        Returns:
            HTML string
        """
        html = ["<div class='error-analysis'>"]
        
        # Add error distribution
        if "error_distribution" in error_analysis:
            html.append("<h3>Error Distribution</h3>")
            html.append(self._format_metrics_table(
                error_analysis["error_distribution"]
            ))
        
        # Add error patterns
        if "error_patterns" in error_analysis:
            html.append("<h3>Error Patterns</h3>")
            html.append("<ul>")
            for pattern, count in error_analysis["error_patterns"].items():
                html.append(f"<li>{pattern}: {count}</li>")
            html.append("</ul>")
        
        # Add error clusters
        if "error_clusters" in error_analysis:
            html.append("<h3>Error Clusters</h3>")
            for cluster_id, cluster in error_analysis["error_clusters"].items():
                html.append(f"<h4>Cluster {cluster_id}</h4>")
                html.append("<ul>")
                for case in cluster["cases"]:
                    html.append(f"<li>{case}</li>")
                html.append("</ul>")
        
        html.append("</div>")
        return "\n".join(html)
    
    def _format_model_comparison(
        self,
        comparison: Dict[str, Any]
    ) -> str:
        """Format model comparison as HTML.
        
        Args:
            comparison: Model comparison results
            
        Returns:
            HTML string
        """
        html = ["<div class='model-comparison'>"]
        
        # Add performance differences
        if "differences" in comparison:
            html.append("<h3>Performance Differences</h3>")
            html.append(self._format_metrics_table(
                comparison["differences"]
            ))
        
        # Add statistical significance
        if "significance" in comparison:
            html.append("<h3>Statistical Significance</h3>")
            html.append("<table class='significance-table'>")
            html.append("<thead><tr>")
            html.append("<th>Metric</th>")
            html.append("<th>p-value</th>")
            html.append("<th>Significant</th>")
            html.append("</tr></thead>")
            html.append("<tbody>")
            
            for metric, stats in comparison["significance"].items():
                html.append("<tr>")
                html.append(f"<td>{metric}</td>")
                html.append(f"<td>{stats['p_value']:.4f}</td>")
                html.append(f"<td>{'Yes' if stats['significant'] else 'No'}</td>")
                html.append("</tr>")
            
            html.append("</tbody></table>")
        
        html.append("</div>")
        return "\n".join(html) 