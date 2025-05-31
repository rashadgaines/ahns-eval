"""LaTeX report generation."""

import logging
from typing import Any, Dict, List, Optional
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .base import BaseReporter, ReportConfig

logger = logging.getLogger(__name__)

class LaTeXReporter(BaseReporter):
    """LaTeX report generation.
    
    This class generates LaTeX tables and reports.
    """
    
    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        **kwargs: Any
    ):
        """Initialize LaTeX reporter.
        
        Args:
            config: Report configuration
            **kwargs: Additional configuration parameters
        """
        super().__init__(config, **kwargs)
        
        # Initialize Jinja2 environment
        template_dir = self.config.latex_template or os.path.join(
            os.path.dirname(__file__),
            "templates"
        )
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["tex"])
        )
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate LaTeX report.
        
        Args:
            results: Evaluation results
            output_path: Output file path
            
        Returns:
            Path to generated report
        """
        # Prepare results
        report_data = self._prepare_results(results)
        
        # Get output path
        output_path = self._get_output_path(output_path, ".tex")
        self._ensure_output_dir(output_path)
        
        try:
            # Load template
            template = self.env.get_template("report.tex")
            
            # Render template
            latex_content = template.render(
                data=report_data,
                metrics=self._format_metrics_table(report_data["results"].get("metrics", {})),
                error_analysis=self._format_error_analysis(report_data["results"].get("error_analysis", {})),
                model_comparison=self._format_model_comparison(report_data["results"].get("model_comparison", {}))
            )
            
            # Write to file
            with open(output_path, "w") as f:
                f.write(latex_content)
            
            logger.info(f"Generated LaTeX report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate LaTeX report: {str(e)}")
            raise
    
    def _format_metrics_table(
        self,
        metrics: Dict[str, Any]
    ) -> str:
        """Format metrics as LaTeX table.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            LaTeX table string
        """
        latex = []
        
        # Start table
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\begin{tabular}{lr}")
        latex.append("\\hline")
        latex.append("\\textbf{Metric} & \\textbf{Value} \\\\")
        latex.append("\\hline")
        
        # Add rows
        for metric, value in metrics.items():
            latex.append(f"{self._escape_latex(metric)} & {value:.4f} \\\\")
        
        # End table
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\caption{Metrics}")
        latex.append("\\label{tab:metrics}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def _format_error_analysis(
        self,
        error_analysis: Dict[str, Any]
    ) -> str:
        """Format error analysis as LaTeX.
        
        Args:
            error_analysis: Error analysis results
            
        Returns:
            LaTeX string
        """
        latex = []
        
        # Add error distribution
        if "error_distribution" in error_analysis:
            latex.append("\\subsection{Error Distribution}")
            latex.append(self._format_metrics_table(
                error_analysis["error_distribution"]
            ))
        
        # Add error patterns
        if "error_patterns" in error_analysis:
            latex.append("\\subsection{Error Patterns}")
            latex.append("\\begin{itemize}")
            for pattern, count in error_analysis["error_patterns"].items():
                latex.append(f"\\item {self._escape_latex(pattern)}: {count}")
            latex.append("\\end{itemize}")
        
        # Add error clusters
        if "error_clusters" in error_analysis:
            latex.append("\\subsection{Error Clusters}")
            for cluster_id, cluster in error_analysis["error_clusters"].items():
                latex.append(f"\\subsubsection{{Cluster {cluster_id}}}")
                latex.append("\\begin{itemize}")
                for case in cluster["cases"]:
                    latex.append(f"\\item {self._escape_latex(case)}")
                latex.append("\\end{itemize}")
        
        return "\n".join(latex)
    
    def _format_model_comparison(
        self,
        comparison: Dict[str, Any]
    ) -> str:
        """Format model comparison as LaTeX.
        
        Args:
            comparison: Model comparison results
            
        Returns:
            LaTeX string
        """
        latex = []
        
        # Add performance differences
        if "differences" in comparison:
            latex.append("\\subsection{Performance Differences}")
            latex.append(self._format_metrics_table(
                comparison["differences"]
            ))
        
        # Add statistical significance
        if "significance" in comparison:
            latex.append("\\subsection{Statistical Significance}")
            latex.append("\\begin{table}[h]")
            latex.append("\\centering")
            latex.append("\\begin{tabular}{lrrl}")
            latex.append("\\hline")
            latex.append("\\textbf{Metric} & \\textbf{p-value} & \\textbf{Significant} \\\\")
            latex.append("\\hline")
            
            for metric, stats in comparison["significance"].items():
                latex.append(
                    f"{self._escape_latex(metric)} & "
                    f"{stats['p_value']:.4f} & "
                    f"{'Yes' if stats['significant'] else 'No'} \\\\"
                )
            
            latex.append("\\hline")
            latex.append("\\end{tabular}")
            latex.append("\\caption{Statistical Significance}")
            latex.append("\\label{tab:significance}")
            latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters.
        
        Args:
            text: Text to escape
            
        Returns:
            Escaped text
        """
        special_chars = {
            "&": "\\&",
            "%": "\\%",
            "$": "\\$",
            "#": "\\#",
            "_": "\\_",
            "{": "\\{",
            "}": "\\}",
            "~": "\\textasciitilde{}",
            "^": "\\textasciicircum{}",
            "\\": "\\textbackslash{}",
            "<": "\\textless{}",
            ">": "\\textgreater{}"
        }
        
        for char, escape in special_chars.items():
            text = text.replace(char, escape)
        
        return text 