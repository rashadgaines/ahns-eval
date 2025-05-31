"""PDF report generation."""

import logging
from typing import Any, Dict, List, Optional
import os
import subprocess
from .base import BaseReporter, ReportConfig
from .latex import LaTeXReporter
from .html import HTMLReporter

logger = logging.getLogger(__name__)

class PDFReporter(BaseReporter):
    """PDF report generation.
    
    This class generates PDF reports from LaTeX or HTML sources.
    """
    
    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        **kwargs: Any
    ):
        """Initialize PDF reporter.
        
        Args:
            config: Report configuration
            **kwargs: Additional configuration parameters
        """
        super().__init__(config, **kwargs)
        
        # Initialize source reporters
        self.latex_reporter = LaTeXReporter(config)
        self.html_reporter = HTMLReporter(config)
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None,
        source_format: str = "latex"
    ) -> str:
        """Generate PDF report.
        
        Args:
            results: Evaluation results
            output_path: Output file path
            source_format: Source format ("latex" or "html")
            
        Returns:
            Path to generated report
        """
        # Get output path
        output_path = self._get_output_path(output_path, ".pdf")
        self._ensure_output_dir(output_path)
        
        try:
            if source_format == "latex":
                return self._generate_from_latex(results, output_path)
            elif source_format == "html":
                return self._generate_from_html(results, output_path)
            else:
                raise ValueError(f"Unsupported source format: {source_format}")
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {str(e)}")
            raise
    
    def _generate_from_latex(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> str:
        """Generate PDF from LaTeX source.
        
        Args:
            results: Evaluation results
            output_path: Output file path
            
        Returns:
            Path to generated PDF
        """
        # Generate LaTeX source
        tex_path = self.latex_reporter.generate_report(results)
        
        try:
            # Run pdflatex
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-output-directory=" + os.path.dirname(output_path),
                    tex_path
                ],
                check=True,
                capture_output=True
            )
            
            # Clean up auxiliary files
            base_name = os.path.splitext(tex_path)[0]
            for ext in [".aux", ".log", ".out"]:
                aux_file = base_name + ext
                if os.path.exists(aux_file):
                    os.remove(aux_file)
            
            logger.info(f"Generated PDF from LaTeX: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"pdflatex failed: {e.stderr.decode()}")
            raise
    
    def _generate_from_html(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> str:
        """Generate PDF from HTML source.
        
        Args:
            results: Evaluation results
            output_path: Output file path
            
        Returns:
            Path to generated PDF
        """
        # Generate HTML source
        html_path = self.html_reporter.generate_report(results)
        
        try:
            # Run wkhtmltopdf
            subprocess.run(
                [
                    "wkhtmltopdf",
                    "--enable-local-file-access",
                    "--dpi", str(self.config.pdf_dpi),
                    html_path,
                    output_path
                ],
                check=True,
                capture_output=True
            )
            
            logger.info(f"Generated PDF from HTML: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"wkhtmltopdf failed: {e.stderr.decode()}")
            raise
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        if not self._is_command_available("pdflatex"):
            logger.warning("pdflatex not found. LaTeX to PDF conversion will not work.")
        
        if not self._is_command_available("wkhtmltopdf"):
            logger.warning("wkhtmltopdf not found. HTML to PDF conversion will not work.")
    
    def _is_command_available(self, command: str) -> bool:
        """Check if a command is available.
        
        Args:
            command: Command to check
            
        Returns:
            True if command is available, False otherwise
        """
        try:
            subprocess.run(
                ["which", command],
                check=True,
                capture_output=True
            )
            return True
        except subprocess.CalledProcessError:
            return False 