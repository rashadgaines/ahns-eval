"""Command-line interface for evaluation framework."""

import logging
import os
from typing import Dict, List, Optional, Any
import click
import yaml
from pathlib import Path
from datetime import datetime

from ..config import EvalConfig
from ..evaluator import Evaluator
from ..reporting import (
    ReportConfig,
    HTMLReporter,
    JSONReporter,
    LaTeXReporter,
    PDFReporter
)

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool) -> None:
    """Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise click.ClickException(f"Failed to load configuration: {str(e)}")

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        click.ClickException: If configuration is invalid
    """
    required_fields = ["model", "dataset", "metrics"]
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            raise click.ClickException(f"Missing required field: {field}")
    
    # Validate model configuration
    model_config = config["model"]
    if not isinstance(model_config, dict):
        raise click.ClickException("Model configuration must be a dictionary")
    
    if "type" not in model_config:
        raise click.ClickException("Model type not specified")
    
    # Validate dataset configuration
    dataset_config = config["dataset"]
    if not isinstance(dataset_config, dict):
        raise click.ClickException("Dataset configuration must be a dictionary")
    
    if "path" not in dataset_config:
        raise click.ClickException("Dataset path not specified")
    
    # Validate metrics configuration
    metrics_config = config["metrics"]
    if not isinstance(metrics_config, list):
        raise click.ClickException("Metrics configuration must be a list")
    
    if not metrics_config:
        raise click.ClickException("No metrics specified")

@click.group()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging"
)
def cli(verbose: bool) -> None:
    """Evaluation framework CLI."""
    setup_logging(verbose)

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="results",
    help="Output directory for results"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["html", "json", "latex", "pdf"]),
    multiple=True,
    default=["html"],
    help="Output format(s) for results"
)
def run(config_path: str, output_dir: str, format: List[str]) -> None:
    """Run evaluation with specified configuration."""
    try:
        # Load and validate configuration
        config = load_config(config_path)
        validate_config(config)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize evaluator
        eval_config = EvalConfig(**config)
        evaluator = Evaluator(eval_config)
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Generate reports
        report_config = ReportConfig(
            title=f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            output_dir=output_dir
        )
        
        for fmt in format:
            if fmt == "html":
                reporter = HTMLReporter(report_config)
            elif fmt == "json":
                reporter = JSONReporter(report_config)
            elif fmt == "latex":
                reporter = LaTeXReporter(report_config)
            elif fmt == "pdf":
                reporter = PDFReporter(report_config)
            
            reporter.generate_report(results)
        
        click.echo(f"Evaluation completed. Results saved to {output_dir}")
        
    except Exception as e:
        raise click.ClickException(str(e))

@cli.command()
@click.option(
    "--results-dir",
    "-r",
    type=click.Path(exists=True),
    default="results",
    help="Directory containing results"
)
def list(results_dir: str) -> None:
    """List available evaluation results."""
    try:
        results_path = Path(results_dir)
        if not results_path.exists():
            raise click.ClickException(f"Results directory not found: {results_dir}")
        
        # Find all result files
        result_files = []
        for ext in [".html", ".json", ".tex", ".pdf"]:
            result_files.extend(results_path.glob(f"*{ext}"))
        
        if not result_files:
            click.echo("No results found")
            return
        
        # Group results by timestamp
        results_by_time = {}
        for file in result_files:
            # Extract timestamp from filename
            timestamp = file.stem.split("_")[-1]
            if timestamp not in results_by_time:
                results_by_time[timestamp] = []
            results_by_time[timestamp].append(file)
        
        # Display results
        click.echo("\nAvailable Results:")
        click.echo("=" * 50)
        
        for timestamp, files in sorted(results_by_time.items(), reverse=True):
            click.echo(f"\nTimestamp: {timestamp}")
            for file in files:
                click.echo(f"  - {file.name}")
        
    except Exception as e:
        raise click.ClickException(str(e))

@cli.command()
@click.argument("result_path", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["html", "json", "latex", "pdf"]),
    default="html",
    help="Output format for results"
)
def results(result_path: str, format: str) -> None:
    """View evaluation results."""
    try:
        result_file = Path(result_path)
        if not result_file.exists():
            raise click.ClickException(f"Result file not found: {result_path}")
        
        # Load results based on format
        if result_file.suffix == ".json":
            with open(result_file, "r") as f:
                results = yaml.safe_load(f)
            
            # Display results
            click.echo("\nEvaluation Results:")
            click.echo("=" * 50)
            
            # Display metrics
            if "metrics" in results:
                click.echo("\nMetrics:")
                for metric, value in results["metrics"].items():
                    click.echo(f"  {metric}: {value:.4f}")
            
            # Display error analysis
            if "error_analysis" in results:
                click.echo("\nError Analysis:")
                for key, value in results["error_analysis"].items():
                    click.echo(f"  {key}: {value}")
            
            # Display model comparison
            if "model_comparison" in results:
                click.echo("\nModel Comparison:")
                for key, value in results["model_comparison"].items():
                    click.echo(f"  {key}: {value}")
        
        else:
            # For non-JSON files, just show the file path
            click.echo(f"Result file: {result_file}")
            click.echo("Use appropriate viewer to open the file")
        
    except Exception as e:
        raise click.ClickException(str(e))

if __name__ == "__main__":
    cli() 