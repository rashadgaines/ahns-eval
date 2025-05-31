"""Results management commands."""

import logging
from typing import Dict, Any, List, Optional
import click
import yaml
from pathlib import Path
from datetime import datetime
import os

from ...reporting import (
    ReportConfig,
    HTMLReporter,
    JSONReporter,
    LaTeXReporter,
    PDFReporter
)

logger = logging.getLogger(__name__)

@click.group()
def results() -> None:
    """Manage evaluation results."""
    pass

@results.command()
@click.option(
    "--results-dir",
    "-r",
    type=click.Path(exists=True),
    default="results",
    help="Directory containing results"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format"
)
def list(results_dir: str, format: str) -> None:
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
        
        if format == "table":
            # Display results
            click.echo("\nAvailable Results:")
            click.echo("=" * 50)
            
            for timestamp, files in sorted(results_by_time.items(), reverse=True):
                click.echo(f"\nTimestamp: {timestamp}")
                for file in files:
                    click.echo(f"  - {file.name}")
        
        else:
            # Convert to dictionary
            results_dict = {
                timestamp: [str(f) for f in files]
                for timestamp, files in results_by_time.items()
            }
            click.echo(yaml.dump(results_dict, default_flow_style=False))
        
    except Exception as e:
        logger.error(f"Failed to list results: {str(e)}")
        raise click.ClickException(str(e))

@results.command()
@click.argument("result_path", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["html", "json", "latex", "pdf"]),
    default="html",
    help="Output format for results"
)
def show(result_path: str, format: str) -> None:
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
        logger.error(f"Failed to show results: {str(e)}")
        raise click.ClickException(str(e))

@results.command()
@click.argument("result_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="results",
    help="Output directory for converted results"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["html", "json", "latex", "pdf"]),
    required=True,
    help="Target format for conversion"
)
def convert(result_path: str, output_dir: str, format: str) -> None:
    """Convert results to a different format."""
    try:
        result_file = Path(result_path)
        if not result_file.exists():
            raise click.ClickException(f"Result file not found: {result_path}")
        
        # Load results
        if result_file.suffix == ".json":
            with open(result_file, "r") as f:
                results = yaml.safe_load(f)
        else:
            raise click.ClickException("Only JSON results can be converted")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report
        report_config = ReportConfig(
            title=f"Converted Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            output_dir=output_dir
        )
        
        if format == "html":
            reporter = HTMLReporter(report_config)
        elif format == "json":
            reporter = JSONReporter(report_config)
        elif format == "latex":
            reporter = LaTeXReporter(report_config)
        elif format == "pdf":
            reporter = PDFReporter(report_config)
        
        output_path = reporter.generate_report(results)
        click.echo(f"Results converted to {format}. Output: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to convert results: {str(e)}")
        raise click.ClickException(str(e))

@results.command()
@click.argument("result_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for summary"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format for summary"
)
def summarize(result_path: str, output: Optional[str], format: str) -> None:
    """Generate a summary of evaluation results."""
    try:
        result_file = Path(result_path)
        if not result_file.exists():
            raise click.ClickException(f"Result file not found: {result_path}")
        
        # Load results
        if result_file.suffix == ".json":
            with open(result_file, "r") as f:
                results = yaml.safe_load(f)
        else:
            raise click.ClickException("Only JSON results can be summarized")
        
        # Generate summary
        summary = {
            "timestamp": results.get("metadata", {}).get("timestamp"),
            "metrics": results.get("metrics", {}),
            "error_summary": {
                "total_errors": len(results.get("error_analysis", {}).get("error_cases", [])),
                "error_types": results.get("error_analysis", {}).get("error_types", {})
            }
        }
        
        # Output summary
        if format == "text":
            output_text = [
                "\nEvaluation Summary:",
                "=" * 50,
                f"\nTimestamp: {summary['timestamp']}",
                "\nMetrics:",
            ]
            
            for metric, value in summary["metrics"].items():
                output_text.append(f"  {metric}: {value:.4f}")
            
            output_text.extend([
                "\nError Summary:",
                f"  Total Errors: {summary['error_summary']['total_errors']}",
                "\nError Types:"
            ])
            
            for error_type, count in summary["error_summary"]["error_types"].items():
                output_text.append(f"  {error_type}: {count}")
            
            output_text = "\n".join(output_text)
            
        else:
            output_text = yaml.dump(summary, default_flow_style=False)
        
        # Write output
        if output:
            with open(output, "w") as f:
                f.write(output_text)
            click.echo(f"Summary written to {output}")
        else:
            click.echo(output_text)
        
    except Exception as e:
        logger.error(f"Failed to summarize results: {str(e)}")
        raise click.ClickException(str(e)) 