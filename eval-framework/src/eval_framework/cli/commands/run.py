"""Run command implementation."""

import logging
import os
from typing import Dict, List, Any, Optional
import click
from datetime import datetime

from ...config import EvalConfig
from ...evaluator import Evaluator
from ...reporting import (
    ReportConfig,
    HTMLReporter,
    JSONReporter,
    LaTeXReporter,
    PDFReporter
)
from ..utils import load_config, validate_config
from ..progress import ProgressTracker

logger = logging.getLogger(__name__)

@click.command()
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
@click.option(
    "--batch-size",
    "-b",
    type=int,
    help="Override batch size from config"
)
@click.option(
    "--device",
    "-d",
    type=str,
    help="Override device from config (e.g., 'cuda:0', 'cpu')"
)
@click.option(
    "--seed",
    "-s",
    type=int,
    help="Override random seed from config"
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from last checkpoint if available"
)
def run(
    config_path: str,
    output_dir: str,
    format: List[str],
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    resume: bool = False
) -> None:
    """Run evaluation with specified configuration."""
    try:
        # Load and validate configuration
        config = load_config(config_path)
        validate_config(config)
        
        # Override config with command line options
        if batch_size is not None:
            config["model"]["batch_size"] = batch_size
        if device is not None:
            config["model"]["device"] = device
        if seed is not None:
            config["seed"] = seed
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize evaluator
        eval_config = EvalConfig(**config)
        evaluator = Evaluator(eval_config)
        
        # Initialize progress tracker
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        if resume:
            tracker = ProgressTracker.load_checkpoint(checkpoint_dir)
            if tracker:
                click.echo("Resuming from checkpoint...")
            else:
                tracker = ProgressTracker(
                    total_steps=len(evaluator.dataset),
                    description="Evaluation",
                    checkpoint_dir=checkpoint_dir
                )
        else:
            tracker = ProgressTracker(
                total_steps=len(evaluator.dataset),
                description="Evaluation",
                checkpoint_dir=checkpoint_dir
            )
        
        # Start progress tracking
        tracker.start()
        
        # Run evaluation with progress tracking
        results = evaluator.evaluate(
            progress_callback=lambda step, desc, error: tracker.update(step, desc, error)
        )
        
        # Finish progress tracking
        tracker.finish()
        
        # Generate reports
        report_config = ReportConfig(
            title=f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            output_dir=output_dir
        )
        
        for fmt in format:
            logger.info(f"Generating {fmt} report...")
            if fmt == "html":
                reporter = HTMLReporter(report_config)
            elif fmt == "json":
                reporter = JSONReporter(report_config)
            elif fmt == "latex":
                reporter = LaTeXReporter(report_config)
            elif fmt == "pdf":
                reporter = PDFReporter(report_config)
            
            reporter.generate_report(results)
            logger.info(f"{fmt} report generated")
        
        click.echo(f"Evaluation completed. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise click.ClickException(str(e)) 