"""Model management commands."""

import logging
from typing import Dict, Any, List, Optional
import click
import yaml
from pathlib import Path
import os

from ...models import ModelRegistry
from ...config import EvalConfig
from ..utils import load_config, validate_config

logger = logging.getLogger(__name__)

@click.group()
def models() -> None:
    """Manage evaluation models."""
    pass

@models.command()
@click.option(
    "--type",
    "-t",
    type=str,
    help="Filter models by type"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format"
)
def list(type: Optional[str], format: str) -> None:
    """List available models."""
    try:
        registry = ModelRegistry()
        available_models = registry.list_models(type)
        
        if format == "table":
            # Print table header
            click.echo("\nAvailable Models:")
            click.echo("=" * 50)
            click.echo(f"{'Type':<15} {'Name':<30} {'Description':<40}")
            click.echo("-" * 50)
            
            # Print model rows
            for model_type, models in available_models.items():
                for model in models:
                    click.echo(
                        f"{model_type:<15} "
                        f"{model['name']:<30} "
                        f"{model['description']:<40}"
                    )
        
        elif format == "json":
            click.echo(yaml.dump(available_models, default_flow_style=False))
        
        else:  # yaml
            click.echo(yaml.dump(available_models, default_flow_style=False))
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise click.ClickException(str(e))

@models.command()
@click.argument("model_name")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Configuration file for model test"
)
@click.option(
    "--input",
    "-i",
    type=str,
    help="Test input (text, file path, or URL)"
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=1,
    help="Batch size for testing"
)
@click.option(
    "--device",
    "-d",
    type=str,
    help="Device to use (e.g., 'cuda:0', 'cpu')"
)
def test(
    model_name: str,
    config: Optional[str],
    input: Optional[str],
    batch_size: int,
    device: Optional[str]
) -> None:
    """Test a model with sample input."""
    try:
        registry = ModelRegistry()
        
        # Load model configuration
        if config:
            config_data = load_config(config)
            validate_config(config_data)
            model_config = config_data["model"]
        else:
            model_config = {
                "type": "text",  # Default type
                "name": model_name,
                "batch_size": batch_size
            }
        
        if device:
            model_config["device"] = device
        
        # Initialize model
        model = registry.get_model(model_config)
        
        # Prepare test input
        if input:
            if os.path.exists(input):
                with open(input, "r") as f:
                    test_input = f.read()
            else:
                test_input = input
        else:
            # Use default test input based on model type
            if model_config["type"] == "text":
                test_input = "This is a test input for the model."
            elif model_config["type"] == "image":
                test_input = "path/to/test/image.jpg"
            else:
                test_input = "path/to/test/audio.wav"
        
        # Run test
        logger.info(f"Testing model: {model_name}")
        logger.info(f"Input: {test_input}")
        
        result = model.predict(test_input)
        
        # Display results
        click.echo("\nTest Results:")
        click.echo("=" * 50)
        click.echo(f"Model: {model_name}")
        click.echo(f"Input: {test_input}")
        click.echo("\nOutput:")
        click.echo(yaml.dump(result, default_flow_style=False))
        
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
        raise click.ClickException(str(e))

@models.command()
@click.argument("model_name")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format"
)
def info(model_name: str, format: str) -> None:
    """Show detailed information about a model."""
    try:
        registry = ModelRegistry()
        model_info = registry.get_model_info(model_name)
        
        if format == "table":
            click.echo(f"\nModel Information: {model_name}")
            click.echo("=" * 50)
            
            # Display basic info
            click.echo("\nBasic Information:")
            click.echo(f"Type: {model_info['type']}")
            click.echo(f"Description: {model_info['description']}")
            
            # Display parameters
            if "parameters" in model_info:
                click.echo("\nParameters:")
                for param, value in model_info["parameters"].items():
                    click.echo(f"  {param}: {value}")
            
            # Display requirements
            if "requirements" in model_info:
                click.echo("\nRequirements:")
                for req in model_info["requirements"]:
                    click.echo(f"  - {req}")
        
        else:
            click.echo(yaml.dump(model_info, default_flow_style=False))
        
    except Exception as e:
        logger.error(f"Failed to get model information: {str(e)}")
        raise click.ClickException(str(e)) 