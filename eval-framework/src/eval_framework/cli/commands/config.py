"""Configuration management commands."""

import logging
import os
from typing import Dict, Any, Optional
import click
import yaml
from pathlib import Path

from ...config import EvalConfig
from ..utils import validate_config

logger = logging.getLogger(__name__)

@click.group()
def config() -> None:
    """Manage evaluation configurations."""
    pass

@config.command()
@click.argument("template", type=click.Choice(["text", "image", "audio"]))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for configuration file"
)
def create(template: str, output: Optional[str]) -> None:
    """Create a new configuration file from template."""
    try:
        # Load template
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "templates",
            f"{template}_config.yaml"
        )
        
        if not os.path.exists(template_path):
            raise click.ClickException(f"Template not found: {template}")
        
        with open(template_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Set output path
        if output is None:
            output = f"{template}_config.yaml"
        
        # Write configuration
        with open(output, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        click.echo(f"Configuration file created: {output}")
        
    except Exception as e:
        logger.error(f"Failed to create configuration: {str(e)}")
        raise click.ClickException(str(e))

@config.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str) -> None:
    """Validate a configuration file."""
    try:
        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        validate_config(config)
        
        # Try to create EvalConfig
        eval_config = EvalConfig(**config)
        
        click.echo("Configuration is valid")
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise click.ClickException(str(e))

@config.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Output format"
)
def show(config_path: str, format: str) -> None:
    """Show configuration details."""
    try:
        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        validate_config(config)
        
        # Create EvalConfig
        eval_config = EvalConfig(**config)
        
        # Show configuration
        if format == "yaml":
            click.echo(yaml.dump(config, default_flow_style=False))
        else:
            click.echo(yaml.dump(config, default_flow_style=False))
        
    except Exception as e:
        logger.error(f"Failed to show configuration: {str(e)}")
        raise click.ClickException(str(e))

@config.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--key",
    "-k",
    required=True,
    help="Configuration key to update (e.g., 'model.batch_size')"
)
@click.option(
    "--value",
    "-v",
    required=True,
    help="New value"
)
def update(config_path: str, key: str, value: str) -> None:
    """Update a configuration value."""
    try:
        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Update value
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Convert value to appropriate type
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
        
        current[keys[-1]] = value
        
        # Validate updated configuration
        validate_config(config)
        
        # Write updated configuration
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        click.echo(f"Updated {key} to {value}")
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {str(e)}")
        raise click.ClickException(str(e)) 