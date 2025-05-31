"""Structured logging utilities."""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import structlog
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    json_logs: bool = False,
    rich_console: bool = True,
) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional path to log file
        json_logs: Whether to use JSON format for logs
        rich_console: Whether to use rich console output
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    if rich_console:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if json_logs:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )
            file_handler.setFormatter(
                logging.Formatter("%(message)s")  # JSON logs are already formatted
            )
        else:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class LoggerAdapter:
    """Adapter for adding context to log messages."""

    def __init__(self, logger: structlog.BoundLogger):
        """Initialize the adapter.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.context: Dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> "LoggerAdapter":
        """Add context to the logger.
        
        Args:
            **kwargs: Context key-value pairs
            
        Returns:
            New logger adapter with added context
        """
        new_adapter = LoggerAdapter(self.logger)
        new_adapter.context = {**self.context, **kwargs}
        return new_adapter

    def _log(
        self, level: str, msg: str, *args: Any, **kwargs: Any
    ) -> None:
        """Log a message with context.
        
        Args:
            level: Log level
            msg: Log message
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        log_func = getattr(self.logger, level)
        log_func(msg, *args, **{**self.context, **kwargs})

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message.
        
        Args:
            msg: Log message
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        self._log("debug", msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message.
        
        Args:
            msg: Log message
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        self._log("info", msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message.
        
        Args:
            msg: Log message
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        self._log("warning", msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message.
        
        Args:
            msg: Log message
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        self._log("error", msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message.
        
        Args:
            msg: Log message
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        self._log("critical", msg, *args, **kwargs)


def log_execution_time(logger: structlog.BoundLogger):
    """Decorator for logging function execution time.
    
    Args:
        logger: Logger instance
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            result = func(*args, **kwargs)
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(
                "Function execution completed",
                function=func.__name__,
                duration_seconds=duration,
            )
            
            return result
        return wrapper
    return decorator


def log_exceptions(logger: structlog.BoundLogger):
    """Decorator for logging exceptions.
    
    Args:
        logger: Logger instance
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    "Exception occurred",
                    function=func.__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise
        return wrapper
    return decorator 