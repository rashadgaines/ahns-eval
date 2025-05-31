"""Plugin registry system for the evaluation framework."""

import importlib
import inspect
import logging
from abc import ABC
from typing import Any, Callable, Dict, Optional, Set, Type, TypeVar, Union, get_type_hints, List, Generic

from pkg_resources import iter_entry_points

from eval_framework.core.base import (
    BaseDataset,
    BaseEvaluator,
    BaseMetric,
    BaseModel,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=ABC)


class Registry(Generic[T]):
    """Base registry class for managing plugins.
    
    This class provides the core functionality for registering and retrieving
    plugins. It supports both manual registration and auto-discovery via
    entry points.
    
    Attributes:
        _registry: Dictionary mapping names to plugin classes
        _base_class: The base class that all plugins must inherit from
    """

    def __init__(self, base_class: Type[T], entry_point_group: str):
        """Initialize the registry.
        
        Args:
            base_class: The base class that all plugins must inherit from
            entry_point_group: The entry point group to discover plugins from
        """
        self._registry: Dict[str, Type[T]] = {}
        self._base_class = base_class
        self._entry_point_group = entry_point_group
        self._discover_plugins()

    def _discover_plugins(self) -> None:
        """Discover plugins via entry points."""
        for entry_point in iter_entry_points(self._entry_point_group):
            try:
                plugin_class = entry_point.load()
                self.register(plugin_class, name=entry_point.name)
            except Exception as e:
                logger.warning(
                    f"Failed to load plugin {entry_point.name}: {str(e)}"
                )

    def register(
        self, plugin_class: Type[T], name: Optional[str] = None
    ) -> Type[T]:
        """Register a plugin class.
        
        Args:
            plugin_class: The plugin class to register
            name: Optional name to register the plugin under
            
        Returns:
            The registered plugin class
            
        Raises:
            ValueError: If the plugin class is invalid
            KeyError: If a plugin with the given name is already registered
        """
        if not inspect.isclass(plugin_class):
            raise ValueError(f"Plugin must be a class: {plugin_class}")
        
        if not issubclass(plugin_class, self._base_class):
            raise ValueError(
                f"Plugin must inherit from {self._base_class.__name__}: {plugin_class}"
            )
        
        if name is None:
            name = plugin_class.__name__
        
        if name in self._registry:
            raise KeyError(f"Plugin already registered with name: {name}")
        
        self._registry[name] = plugin_class
        logger.debug(f"Registered plugin {name}: {plugin_class}")
        return plugin_class

    def get(self, name: str) -> Type[T]:
        """Get a registered plugin class.
        
        Args:
            name: Name of the plugin to retrieve
            
        Returns:
            The registered plugin class
            
        Raises:
            KeyError: If no plugin is registered with the given name
        """
        if name not in self._registry:
            raise KeyError(f"No plugin registered with name: {name}")
        return self._registry[name]

    def list_available(self) -> Set[str]:
        """List all available plugin names.
        
        Returns:
            Set of available plugin names
        """
        return set(self._registry.keys())

    def create(self, name: str, **kwargs: Any) -> T:
        """Create an instance of a registered plugin.
        
        Args:
            name: Name of the plugin to instantiate
            **kwargs: Arguments to pass to the plugin constructor
            
        Returns:
            An instance of the plugin
            
        Raises:
            KeyError: If no plugin is registered with the given name
        """
        plugin_class = self.get(name)
        return plugin_class(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._registry


class ModelRegistry(Registry[BaseModel]):
    """Registry for model plugins."""

    def __init__(self):
        """Initialize the model registry."""
        super().__init__(BaseModel, "eval_framework.models")


class EvaluatorRegistry(Registry[BaseEvaluator]):
    """Registry for evaluator plugins."""

    def __init__(self):
        """Initialize the evaluator registry."""
        super().__init__(BaseEvaluator, "eval_framework.evaluators")


class MetricRegistry(Registry[BaseMetric]):
    """Registry for metric plugins."""

    def __init__(self):
        """Initialize the metric registry."""
        super().__init__(BaseMetric, "eval_framework.metrics")


class DatasetRegistry(Registry[BaseDataset]):
    """Registry for dataset plugins."""

    def __init__(self):
        """Initialize the dataset registry."""
        super().__init__(BaseDataset, "eval_framework.datasets")


# Global registry instances
model_registry = ModelRegistry()
evaluator_registry = EvaluatorRegistry()
metric_registry = MetricRegistry()
dataset_registry = DatasetRegistry()


def register_model(name: Optional[str] = None) -> Type[BaseModel]:
    """Decorator to register a model plugin.
    
    Args:
        name: Optional name to register the model under
        
    Returns:
        Decorator function
    """

    def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
        model_registry.register(model_class, name=name)
        return model_class

    return decorator


def register_evaluator(name: Optional[str] = None) -> Type[BaseEvaluator]:
    """Decorator to register an evaluator plugin.
    
    Args:
        name: Optional name to register the evaluator under
        
    Returns:
        Decorator function
    """

    def decorator(evaluator_class: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
        evaluator_registry.register(evaluator_class, name=name)
        return evaluator_class

    return decorator


def register_metric(name: Optional[str] = None) -> Type[BaseMetric]:
    """Decorator to register a metric plugin.
    
    Args:
        name: Optional name to register the metric under
        
    Returns:
        Decorator function
    """

    def decorator(metric_class: Type[BaseMetric]) -> Type[BaseMetric]:
        metric_registry.register(metric_class, name=name)
        return metric_class

    return decorator


def register_dataset(name: Optional[str] = None) -> Type[BaseDataset]:
    """Decorator to register a dataset plugin.
    
    Args:
        name: Optional name to register the dataset under
        
    Returns:
        Decorator function
    """

    def decorator(dataset_class: Type[BaseDataset]) -> Type[BaseDataset]:
        dataset_registry.register(dataset_class, name=name)
        return dataset_class

    return decorator 