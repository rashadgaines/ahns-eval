import pytest
from eval_framework.core.registry import ModelRegistry, DatasetRegistry, EvaluatorRegistry, MetricRegistry

class TestModelRegistry:
    def test_model_registry_singleton(self):
        """Test that ModelRegistry is a singleton."""
        registry1 = ModelRegistry()
        registry2 = ModelRegistry()
        assert registry1 is registry2

    def test_register_model(self):
        """Test registering a model."""
        registry = ModelRegistry()
        
        @registry.register("test_model")
        class TestModel:
            pass

        assert "test_model" in registry.get_available_models()
        assert registry.get_model_class("test_model") == TestModel

    def test_register_duplicate_model(self):
        """Test registering a duplicate model name."""
        registry = ModelRegistry()
        
        @registry.register("test_model")
        class TestModel1:
            pass

        with pytest.raises(ValueError):
            @registry.register("test_model")
            class TestModel2:
                pass

    def test_get_nonexistent_model(self):
        """Test getting a nonexistent model."""
        registry = ModelRegistry()
        with pytest.raises(KeyError):
            registry.get_model_class("nonexistent_model")

class TestDatasetRegistry:
    def test_dataset_registry_singleton(self):
        """Test that DatasetRegistry is a singleton."""
        registry1 = DatasetRegistry()
        registry2 = DatasetRegistry()
        assert registry1 is registry2

    def test_register_dataset(self):
        """Test registering a dataset."""
        registry = DatasetRegistry()
        
        @registry.register("test_dataset")
        class TestDataset:
            pass

        assert "test_dataset" in registry.get_available_datasets()
        assert registry.get_dataset_class("test_dataset") == TestDataset

    def test_register_duplicate_dataset(self):
        """Test registering a duplicate dataset name."""
        registry = DatasetRegistry()
        
        @registry.register("test_dataset")
        class TestDataset1:
            pass

        with pytest.raises(ValueError):
            @registry.register("test_dataset")
            class TestDataset2:
                pass

    def test_get_nonexistent_dataset(self):
        """Test getting a nonexistent dataset."""
        registry = DatasetRegistry()
        with pytest.raises(KeyError):
            registry.get_dataset_class("nonexistent_dataset")

class TestEvaluatorRegistry:
    def test_evaluator_registry_singleton(self):
        """Test that EvaluatorRegistry is a singleton."""
        registry1 = EvaluatorRegistry()
        registry2 = EvaluatorRegistry()
        assert registry1 is registry2

    def test_register_evaluator(self):
        """Test registering an evaluator."""
        registry = EvaluatorRegistry()
        
        @registry.register("test_evaluator")
        class TestEvaluator:
            pass

        assert "test_evaluator" in registry.get_available_evaluators()
        assert registry.get_evaluator_class("test_evaluator") == TestEvaluator

    def test_register_duplicate_evaluator(self):
        """Test registering a duplicate evaluator name."""
        registry = EvaluatorRegistry()
        
        @registry.register("test_evaluator")
        class TestEvaluator1:
            pass

        with pytest.raises(ValueError):
            @registry.register("test_evaluator")
            class TestEvaluator2:
                pass

    def test_get_nonexistent_evaluator(self):
        """Test getting a nonexistent evaluator."""
        registry = EvaluatorRegistry()
        with pytest.raises(KeyError):
            registry.get_evaluator_class("nonexistent_evaluator")

class TestMetricRegistry:
    def test_metric_registry_singleton(self):
        """Test that MetricRegistry is a singleton."""
        registry1 = MetricRegistry()
        registry2 = MetricRegistry()
        assert registry1 is registry2

    def test_register_metric(self):
        """Test registering a metric."""
        registry = MetricRegistry()
        
        @registry.register("test_metric")
        class TestMetric:
            pass

        assert "test_metric" in registry.get_available_metrics()
        assert registry.get_metric_class("test_metric") == TestMetric

    def test_register_duplicate_metric(self):
        """Test registering a duplicate metric name."""
        registry = MetricRegistry()
        
        @registry.register("test_metric")
        class TestMetric1:
            pass

        with pytest.raises(ValueError):
            @registry.register("test_metric")
            class TestMetric2:
                pass

    def test_get_nonexistent_metric(self):
        """Test getting a nonexistent metric."""
        registry = MetricRegistry()
        with pytest.raises(KeyError):
            registry.get_metric_class("nonexistent_metric") 