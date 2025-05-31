import pytest
from eval_framework.core.base import BaseModel, BaseDataset, BaseEvaluator, BaseMetric

class TestBaseModel:
    def test_base_model_initialization(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()

    def test_base_model_abstract_methods(self):
        """Test that BaseModel has required abstract methods."""
        class DummyModel(BaseModel):
            pass

        with pytest.raises(TypeError):
            DummyModel()

        class ValidModel(BaseModel):
            def load(self):
                pass

            def predict(self, inputs):
                pass

            def get_info(self):
                return {"name": "test_model"}

        model = ValidModel()
        assert model.get_info()["name"] == "test_model"

class TestBaseDataset:
    def test_base_dataset_initialization(self):
        """Test that BaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDataset()

    def test_base_dataset_abstract_methods(self):
        """Test that BaseDataset has required abstract methods."""
        class DummyDataset(BaseDataset):
            pass

        with pytest.raises(TypeError):
            DummyDataset()

        class ValidDataset(BaseDataset):
            def load(self):
                pass

            def get_samples(self):
                return []

            def get_info(self):
                return {"name": "test_dataset"}

        dataset = ValidDataset()
        assert dataset.get_info()["name"] == "test_dataset"

class TestBaseEvaluator:
    def test_base_evaluator_initialization(self):
        """Test that BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator()

    def test_base_evaluator_abstract_methods(self):
        """Test that BaseEvaluator has required abstract methods."""
        class DummyEvaluator(BaseEvaluator):
            pass

        with pytest.raises(TypeError):
            DummyEvaluator()

        class ValidEvaluator(BaseEvaluator):
            def evaluate(self, predictions, references):
                return {"score": 0.0}

            def get_info(self):
                return {"name": "test_evaluator"}

        evaluator = ValidEvaluator()
        assert evaluator.get_info()["name"] == "test_evaluator"
        assert evaluator.evaluate([], [])["score"] == 0.0

class TestBaseMetric:
    def test_base_metric_initialization(self):
        """Test that BaseMetric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetric()

    def test_base_metric_abstract_methods(self):
        """Test that BaseMetric has required abstract methods."""
        class DummyMetric(BaseMetric):
            pass

        with pytest.raises(TypeError):
            DummyMetric()

        class ValidMetric(BaseMetric):
            def compute(self, predictions, references):
                return {"score": 0.0}

            def get_info(self):
                return {"name": "test_metric"}

        metric = ValidMetric()
        assert metric.get_info()["name"] == "test_metric"
        assert metric.compute([], [])["score"] == 0.0 