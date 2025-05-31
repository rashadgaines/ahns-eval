import pytest
import tempfile
from pathlib import Path
from eval_framework.core.engine import EvaluationEngine
from eval_framework.models.text_model import TextModel
from eval_framework.evaluators.exact_match import ExactMatchEvaluator
from eval_framework.metrics.rouge import ROUGEMetric

class TestMultiModelComparison:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def base_config(self, temp_dir):
        """Create a base test configuration."""
        return {
            "dataset": {
                "type": "text",
                "name": "test_dataset",
                "split": "test",
                "max_samples": 100,
                "batch_size": 4
            },
            "evaluator": {
                "type": "exact_match",
                "normalize_text": True,
                "case_sensitive": False
            },
            "metrics": [
                {
                    "name": "rouge",
                    "type": "rouge",
                    "metrics": ["rouge1", "rouge2", "rougeL"],
                    "use_stemmer": True
                }
            ],
            "output": {
                "formats": ["json"],
                "save_predictions": True,
                "save_metrics": True,
                "save_error_analysis": True,
                "output_dir": str(temp_dir)
            }
        }

    @pytest.fixture
    def engine(self):
        """Create an evaluation engine instance."""
        return EvaluationEngine()

    def test_compare_models(self, engine, base_config, temp_dir):
        """Test comparing multiple models."""
        models = [
            {
                "type": "text",
                "name": "gpt2",
                "batch_size": 4,
                "device": "cpu"
            },
            {
                "type": "text",
                "name": "gpt2-medium",
                "batch_size": 4,
                "device": "cpu"
            }
        ]
        
        results = engine.compare_models(models, base_config)
        
        # Verify results structure
        assert len(results) == len(models)
        for model_results in results:
            assert "metrics" in model_results
            assert "predictions" in model_results
            assert "error_analysis" in model_results
        
        # Verify output files
        for i, model in enumerate(models):
            model_dir = temp_dir / model["name"]
            assert model_dir.exists()
            assert (model_dir / "predictions.json").exists()
            assert (model_dir / "metrics.json").exists()
            assert (model_dir / "error_analysis.json").exists()

    def test_compare_models_with_different_batch_sizes(self, engine, base_config):
        """Test comparing models with different batch sizes."""
        models = [
            {
                "type": "text",
                "name": "gpt2",
                "batch_size": 4,
                "device": "cpu"
            },
            {
                "type": "text",
                "name": "gpt2-medium",
                "batch_size": 8,
                "device": "cpu"
            }
        ]
        
        results = engine.compare_models(models, base_config)
        
        assert len(results) == len(models)
        for model_results in results:
            assert "metrics" in model_results
            assert all(0 <= score <= 1 for score in model_results["metrics"].values())

    def test_compare_models_with_different_metrics(self, engine, base_config):
        """Test comparing models with different metrics."""
        models = [
            {
                "type": "text",
                "name": "gpt2",
                "batch_size": 4,
                "device": "cpu"
            },
            {
                "type": "text",
                "name": "gpt2-medium",
                "batch_size": 4,
                "device": "cpu"
            }
        ]
        
        # Add different metrics for each model
        base_config["metrics"] = [
            {
                "name": "rouge",
                "type": "rouge",
                "metrics": ["rouge1"],
                "use_stemmer": True
            }
        ]
        
        results = engine.compare_models(models, base_config)
        
        assert len(results) == len(models)
        for model_results in results:
            assert "rouge1" in model_results["metrics"]
            assert "rouge2" not in model_results["metrics"]
            assert "rougeL" not in model_results["metrics"]

    def test_compare_models_with_error(self, engine, base_config):
        """Test comparing models when one fails."""
        models = [
            {
                "type": "text",
                "name": "gpt2",
                "batch_size": 4,
                "device": "cpu"
            },
            {
                "type": "text",
                "name": "nonexistent_model",
                "batch_size": 4,
                "device": "cpu"
            }
        ]
        
        with pytest.raises(Exception):
            engine.compare_models(models, base_config)

    def test_compare_models_with_progress(self, engine, base_config):
        """Test comparing models with progress tracking."""
        models = [
            {
                "type": "text",
                "name": "gpt2",
                "batch_size": 4,
                "device": "cpu"
            },
            {
                "type": "text",
                "name": "gpt2-medium",
                "batch_size": 4,
                "device": "cpu"
            }
        ]
        
        progress_updates = []
        
        def progress_callback(current, total, message):
            progress_updates.append((current, total, message))
        
        results = engine.compare_models(
            models,
            base_config,
            progress_callback=progress_callback
        )
        
        assert len(progress_updates) > 0
        assert all(0 <= current <= total for current, total, _ in progress_updates)
        assert all(isinstance(message, str) for _, _, message in progress_updates)

    def test_compare_models_with_checkpointing(self, engine, base_config, temp_dir):
        """Test comparing models with checkpointing."""
        models = [
            {
                "type": "text",
                "name": "gpt2",
                "batch_size": 4,
                "device": "cpu"
            },
            {
                "type": "text",
                "name": "gpt2-medium",
                "batch_size": 4,
                "device": "cpu"
            }
        ]
        
        base_config["checkpoint_dir"] = str(temp_dir / "checkpoints")
        
        results = engine.compare_models(models, base_config)
        
        # Verify checkpoint files
        checkpoint_dir = Path(base_config["checkpoint_dir"])
        assert checkpoint_dir.exists()
        assert any(checkpoint_dir.glob("*.ckpt"))
        
        # Verify results
        assert len(results) == len(models)
        for model_results in results:
            assert "metrics" in model_results
            assert all(0 <= score <= 1 for score in model_results["metrics"].values())

    def test_compare_models_with_custom_evaluator(self, engine, base_config):
        """Test comparing models with custom evaluator."""
        models = [
            {
                "type": "text",
                "name": "gpt2",
                "batch_size": 4,
                "device": "cpu"
            },
            {
                "type": "text",
                "name": "gpt2-medium",
                "batch_size": 4,
                "device": "cpu"
            }
        ]
        
        class CustomEvaluator(ExactMatchEvaluator):
            def evaluate(self, predictions, references):
                results = super().evaluate(predictions, references)
                results["custom_score"] = 0.5
                return results
        
        base_config["evaluator"]["type"] = "custom"
        base_config["evaluator"]["class"] = CustomEvaluator
        
        results = engine.compare_models(models, base_config)
        
        assert len(results) == len(models)
        for model_results in results:
            assert "custom_score" in model_results["metrics"]
            assert model_results["metrics"]["custom_score"] == 0.5 