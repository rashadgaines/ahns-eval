import pytest
import tempfile
import os
from pathlib import Path
from eval_framework.core.engine import EvaluationEngine
from eval_framework.models.text_model import TextModel
from eval_framework.datasets.text_dataset import TextDataset
from eval_framework.evaluators.exact_match import ExactMatchEvaluator
from eval_framework.metrics.rouge import ROUGEMetric

class TestEvaluationPipeline:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def config(self, temp_dir):
        """Create a test configuration."""
        return {
            "model": {
                "type": "text",
                "name": "gpt2",
                "batch_size": 4,
                "device": "cpu"
            },
            "dataset": {
                "type": "text",
                "name": "test_dataset",
                "split": "test",
                "max_samples": 10,
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

    def test_full_pipeline(self, engine, config, temp_dir):
        """Test the complete evaluation pipeline."""
        # Run evaluation
        results = engine.evaluate(config)
        
        # Verify results structure
        assert "metrics" in results
        assert "predictions" in results
        assert "error_analysis" in results
        
        # Verify output files
        assert (temp_dir / "predictions.json").exists()
        assert (temp_dir / "metrics.json").exists()
        assert (temp_dir / "error_analysis.json").exists()
        
        # Verify metrics
        assert "rouge1" in results["metrics"]
        assert "rouge2" in results["metrics"]
        assert "rougeL" in results["metrics"]
        assert all(0 <= score <= 1 for score in results["metrics"].values())

    def test_pipeline_with_invalid_config(self, engine):
        """Test pipeline with invalid configuration."""
        invalid_config = {
            "model": {
                "type": "nonexistent_type"
            }
        }
        
        with pytest.raises(ValueError):
            engine.evaluate(invalid_config)

    def test_pipeline_with_empty_dataset(self, engine, config):
        """Test pipeline with empty dataset."""
        config["dataset"]["max_samples"] = 0
        
        results = engine.evaluate(config)
        
        assert results["metrics"]["rouge1"] == 0.0
        assert len(results["predictions"]) == 0

    def test_pipeline_with_large_dataset(self, engine, config):
        """Test pipeline with large dataset."""
        config["dataset"]["max_samples"] = 1000
        config["model"]["batch_size"] = 32
        
        results = engine.evaluate(config)
        
        assert len(results["predictions"]) == 1000
        assert all(0 <= score <= 1 for score in results["metrics"].values())

    def test_pipeline_with_multiple_metrics(self, engine, config):
        """Test pipeline with multiple metrics."""
        config["metrics"].append({
            "name": "exact_match",
            "type": "exact_match",
            "normalize_text": True,
            "case_sensitive": False
        })
        
        results = engine.evaluate(config)
        
        assert "exact_match" in results["metrics"]
        assert "rouge1" in results["metrics"]
        assert all(0 <= score <= 1 for score in results["metrics"].values())

    def test_pipeline_error_recovery(self, engine, config):
        """Test pipeline error recovery."""
        # Simulate a model error
        config["model"]["name"] = "nonexistent_model"
        
        with pytest.raises(Exception):
            engine.evaluate(config)
        
        # Verify that temporary files are cleaned up
        assert not any(temp_dir.glob("*.tmp"))

    def test_pipeline_with_custom_evaluator(self, engine, config):
        """Test pipeline with custom evaluator."""
        class CustomEvaluator(ExactMatchEvaluator):
            def evaluate(self, predictions, references):
                results = super().evaluate(predictions, references)
                results["custom_score"] = 0.5
                return results
        
        config["evaluator"]["type"] = "custom"
        config["evaluator"]["class"] = CustomEvaluator
        
        results = engine.evaluate(config)
        
        assert "custom_score" in results["metrics"]
        assert results["metrics"]["custom_score"] == 0.5

    def test_pipeline_with_progress_callback(self, engine, config):
        """Test pipeline with progress callback."""
        progress_updates = []
        
        def progress_callback(current, total, message):
            progress_updates.append((current, total, message))
        
        results = engine.evaluate(config, progress_callback=progress_callback)
        
        assert len(progress_updates) > 0
        assert all(0 <= current <= total for current, total, _ in progress_updates)
        assert all(isinstance(message, str) for _, _, message in progress_updates)

    def test_pipeline_with_checkpointing(self, engine, config, temp_dir):
        """Test pipeline with checkpointing."""
        config["checkpoint_dir"] = str(temp_dir / "checkpoints")
        
        # Run evaluation with checkpointing
        results = engine.evaluate(config)
        
        # Verify checkpoint files
        checkpoint_dir = Path(config["checkpoint_dir"])
        assert checkpoint_dir.exists()
        assert any(checkpoint_dir.glob("*.ckpt"))
        
        # Verify results are the same as without checkpointing
        assert "metrics" in results
        assert "predictions" in results
        assert all(0 <= score <= 1 for score in results["metrics"].values()) 