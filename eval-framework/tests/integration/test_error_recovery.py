import pytest
import tempfile
import os
from pathlib import Path
from eval_framework.core.engine import EvaluationEngine
from eval_framework.models.text_model import TextModel
from eval_framework.datasets.text_dataset import TextDataset

class TestErrorRecovery:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def base_config(self, temp_dir):
        """Create a base test configuration."""
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
                    "metrics": ["rouge1"],
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

    def test_model_load_error(self, engine, base_config):
        """Test recovery from model loading error."""
        base_config["model"]["name"] = "nonexistent_model"
        
        with pytest.raises(Exception) as exc_info:
            engine.evaluate(base_config)
        
        assert "Model loading failed" in str(exc_info.value)
        assert not any(temp_dir.glob("*.tmp"))

    def test_dataset_load_error(self, engine, base_config):
        """Test recovery from dataset loading error."""
        base_config["dataset"]["name"] = "nonexistent_dataset"
        
        with pytest.raises(Exception) as exc_info:
            engine.evaluate(base_config)
        
        assert "Dataset loading failed" in str(exc_info.value)
        assert not any(temp_dir.glob("*.tmp"))

    def test_prediction_error(self, engine, base_config):
        """Test recovery from prediction error."""
        base_config["model"]["error_sample_index"] = 50
        
        with pytest.raises(Exception) as exc_info:
            engine.evaluate(base_config)
        
        assert "Prediction failed" in str(exc_info.value)
        assert not any(temp_dir.glob("*.tmp"))

    def test_evaluation_error(self, engine, base_config):
        """Test recovery from evaluation error."""
        base_config["evaluator"]["type"] = "nonexistent_evaluator"
        
        with pytest.raises(Exception) as exc_info:
            engine.evaluate(base_config)
        
        assert "Evaluation failed" in str(exc_info.value)
        assert not any(temp_dir.glob("*.tmp"))

    def test_metric_computation_error(self, engine, base_config):
        """Test recovery from metric computation error."""
        base_config["metrics"][0]["type"] = "nonexistent_metric"
        
        with pytest.raises(Exception) as exc_info:
            engine.evaluate(base_config)
        
        assert "Metric computation failed" in str(exc_info.value)
        assert not any(temp_dir.glob("*.tmp"))

    def test_checkpoint_error(self, engine, base_config, temp_dir):
        """Test recovery from checkpoint error."""
        base_config["checkpoint_dir"] = str(temp_dir / "checkpoints")
        os.chmod(temp_dir, 0o444)  # Make directory read-only
        
        with pytest.raises(Exception) as exc_info:
            engine.evaluate(base_config)
        
        assert "Checkpoint failed" in str(exc_info.value)
        assert not any(temp_dir.glob("*.tmp"))
        
        os.chmod(temp_dir, 0o755)  # Restore permissions

    def test_output_error(self, engine, base_config, temp_dir):
        """Test recovery from output error."""
        os.chmod(temp_dir, 0o444)  # Make directory read-only
        
        with pytest.raises(Exception) as exc_info:
            engine.evaluate(base_config)
        
        assert "Output failed" in str(exc_info.value)
        assert not any(temp_dir.glob("*.tmp"))
        
        os.chmod(temp_dir, 0o755)  # Restore permissions

    def test_memory_error(self, engine, base_config):
        """Test recovery from memory error."""
        base_config["dataset"]["max_samples"] = 1000000  # Very large dataset
        
        with pytest.raises(MemoryError) as exc_info:
            engine.evaluate(base_config)
        
        assert "Memory error" in str(exc_info.value)
        assert not any(temp_dir.glob("*.tmp"))

    def test_network_error(self, engine, base_config):
        """Test recovery from network error."""
        base_config["model"]["simulate_network_error"] = True
        
        with pytest.raises(Exception) as exc_info:
            engine.evaluate(base_config)
        
        assert "Network error" in str(exc_info.value)
        assert not any(temp_dir.glob("*.tmp"))

    def test_corrupted_checkpoint(self, engine, base_config, temp_dir):
        """Test recovery from corrupted checkpoint."""
        base_config["checkpoint_dir"] = str(temp_dir / "checkpoints")
        
        # Create a corrupted checkpoint file
        checkpoint_dir = Path(base_config["checkpoint_dir"])
        checkpoint_dir.mkdir(exist_ok=True)
        with open(checkpoint_dir / "corrupted.ckpt", "w") as f:
            f.write("corrupted data")
        
        with pytest.raises(Exception) as exc_info:
            engine.evaluate(base_config)
        
        assert "Checkpoint corrupted" in str(exc_info.value)
        assert not any(temp_dir.glob("*.tmp"))

    def test_concurrent_access(self, engine, base_config, temp_dir):
        """Test recovery from concurrent access error."""
        base_config["checkpoint_dir"] = str(temp_dir / "checkpoints")
        
        # Simulate concurrent access
        base_config["simulate_concurrent_access"] = True
        
        with pytest.raises(Exception) as exc_info:
            engine.evaluate(base_config)
        
        assert "Concurrent access" in str(exc_info.value)
        assert not any(temp_dir.glob("*.tmp"))

    def test_error_logging(self, engine, base_config, temp_dir):
        """Test error logging functionality."""
        base_config["model"]["name"] = "nonexistent_model"
        base_config["logging"] = {
            "level": "ERROR",
            "save_logs": True,
            "log_dir": str(temp_dir / "logs")
        }
        
        with pytest.raises(Exception):
            engine.evaluate(base_config)
        
        # Verify error logs
        log_dir = Path(base_config["logging"]["log_dir"])
        assert log_dir.exists()
        assert any(log_dir.glob("*.log"))
        
        # Verify error details in logs
        log_file = next(log_dir.glob("*.log"))
        with open(log_file) as f:
            log_content = f.read()
            assert "Model loading failed" in log_content
            assert "nonexistent_model" in log_content 