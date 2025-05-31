import pytest
import tempfile
import os
from pathlib import Path
from eval_framework.core.engine import EvaluationEngine
from eval_framework.models.text_model import TextModel
from eval_framework.datasets.text_dataset import TextDataset

class TestLargeDatasetHandling:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def large_config(self, temp_dir):
        """Create a configuration for large dataset testing."""
        return {
            "model": {
                "type": "text",
                "name": "gpt2",
                "batch_size": 32,
                "device": "cpu"
            },
            "dataset": {
                "type": "text",
                "name": "large_test_dataset",
                "split": "test",
                "max_samples": 10000,
                "batch_size": 32
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

    def test_large_dataset_processing(self, engine, large_config):
        """Test processing a large dataset."""
        results = engine.evaluate(large_config)
        
        assert len(results["predictions"]) == large_config["dataset"]["max_samples"]
        assert "metrics" in results
        assert all(0 <= score <= 1 for score in results["metrics"].values())

    def test_large_dataset_memory_usage(self, engine, large_config):
        """Test memory usage with large dataset."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        results = engine.evaluate(large_config)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Verify memory usage is reasonable (less than 1GB)
        assert memory_increase < 1024 * 1024 * 1024

    def test_large_dataset_batch_processing(self, engine, large_config):
        """Test batch processing with large dataset."""
        batch_sizes = [16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            large_config["model"]["batch_size"] = batch_size
            large_config["dataset"]["batch_size"] = batch_size
            
            results = engine.evaluate(large_config)
            
            assert len(results["predictions"]) == large_config["dataset"]["max_samples"]
            assert all(0 <= score <= 1 for score in results["metrics"].values())

    def test_large_dataset_checkpointing(self, engine, large_config, temp_dir):
        """Test checkpointing with large dataset."""
        large_config["checkpoint_dir"] = str(temp_dir / "checkpoints")
        
        results = engine.evaluate(large_config)
        
        # Verify checkpoint files
        checkpoint_dir = Path(large_config["checkpoint_dir"])
        assert checkpoint_dir.exists()
        assert any(checkpoint_dir.glob("*.ckpt"))
        
        # Verify results
        assert len(results["predictions"]) == large_config["dataset"]["max_samples"]
        assert all(0 <= score <= 1 for score in results["metrics"].values())

    def test_large_dataset_resume(self, engine, large_config, temp_dir):
        """Test resuming evaluation with large dataset."""
        large_config["checkpoint_dir"] = str(temp_dir / "checkpoints")
        
        # Run first half of evaluation
        large_config["dataset"]["max_samples"] = 5000
        results1 = engine.evaluate(large_config)
        
        # Resume with full dataset
        large_config["dataset"]["max_samples"] = 10000
        results2 = engine.evaluate(large_config)
        
        # Verify results
        assert len(results2["predictions"]) == 10000
        assert all(0 <= score <= 1 for score in results2["metrics"].values())

    def test_large_dataset_progress_tracking(self, engine, large_config):
        """Test progress tracking with large dataset."""
        progress_updates = []
        
        def progress_callback(current, total, message):
            progress_updates.append((current, total, message))
        
        results = engine.evaluate(large_config, progress_callback=progress_callback)
        
        assert len(progress_updates) > 0
        assert all(0 <= current <= total for current, total, _ in progress_updates)
        assert all(isinstance(message, str) for _, _, message in progress_updates)
        
        # Verify progress updates are frequent enough
        assert len(progress_updates) >= large_config["dataset"]["max_samples"] / large_config["model"]["batch_size"]

    def test_large_dataset_error_recovery(self, engine, large_config):
        """Test error recovery with large dataset."""
        # Simulate an error halfway through
        large_config["dataset"]["max_samples"] = 10000
        large_config["error_sample_index"] = 5000
        
        with pytest.raises(Exception):
            engine.evaluate(large_config)
        
        # Verify that temporary files are cleaned up
        assert not any(temp_dir.glob("*.tmp"))

    def test_large_dataset_parallel_processing(self, engine, large_config):
        """Test parallel processing with large dataset."""
        large_config["num_workers"] = 4
        
        results = engine.evaluate(large_config)
        
        assert len(results["predictions"]) == large_config["dataset"]["max_samples"]
        assert all(0 <= score <= 1 for score in results["metrics"].values())

    def test_large_dataset_disk_usage(self, engine, large_config, temp_dir):
        """Test disk usage with large dataset."""
        initial_size = sum(f.stat().st_size for f in temp_dir.rglob("*"))
        
        results = engine.evaluate(large_config)
        
        final_size = sum(f.stat().st_size for f in temp_dir.rglob("*"))
        size_increase = final_size - initial_size
        
        # Verify disk usage is reasonable (less than 100MB)
        assert size_increase < 100 * 1024 * 1024 