import pytest
import time
import psutil
import os
import asyncio
import tempfile
from pathlib import Path
from eval_framework.core.engine import EvaluationEngine
from eval_framework.models.text_model import TextModel
from eval_framework.datasets.text_dataset import TextDataset

class TestBenchmarks:
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
                "max_samples": 1000,
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

    def test_throughput_measurement(self, engine, base_config):
        """Test throughput measurement for different batch sizes."""
        batch_sizes = [1, 4, 8, 16, 32]
        results = {}
        
        for batch_size in batch_sizes:
            base_config["model"]["batch_size"] = batch_size
            base_config["dataset"]["batch_size"] = batch_size
            
            start_time = time.time()
            engine.evaluate(base_config)
            end_time = time.time()
            
            total_samples = base_config["dataset"]["max_samples"]
            duration = end_time - start_time
            samples_per_second = total_samples / duration
            
            results[batch_size] = {
                "duration": duration,
                "samples_per_second": samples_per_second
            }
        
        # Verify throughput increases with batch size
        for i in range(1, len(batch_sizes)):
            assert results[batch_sizes[i]]["samples_per_second"] >= results[batch_sizes[i-1]]["samples_per_second"]

    def test_memory_usage_profiling(self, engine, base_config):
        """Test memory usage profiling during evaluation."""
        process = psutil.Process(os.getpid())
        memory_samples = []
        
        def memory_callback():
            memory_samples.append(process.memory_info().rss)
        
        # Run evaluation with memory sampling
        start_memory = process.memory_info().rss
        engine.evaluate(base_config, memory_callback=memory_callback)
        end_memory = process.memory_info().rss
        
        # Calculate memory statistics
        max_memory = max(memory_samples)
        avg_memory = sum(memory_samples) / len(memory_samples)
        memory_increase = end_memory - start_memory
        
        # Verify memory usage is reasonable
        assert max_memory < 1024 * 1024 * 1024  # Less than 1GB
        assert memory_increase < 512 * 1024 * 1024  # Less than 512MB increase
        assert len(memory_samples) > 0  # Memory was sampled

    def test_scalability(self, engine, base_config):
        """Test scalability with increasing dataset sizes."""
        dataset_sizes = [100, 1000, 10000]
        results = {}
        
        for size in dataset_sizes:
            base_config["dataset"]["max_samples"] = size
            
            start_time = time.time()
            engine.evaluate(base_config)
            end_time = time.time()
            
            duration = end_time - start_time
            samples_per_second = size / duration
            
            results[size] = {
                "duration": duration,
                "samples_per_second": samples_per_second
            }
        
        # Verify linear scaling (with some tolerance)
        for i in range(1, len(dataset_sizes)):
            ratio = results[dataset_sizes[i]]["duration"] / results[dataset_sizes[i-1]]["duration"]
            expected_ratio = dataset_sizes[i] / dataset_sizes[i-1]
            assert 0.8 * expected_ratio <= ratio <= 1.2 * expected_ratio

    @pytest.mark.asyncio
    async def test_async_performance(self, engine, base_config):
        """Test async performance with concurrent evaluations."""
        num_concurrent = 4
        tasks = []
        
        async def run_evaluation():
            return await asyncio.to_thread(engine.evaluate, base_config)
        
        # Run multiple evaluations concurrently
        start_time = time.time()
        for _ in range(num_concurrent):
            tasks.append(asyncio.create_task(run_evaluation()))
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_duration = end_time - start_time
        total_samples = base_config["dataset"]["max_samples"] * num_concurrent
        samples_per_second = total_samples / total_duration
        
        # Verify all evaluations completed successfully
        assert len(results) == num_concurrent
        for result in results:
            assert "metrics" in result
            assert "predictions" in result
        
        # Verify throughput is better than sequential execution
        sequential_duration = sum(
            asyncio.run(asyncio.to_thread(engine.evaluate, base_config))
            for _ in range(num_concurrent)
        )
        assert total_duration < sequential_duration

    def test_batch_processing_efficiency(self, engine, base_config):
        """Test efficiency of different batch processing strategies."""
        batch_sizes = [1, 4, 8, 16, 32]
        results = {}
        
        for batch_size in batch_sizes:
            base_config["model"]["batch_size"] = batch_size
            base_config["dataset"]["batch_size"] = batch_size
            
            # Measure CPU usage
            process = psutil.Process(os.getpid())
            cpu_percentages = []
            
            def cpu_callback():
                cpu_percentages.append(process.cpu_percent())
            
            start_time = time.time()
            engine.evaluate(base_config, cpu_callback=cpu_callback)
            end_time = time.time()
            
            duration = end_time - start_time
            avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
            
            results[batch_size] = {
                "duration": duration,
                "avg_cpu": avg_cpu,
                "cpu_samples": len(cpu_percentages)
            }
        
        # Verify CPU utilization increases with batch size
        for i in range(1, len(batch_sizes)):
            assert results[batch_sizes[i]]["avg_cpu"] >= results[batch_sizes[i-1]]["avg_cpu"]

    def test_parallel_processing_scaling(self, engine, base_config):
        """Test scaling with different numbers of worker processes."""
        num_workers = [1, 2, 4, 8]
        results = {}
        
        for workers in num_workers:
            base_config["num_workers"] = workers
            
            start_time = time.time()
            engine.evaluate(base_config)
            end_time = time.time()
            
            duration = end_time - start_time
            total_samples = base_config["dataset"]["max_samples"]
            samples_per_second = total_samples / duration
            
            results[workers] = {
                "duration": duration,
                "samples_per_second": samples_per_second
            }
        
        # Verify performance scales with number of workers (with diminishing returns)
        for i in range(1, len(num_workers)):
            speedup = results[num_workers[i]]["samples_per_second"] / results[num_workers[i-1]]["samples_per_second"]
            assert speedup > 1.0  # Should see some improvement
            assert speedup < num_workers[i] / num_workers[i-1]  # But not linear scaling

    def test_memory_efficiency(self, engine, base_config):
        """Test memory efficiency with different configurations."""
        configs = [
            {"batch_size": 1, "num_workers": 1},
            {"batch_size": 4, "num_workers": 1},
            {"batch_size": 4, "num_workers": 4},
            {"batch_size": 8, "num_workers": 4}
        ]
        
        results = {}
        process = psutil.Process(os.getpid())
        
        for config in configs:
            base_config["model"]["batch_size"] = config["batch_size"]
            base_config["dataset"]["batch_size"] = config["batch_size"]
            base_config["num_workers"] = config["num_workers"]
            
            start_memory = process.memory_info().rss
            engine.evaluate(base_config)
            end_memory = process.memory_info().rss
            
            memory_used = end_memory - start_memory
            results[f"{config['batch_size']}_{config['num_workers']}"] = memory_used
        
        # Verify memory usage is proportional to batch size and workers
        for i in range(1, len(configs)):
            prev_config = configs[i-1]
            curr_config = configs[i]
            
            prev_memory = results[f"{prev_config['batch_size']}_{prev_config['num_workers']}"]
            curr_memory = results[f"{curr_config['batch_size']}_{curr_config['num_workers']}"]
            
            # Memory should increase but not linearly
            assert curr_memory > prev_memory
            assert curr_memory < prev_memory * 2  # Less than double 