# Troubleshooting Guide

## Common Issues and Solutions

### Model Loading Issues

1. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solutions:**
   - Reduce batch size
   - Use gradient accumulation
   - Enable memory efficient attention
   - Use model parallelism
   ```python
   config = {
       "model": {
           "batch_size": 2,  # Reduced from 4
           "gradient_accumulation_steps": 4,
           "memory_efficient_attention": True
       }
   }
   ```

2. **Model Not Found**
   ```
   ModelNotFoundError: Model 'gpt2' not found
   ```
   **Solutions:**
   - Check model name spelling
   - Verify model is installed
   - Check internet connection
   - Use local model path
   ```python
   config = {
       "model": {
           "name": "gpt2",
           "local_path": "/path/to/local/model"
       }
   }
   ```

### Dataset Issues

1. **File Not Found**
   ```
   FileNotFoundError: Dataset file not found
   ```
   **Solutions:**
   - Verify file paths
   - Check file permissions
   - Ensure data is downloaded
   - Use absolute paths
   ```python
   config = {
       "dataset": {
           "path": "/absolute/path/to/dataset",
           "download": True
       }
   }
   ```

2. **Memory Issues with Large Datasets**
   ```
   MemoryError: Unable to allocate memory
   ```
   **Solutions:**
   - Use batch loading
   - Implement data streaming
   - Reduce max samples
   - Use memory mapping
   ```python
   config = {
       "dataset": {
           "batch_size": 32,
           "streaming": True,
           "max_samples": 1000,
           "use_mmap": True
       }
   }
   ```

### Evaluation Issues

1. **Metric Computation Errors**
   ```
   ValueError: Invalid metric configuration
   ```
   **Solutions:**
   - Check metric parameters
   - Verify input formats
   - Use compatible metrics
   - Handle edge cases
   ```python
   config = {
       "metrics": [
           {
               "name": "rouge",
               "type": "rouge",
               "metrics": ["rouge1", "rouge2"],
               "use_stemmer": True,
               "handle_edge_cases": True
           }
       ]
   }
   ```

2. **Slow Evaluation**
   ```
   PerformanceWarning: Evaluation is taking longer than expected
   ```
   **Solutions:**
   - Enable parallel processing
   - Use GPU acceleration
   - Optimize batch size
   - Cache intermediate results
   ```python
   config = {
       "evaluation": {
           "parallel": True,
           "num_workers": 4,
           "use_gpu": True,
           "cache_results": True
       }
   }
   ```

### Output Issues

1. **File Permission Errors**
   ```
   PermissionError: Unable to write to output directory
   ```
   **Solutions:**
   - Check directory permissions
   - Create output directory
   - Use absolute paths
   - Handle existing files
   ```python
   import os
   from pathlib import Path
   
   output_dir = Path("output/results")
   output_dir.mkdir(parents=True, exist_ok=True)
   ```

2. **Format Errors**
   ```
   ValueError: Unsupported output format
   ```
   **Solutions:**
   - Check format support
   - Install required packages
   - Use supported formats
   - Handle format conversion
   ```python
   config = {
       "output": {
           "formats": ["json", "html"],
           "convert_formats": True
       }
   }
   ```

## Performance Optimization

### Memory Optimization

1. **High Memory Usage**
   **Solutions:**
   - Use memory profiling
   - Implement garbage collection
   - Clear caches
   - Monitor memory usage
   ```python
   import gc
   import psutil
   
   def monitor_memory():
       process = psutil.Process()
       print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   
   # Clear memory
   gc.collect()
   ```

2. **Memory Leaks**
   **Solutions:**
   - Check resource cleanup
   - Use context managers
   - Monitor object lifecycle
   - Implement proper disposal
   ```python
   from contextlib import contextmanager
   
   @contextmanager
   def managed_evaluation():
       engine = EvaluationEngine()
       try:
           yield engine
       finally:
           engine.cleanup()
   ```

### Speed Optimization

1. **Slow Processing**
   **Solutions:**
   - Profile code
   - Optimize bottlenecks
   - Use caching
   - Implement batching
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_computation(x):
       return expensive_operation(x)
   ```

2. **I/O Bottlenecks**
   **Solutions:**
   - Use async I/O
   - Implement buffering
   - Optimize file operations
   - Use efficient formats
   ```python
   import aiofiles
   
   async def async_file_operation():
       async with aiofiles.open('file.txt', 'r') as f:
           content = await f.read()
   ```

## Debugging Tools

### Logging

1. **Enable Debug Logging**
   ```python
   import logging
   
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   ```

2. **Structured Logging**
   ```python
   import structlog
   
   logger = structlog.get_logger()
   logger.info("processing_batch", 
       batch_size=32,
       model="gpt2",
       device="cuda:0"
   )
   ```

### Profiling

1. **CPU Profiling**
   ```python
   import cProfile
   
   profiler = cProfile.Profile()
   profiler.enable()
   # Run evaluation
   profiler.disable()
   profiler.print_stats()
   ```

2. **Memory Profiling**
   ```python
   from memory_profiler import profile
   
   @profile
   def evaluate_model():
       # Evaluation code
       pass
   ```

## Common Error Messages

### Model Errors

1. **Input Shape Mismatch**
   ```
   RuntimeError: Expected input shape (batch_size, seq_len) but got (batch_size, seq_len, dim)
   ```
   **Solution:**
   ```python
   # Reshape input
   input_tensor = input_tensor.view(batch_size, seq_len)
   ```

2. **Device Mismatch**
   ```
   RuntimeError: Expected all tensors to be on the same device
   ```
   **Solution:**
   ```python
   # Move tensors to same device
   tensor = tensor.to(device)
   ```

### Dataset Errors

1. **Invalid Data Format**
   ```
   ValueError: Invalid data format
   ```
   **Solution:**
   ```python
   # Validate data format
   def validate_data(data):
       assert isinstance(data, (list, tuple))
       assert all(isinstance(x, str) for x in data)
   ```

2. **Missing Data**
   ```
   KeyError: Required field not found
   ```
   **Solution:**
   ```python
   # Handle missing data
   def get_field(data, field, default=None):
       return data.get(field, default)
   ```

## Getting Help

### Support Channels

1. **GitHub Issues**
   - Report bugs
   - Request features
   - Ask questions
   - Share solutions

2. **Documentation**
   - Read the docs
   - Check examples
   - Review API reference
   - Follow tutorials

3. **Community**
   - Join discussions
   - Share experiences
   - Contribute code
   - Help others 