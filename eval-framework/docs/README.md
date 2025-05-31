# Evaluation Framework

A comprehensive framework for evaluating language models and multimodal models across various tasks and metrics.

## Features

- **Multiple Model Types**
  - Text models (GPT, BERT, etc.)
  - Multimodal models (CLIP, etc.)
  - Custom model support

- **Comprehensive Metrics**
  - ROUGE scores
  - BLEU scores
  - Exact match
  - Semantic similarity
  - CLIP scores
  - Custom metrics

- **Flexible Evaluation**
  - Batch processing
  - Progress tracking
  - Error handling
  - Checkpointing
  - Parallel processing

- **Rich Output**
  - JSON reports
  - HTML visualizations
  - Comparative analysis
  - Error analysis
  - Performance metrics

## Quick Start

1. **Installation**
   ```bash
   pip install eval-framework
   ```

2. **Basic Usage**
   ```python
   from eval_framework.core.engine import EvaluationEngine
   
   # Initialize engine
   engine = EvaluationEngine()
   
   # Configure evaluation
   config = {
       "model": {
           "type": "text",
           "name": "gpt2",
           "batch_size": 4
       },
       "dataset": {
           "type": "text",
           "name": "test_dataset",
           "split": "test"
       },
       "evaluator": {
           "type": "exact_match"
       }
   }
   
   # Run evaluation
   results = engine.evaluate(config)
   ```

3. **View Results**
   ```python
   # Print metrics
   print(results["metrics"])
   
   # Save results
   engine.save_results(results, "output/results.json")
   ```

## Examples

Check out our example notebooks for detailed usage:

- [Basic Usage](examples/basic_usage.py)
- [Custom Evaluator](examples/custom_evaluator.py)
- [Multimodal Evaluation](examples/multimodal_example.py)
- [Batch Evaluation](examples/batch_evaluation.py)

## Documentation

- [API Reference](api.md)
- [Tutorials](tutorials/)
- [Best Practices](best_practices.md)
- [Troubleshooting](troubleshooting.md)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 