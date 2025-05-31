"""
Basic usage example for the evaluation framework.
This example demonstrates how to:
1. Load a model and dataset
2. Run a simple evaluation
3. Process and display results
"""

import os
from pathlib import Path
from eval_framework.core.engine import EvaluationEngine
from eval_framework.models.text_model import TextModel
from eval_framework.datasets.text_dataset import TextDataset
from eval_framework.evaluators.exact_match import ExactMatchEvaluator
from eval_framework.metrics.rouge import ROUGEMetric

def main():
    # Create output directory
    output_dir = Path("output/basic_usage")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure evaluation
    config = {
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
                "metrics": ["rouge1", "rouge2", "rougeL"],
                "use_stemmer": True
            }
        ],
        "output": {
            "formats": ["json", "html"],
            "save_predictions": True,
            "save_metrics": True,
            "save_error_analysis": True,
            "output_dir": str(output_dir)
        }
    }
    
    # Initialize evaluation engine
    engine = EvaluationEngine()
    
    # Define progress callback
    def progress_callback(current, total, message):
        print(f"Progress: {current}/{total} - {message}")
    
    # Run evaluation
    print("Starting evaluation...")
    results = engine.evaluate(config, progress_callback=progress_callback)
    
    # Display results
    print("\nEvaluation Results:")
    print("------------------")
    print(f"Metrics:")
    for metric_name, score in results["metrics"].items():
        print(f"  {metric_name}: {score:.4f}")
    
    print(f"\nTotal samples evaluated: {len(results['predictions'])}")
    print(f"Results saved to: {output_dir}")
    
    # Display sample predictions
    print("\nSample Predictions:")
    print("------------------")
    for i, (pred, ref) in enumerate(zip(results["predictions"][:3], results["references"][:3])):
        print(f"\nSample {i+1}:")
        print(f"Reference: {ref}")
        print(f"Prediction: {pred}")

if __name__ == "__main__":
    main() 