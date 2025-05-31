"""
Batch evaluation example for the evaluation framework.
This example demonstrates how to:
1. Run evaluations with multiple models
2. Compare results across different configurations
3. Generate comparative reports
4. Handle large-scale evaluations
"""

import os
import asyncio
from pathlib import Path
from eval_framework.core.engine import EvaluationEngine
from eval_framework.models.text_model import TextModel
from eval_framework.datasets.text_dataset import TextDataset

def main():
    # Create output directory
    output_dir = Path("output/batch_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define model configurations
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
        },
        {
            "type": "text",
            "name": "gpt2-large",
            "batch_size": 2,
            "device": "cpu"
        }
    ]
    
    # Define dataset configurations
    datasets = [
        {
            "type": "text",
            "name": "test_dataset",
            "split": "test",
            "max_samples": 100,
            "batch_size": 4
        },
        {
            "type": "text",
            "name": "test_dataset",
            "split": "test",
            "max_samples": 1000,
            "batch_size": 8
        }
    ]
    
    # Define evaluator configurations
    evaluators = [
        {
            "type": "exact_match",
            "normalize_text": True,
            "case_sensitive": False
        },
        {
            "type": "rouge",
            "metrics": ["rouge1", "rouge2", "rougeL"],
            "use_stemmer": True
        }
    ]
    
    # Initialize evaluation engine
    engine = EvaluationEngine()
    
    # Define progress callback
    def progress_callback(current, total, message):
        print(f"Progress: {current}/{total} - {message}")
    
    # Run batch evaluation
    print("Starting batch evaluation...")
    results = {}
    
    for model_config in models:
        model_name = model_config["name"]
        results[model_name] = {}
        
        for dataset_config in datasets:
            dataset_name = f"{dataset_config['name']}_{dataset_config['max_samples']}"
            results[model_name][dataset_name] = {}
            
            for evaluator_config in evaluators:
                evaluator_type = evaluator_config["type"]
                
                # Configure evaluation
                config = {
                    "model": model_config,
                    "dataset": dataset_config,
                    "evaluator": evaluator_config,
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
                        "output_dir": str(output_dir / model_name / dataset_name / evaluator_type)
                    }
                }
                
                print(f"\nEvaluating {model_name} on {dataset_name} with {evaluator_type}...")
                result = engine.evaluate(config, progress_callback=progress_callback)
                results[model_name][dataset_name][evaluator_type] = result
    
    # Generate comparative report
    print("\nGenerating comparative report...")
    engine.generate_comparative_report(
        results=results,
        output_path=output_dir / "comparative_report.html",
        metrics=["rouge1", "rouge2", "rougeL"]
    )
    
    # Display summary
    print("\nEvaluation Summary:")
    print("------------------")
    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        for dataset_name, dataset_results in model_results.items():
            print(f"\n  Dataset: {dataset_name}")
            for evaluator_type, evaluator_results in dataset_results.items():
                print(f"\n    Evaluator: {evaluator_type}")
                for metric_name, score in evaluator_results["metrics"].items():
                    print(f"      {metric_name}: {score:.4f}")
    
    print(f"\nDetailed results saved to: {output_dir}")
    print(f"Comparative report: {output_dir}/comparative_report.html")

if __name__ == "__main__":
    main() 