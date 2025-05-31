"""
Multimodal evaluation example for the evaluation framework.
This example demonstrates how to:
1. Load and evaluate multimodal models (text + image)
2. Process image-text pairs
3. Use multimodal metrics
4. Visualize results
"""

import os
from pathlib import Path
from eval_framework.core.engine import EvaluationEngine
from eval_framework.models.multimodal_model import MultimodalModel
from eval_framework.datasets.multimodal_dataset import MultimodalDataset
from eval_framework.evaluators.multimodal import MultimodalEvaluator
from eval_framework.metrics.clip_score import CLIPScore
from eval_framework.metrics.image_caption_bleu import ImageCaptionBLEU

def main():
    # Create output directory
    output_dir = Path("output/multimodal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure evaluation
    config = {
        "model": {
            "type": "multimodal",
            "name": "clip-vit-base-patch32",
            "batch_size": 4,
            "device": "cuda:0",
            "image_size": 224,
            "text_model": {
                "name": "gpt2",
                "max_length": 77
            }
        },
        "dataset": {
            "type": "multimodal",
            "name": "coco_captions",
            "split": "val",
            "max_samples": 100,
            "batch_size": 4,
            "image_dir": "data/coco/images",
            "annotations_file": "data/coco/annotations/captions_val2017.json",
            "image_transform": {
                "resize": [224, 224],
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            }
        },
        "evaluator": {
            "type": "multimodal",
            "similarity_threshold": 0.7,
            "modalities": ["text", "image"],
            "metrics": ["cosine_similarity", "euclidean_distance"]
        },
        "metrics": [
            {
                "name": "clip_score",
                "type": "clip_score",
                "model_name": "clip-vit-base-patch32"
            },
            {
                "name": "image_caption_bleu",
                "type": "image_caption_bleu",
                "n_grams": [1, 2, 3, 4]
            }
        ],
        "output": {
            "formats": ["json", "html"],
            "save_predictions": True,
            "save_metrics": True,
            "save_error_analysis": True,
            "save_visualizations": True,
            "output_dir": str(output_dir)
        }
    }
    
    # Initialize evaluation engine
    engine = EvaluationEngine()
    
    # Define progress callback
    def progress_callback(current, total, message):
        print(f"Progress: {current}/{total} - {message}")
    
    # Run evaluation
    print("Starting multimodal evaluation...")
    results = engine.evaluate(config, progress_callback=progress_callback)
    
    # Display results
    print("\nEvaluation Results:")
    print("------------------")
    print("\nCLIP Score Results:")
    for metric_name, score in results["metrics"]["clip_score"].items():
        print(f"  {metric_name}: {score:.4f}")
    
    print("\nBLEU Score Results:")
    for metric_name, score in results["metrics"]["image_caption_bleu"].items():
        print(f"  {metric_name}: {score:.4f}")
    
    print(f"\nTotal samples evaluated: {len(results['predictions'])}")
    print(f"Results saved to: {output_dir}")
    
    # Display sample predictions with visualizations
    print("\nSample Predictions:")
    print("------------------")
    for i, (pred, ref, image_path) in enumerate(zip(
        results["predictions"][:3],
        results["references"][:3],
        results["image_paths"][:3]
    )):
        print(f"\nSample {i+1}:")
        print(f"Image: {image_path}")
        print(f"Reference: {ref}")
        print(f"Prediction: {pred}")
        print(f"CLIP Score: {results['metrics']['clip_score']['clip_score'][i]:.4f}")
        print(f"BLEU-4: {results['metrics']['image_caption_bleu']['bleu-4'][i]:.4f}")
        
        # Save visualization
        vis_path = output_dir / f"sample_{i+1}_visualization.png"
        engine.visualize_prediction(
            image_path=image_path,
            prediction=pred,
            reference=ref,
            output_path=vis_path
        )
        print(f"Visualization saved to: {vis_path}")

if __name__ == "__main__":
    main() 