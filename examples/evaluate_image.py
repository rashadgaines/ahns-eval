"""
Example script demonstrating how to use the AHNS framework to evaluate a single image.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from ahns.core import AHNSEvaluator
from ahns.image_evaluators import ImageHarmonyEvaluator, ImageNoveltyEvaluator
from ahns.utils import save_evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate an image using AHNS")
    parser.add_argument("image_path", type=str, help="Path to the image to evaluate")
    parser.add_argument(
        "--reference-embeddings",
        type=str,
        help="Path to reference image embeddings (optional)"
    )
    parser.add_argument(
        "--prior-embeddings",
        type=str,
        help="Path to prior generated image embeddings (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results"
    )
    args = parser.parse_args()
    
    # Load image
    image = Image.open(args.image_path)
    
    # Initialize evaluators
    harmony_evaluator = ImageHarmonyEvaluator()
    novelty_evaluator = ImageNoveltyEvaluator()
    
    # Initialize AHNS evaluator
    evaluator = AHNSEvaluator(
        harmony_evaluator=harmony_evaluator,
        novelty_evaluator=novelty_evaluator,
        harmony_weight=0.5,
        novelty_weight=0.5
    )
    
    # Load embeddings if provided
    reference_embeddings = None
    if args.reference_embeddings:
        reference_embeddings = torch.load(args.reference_embeddings)
    
    prior_embeddings = None
    if args.prior_embeddings:
        prior_embeddings = torch.load(args.prior_embeddings)
    
    # Evaluate image
    result = evaluator.evaluate(
        image=image,
        reference_embeddings=reference_embeddings,
        prior_embeddings=prior_embeddings,
        metadata={"image_path": str(args.image_path)}
    )
    
    # Print results
    print("\nAHNS Evaluation Results:")
    print(f"Image: {args.image_path}")
    print(f"AHNS Score: {result.ahns_score:.3f}")
    print(f"Harmony Score: {result.harmony_score:.3f}")
    print(f"Novelty Score: {result.novelty_score:.3f}")
    print("\nComponent Scores:")
    for component, score in result.components.items():
        print(f"{component}: {score:.3f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    save_evaluation_results(
        results=[{
            "image_path": str(args.image_path),
            "ahns_score": result.ahns_score,
            "harmony_score": result.harmony_score,
            "novelty_score": result.novelty_score,
            **result.components,
            **result.metadata
        }],
        output_dir=output_dir,
        prefix="single_image_eval"
    )


if __name__ == "__main__":
    main() 