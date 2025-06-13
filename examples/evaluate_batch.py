"""
Example script demonstrating how to use the AHNS framework to evaluate a batch of images.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm

from ahns.core import AHNSEvaluator
from ahns.image_evaluators import ImageHarmonyEvaluator, ImageNoveltyEvaluator
from ahns.utils import (
    compute_batch_embeddings,
    save_evaluation_results,
    save_embeddings
)


def load_images(image_dir: Path) -> List[Image.Image]:
    """Load all images from a directory."""
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    return [Image.open(p) for p in image_paths]


def evaluate_batch(
    images: List[Image.Image],
    evaluator: AHNSEvaluator,
    reference_embeddings: Optional[torch.Tensor] = None,
    prior_embeddings: Optional[torch.Tensor] = None,
    batch_size: int = 32
) -> List[dict]:
    """Evaluate a batch of images."""
    results = []
    
    for i in tqdm(range(0, len(images), batch_size), desc="Evaluating images"):
        batch_images = images[i:i + batch_size]
        
        for image in batch_images:
            result = evaluator.evaluate(
                image=image,
                reference_embeddings=reference_embeddings,
                prior_embeddings=prior_embeddings
            )
            
            results.append({
                "ahns_score": result.ahns_score,
                "harmony_score": result.harmony_score,
                "novelty_score": result.novelty_score,
                **result.components,
                **result.metadata
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a batch of images using AHNS")
    parser.add_argument(
        "image_dir",
        type=str,
        help="Directory containing images to evaluate"
    )
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save computed embeddings for future use"
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load images
    image_dir = Path(args.image_dir)
    images = load_images(image_dir)
    print(f"Loaded {len(images)} images from {image_dir}")
    
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
    
    # Evaluate images
    results = evaluate_batch(
        images=images,
        evaluator=evaluator,
        reference_embeddings=reference_embeddings,
        prior_embeddings=prior_embeddings,
        batch_size=args.batch_size
    )
    
    # Save results
    save_evaluation_results(
        results=results,
        output_dir=output_dir,
        prefix="batch_eval"
    )
    
    # Save embeddings if requested
    if args.save_embeddings:
        embeddings = compute_batch_embeddings(
            images=images,
            model=novelty_evaluator.model,
            processor=novelty_evaluator.processor,
            batch_size=args.batch_size
        )
        
        save_embeddings(
            embeddings=embeddings,
            output_path=output_dir / "batch_embeddings.pt",
            metadata={
                "num_images": len(images),
                "image_dir": str(image_dir)
            }
        )
    
    # Print summary
    ahns_scores = [r["ahns_score"] for r in results]
    harmony_scores = [r["harmony_score"] for r in results]
    novelty_scores = [r["novelty_score"] for r in results]
    
    print("\nEvaluation Summary:")
    print(f"Number of images: {len(images)}")
    print(f"Mean AHNS Score: {sum(ahns_scores) / len(ahns_scores):.3f}")
    print(f"Mean Harmony Score: {sum(harmony_scores) / len(harmony_scores):.3f}")
    print(f"Mean Novelty Score: {sum(novelty_scores) / len(novelty_scores):.3f}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main() 