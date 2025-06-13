"""
Script for generating images using Grok and evaluating them with AHNS.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import os

import torch
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

from ahns.core import AHNSEvaluator
from ahns.grok_interface import GrokImageGenerator
from ahns.image_evaluators import ImageHarmonyEvaluator, ImageNoveltyEvaluator
from ahns.utils import (
    compute_batch_embeddings,
    save_evaluation_results,
    save_embeddings
)

load_dotenv()

def generate_and_evaluate(
    generator: GrokImageGenerator,
    prompts: List[str],
    num_images_per_prompt: int,
    evaluator: AHNSEvaluator,
    output_dir: str,
    save_images: bool = False,
    save_embeddings: bool = False,
    reference_embeddings: Optional[torch.Tensor] = None,
    prior_embeddings: Optional[torch.Tensor] = None
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """
    Generate and evaluate images for a list of prompts.
    """
    results = []
    all_images = []
    
    for prompt in tqdm(prompts, desc="Processing prompts"):
        # Generate images
        images = generator.generate_image(
            prompt=prompt,
            num_images=num_images_per_prompt
        )
        all_images.extend(images)
        
        # Evaluate each image
        for i, image in enumerate(images):
            # Save image if requested
            if save_images:
                image_path = os.path.join(output_dir, "images", f"{prompt}_{i}.png")
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image.save(image_path)
            
            # Evaluate image
            score = evaluator.evaluate(
                image=image,
                reference_embeddings=reference_embeddings,
                prior_embeddings=prior_embeddings,
                metadata={"prompt": prompt, "index": i}
            )
            
            results.append({
                "prompt": prompt,
                "index": i,
                "harmony_score": score.harmony_score,
                "novelty_score": score.novelty_score,
                "ahns_score": score.ahns_score,
                **score.components,
                **score.metadata
            })
    
    return results, all_images


def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate images using Grok and AHNS")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="Prompts to generate images from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--style",
        type=str,
        help="Style to apply to generated images"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Image dimensions (width height)"
    )
    parser.add_argument(
        "--reference-embeddings",
        type=str,
        help="Path to reference image embeddings"
    )
    parser.add_argument(
        "--prior-embeddings",
        type=str,
        help="Path to prior generated image embeddings"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save generated images"
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save computed embeddings"
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Grok generator
    generator = GrokImageGenerator()
    
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
    
    # Generate and evaluate images
    results, images = generate_and_evaluate(
        generator=generator,
        prompts=args.prompts,
        num_images_per_prompt=args.num_images,
        evaluator=evaluator,
        output_dir=str(output_dir),
        save_images=args.save_images,
        save_embeddings=args.save_embeddings,
        reference_embeddings=reference_embeddings,
        prior_embeddings=prior_embeddings
    )
    
    # Save results
    save_evaluation_results(
        results=results,
        output_dir=output_dir,
        prefix="grok_eval"
    )
    
    # Save embeddings if requested
    if args.save_embeddings:
        embeddings = compute_batch_embeddings(
            images=images,
            model=novelty_evaluator.model,
            processor=novelty_evaluator.processor
        )
        
        save_embeddings(
            embeddings=embeddings,
            output_path=output_dir / "grok_embeddings.pt",
            metadata={
                "num_images": len(images),
                "num_prompts": len(args.prompts),
                "style": args.style,
                "size": args.size
            }
        )
    
    # Print summary
    ahns_scores = [r["ahns_score"] for r in results]
    harmony_scores = [r["harmony_score"] for r in results]
    novelty_scores = [r["novelty_score"] for r in results]
    
    print("\nEvaluation Summary:")
    print(f"Number of prompts: {len(args.prompts)}")
    print(f"Images per prompt: {args.num_images}")
    print(f"Total images: {len(images)}")
    print(f"Mean AHNS Score: {sum(ahns_scores) / len(ahns_scores):.3f}")
    print(f"Mean Harmony Score: {sum(harmony_scores) / len(harmony_scores):.3f}")
    print(f"Mean Novelty Score: {sum(novelty_scores) / len(novelty_scores):.3f}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main() 