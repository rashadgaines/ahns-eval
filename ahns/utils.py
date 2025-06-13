"""
Utility functions for the AHNS framework.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image


def save_embeddings(
    embeddings: torch.Tensor,
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save image embeddings and optional metadata to disk.
    
    Args:
        embeddings: Tensor of image embeddings
        output_path: Path to save the embeddings
        metadata: Optional metadata to save alongside embeddings
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    torch.save(embeddings, output_path)
    
    # Save metadata if provided
    if metadata:
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_embeddings(
    input_path: Union[str, Path]
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """
    Load image embeddings and optional metadata from disk.
    
    Args:
        input_path: Path to the saved embeddings
        
    Returns:
        Tuple of (embeddings tensor, metadata dictionary)
    """
    input_path = Path(input_path)
    
    # Load embeddings
    embeddings = torch.load(input_path)
    
    # Load metadata if it exists
    metadata_path = input_path.with_suffix(".json")
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    return embeddings, metadata


def save_evaluation_results(
    results: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    prefix: str = "ahns_eval"
) -> None:
    """
    Save evaluation results to disk in various formats.
    
    Args:
        results: List of evaluation result dictionaries
        output_dir: Directory to save results
        prefix: Prefix for output filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / f"{prefix}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Save as JSON
    json_path = output_dir / f"{prefix}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary statistics
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    summary = {
        "mean_ahns": df["ahns_score"].mean(),
        "std_ahns": df["ahns_score"].std(),
        "mean_harmony": df["harmony_score"].mean(),
        "std_harmony": df["harmony_score"].std(),
        "mean_novelty": df["novelty_score"].mean(),
        "std_novelty": df["novelty_score"].std(),
        "component_stats": {
            col: {
                "mean": df[col].mean(),
                "std": df[col].std()
            }
            for col in numeric_columns
            if col not in ["ahns_score", "harmony_score", "novelty_score"]
        }
    }
    
    # Save summary
    summary_path = output_dir / f"{prefix}_{timestamp}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def load_evaluation_results(
    input_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Load evaluation results from disk.
    
    Args:
        input_path: Path to the saved results (CSV or JSON)
        
    Returns:
        DataFrame containing the evaluation results
    """
    input_path = Path(input_path)
    
    if input_path.suffix == ".csv":
        return pd.read_csv(input_path)
    elif input_path.suffix == ".json":
        with open(input_path, "r") as f:
            results = json.load(f)
        return pd.DataFrame(results)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")


def preprocess_image(
    image: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess an image for evaluation.
    
    Args:
        image: PIL Image to preprocess
        target_size: Optional target size (width, height)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image tensor
    """
    # Resize if target size is specified
    if target_size:
        image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image)).float()
    
    # Normalize if requested
    if normalize:
        image_tensor = image_tensor / 255.0
    
    # Add batch dimension if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def compute_batch_embeddings(
    images: List[Image.Image],
    model: torch.nn.Module,
    processor: Any,
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute embeddings for a batch of images.
    
    Args:
        images: List of PIL Images
        model: Model to use for computing embeddings
        processor: Image processor for the model
        batch_size: Batch size for processing
        device: Device to use for computation
        
    Returns:
        Tensor of image embeddings
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0) 