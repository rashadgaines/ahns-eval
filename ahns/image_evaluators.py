"""
Image-specific implementations of harmony and novelty evaluators for AHNS.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms
from transformers import CLIPProcessor, CLIPVisionModel

from ahns.core import HarmonyEvaluator, NoveltyEvaluator


class ImageHarmonyEvaluator(HarmonyEvaluator):
    """Evaluates aesthetic harmony in images using color theory, composition, and texture."""
    
    def __init__(
        self,
        n_colors: int = 5,
        color_harmony_weight: float = 0.4,
        composition_weight: float = 0.3,
        texture_weight: float = 0.3
    ):
        """
        Initialize the image harmony evaluator.
        
        Args:
            n_colors: Number of dominant colors to extract
            color_harmony_weight: Weight for color harmony component
            composition_weight: Weight for composition component
            texture_weight: Weight for texture component
        """
        self.n_colors = n_colors
        self.color_harmony_weight = color_harmony_weight
        self.composition_weight = composition_weight
        self.texture_weight = texture_weight
        
        # Verify weights sum to 1
        total_weight = color_harmony_weight + composition_weight + texture_weight
        if not math.isclose(total_weight, 1.0):
            raise ValueError("Component weights must sum to 1.0")
    
    def evaluate(self, image: Image.Image) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the aesthetic harmony of an image.
        
        Args:
            image: PIL Image to evaluate
            
        Returns:
            Tuple of (overall harmony score, component scores)
        """
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Evaluate color harmony
        color_score = self._evaluate_color_harmony(img_array)
        
        # Evaluate composition
        comp_score = self._evaluate_composition(img_array)
        
        # Evaluate texture
        texture_score = self._evaluate_texture(img_array)
        
        # Compute weighted average
        harmony_score = (
            self.color_harmony_weight * color_score +
            self.composition_weight * comp_score +
            self.texture_weight * texture_score
        )
        
        components = {
            "color_harmony": color_score,
            "composition": comp_score,
            "texture": texture_score
        }
        
        return harmony_score, components
    
    def _evaluate_color_harmony(self, img_array: np.ndarray) -> float:
        """Evaluate color harmony using k-means clustering and color theory."""
        # Reshape image for clustering
        pixels = img_array.reshape(-1, 3)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
        kmeans.fit(pixels)
        
        # Get dominant colors
        colors = kmeans.cluster_centers_
        
        # Convert to HSV for color harmony analysis
        colors_hsv = cv2.cvtColor(np.uint8([colors]), cv2.COLOR_RGB2HSV)[0]
        
        # Calculate color harmony score based on hue relationships
        harmony_score = self._calculate_color_harmony_score(colors_hsv[:, 0])
        
        return harmony_score
    
    def _calculate_color_harmony_score(self, hues: np.ndarray) -> float:
        """Calculate color harmony score based on hue relationships."""
        # Define common color harmony schemes (in degrees)
        schemes = {
            "complementary": [0, 180],
            "triadic": [0, 120, 240],
            "analogous": [0, 30, 60],
            "split_complementary": [0, 150, 210]
        }
        
        # Calculate minimum distance to any harmony scheme
        min_dist = float("inf")
        for scheme in schemes.values():
            dist = self._calculate_scheme_distance(hues, scheme)
            min_dist = min(min_dist, dist)
        
        # Convert distance to score (closer to scheme = higher score)
        return math.exp(-0.1 * min_dist)
    
    def _calculate_scheme_distance(self, hues: np.ndarray, scheme: List[float]) -> float:
        """Calculate distance between current colors and a harmony scheme."""
        # Normalize hues to [0, 360]
        hues = hues * 2  # Convert from [0, 180] to [0, 360]
        
        # Calculate minimum distance between each hue and scheme
        total_dist = 0
        for hue in hues:
            min_hue_dist = min(abs(hue - s) for s in scheme)
            total_dist += min_hue_dist
        
        return total_dist / len(hues)
    
    def _evaluate_composition(self, img_array: np.ndarray) -> float:
        """Evaluate composition using saliency and quadrant analysis."""
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate saliency map (simplified version)
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        _, saliency_map = saliency.computeSaliency(gray)
        
        # Divide image into quadrants
        h, w = saliency_map.shape
        mid_h, mid_w = h // 2, w // 2
        
        quadrants = [
            saliency_map[:mid_h, :mid_w],
            saliency_map[:mid_h, mid_w:],
            saliency_map[mid_h:, :mid_w],
            saliency_map[mid_h:, mid_w:]
        ]
        
        # Calculate entropy for each quadrant
        entropies = [self._calculate_entropy(q) for q in quadrants]
        
        # Calculate composition score based on entropy distribution
        comp_score = 1 - (max(entropies) - min(entropies)) / math.log(4)
        
        return comp_score
    
    def _calculate_entropy(self, quadrant: np.ndarray) -> float:
        """Calculate entropy of a quadrant."""
        # Normalize to [0, 1]
        quadrant = (quadrant - quadrant.min()) / (quadrant.max() - quadrant.min() + 1e-8)
        
        # Calculate histogram
        hist = np.histogram(quadrant, bins=50, range=(0, 1))[0]
        hist = hist / hist.sum()
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        
        return entropy
    
    def _evaluate_texture(self, img_array: np.ndarray) -> float:
        """Evaluate texture consistency using local feature analysis."""
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Extract local features using Sobel operators
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Divide image into patches
        patch_size = 32
        h, w = magnitude.shape
        patches = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = magnitude[i:i + patch_size, j:j + patch_size]
                patches.append(patch.flatten())
        
        # Calculate cosine similarity between patches
        patches = np.array(patches)
        similarity_matrix = np.zeros((len(patches), len(patches)))
        
        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                similarity = np.dot(patches[i], patches[j]) / (
                    np.linalg.norm(patches[i]) * np.linalg.norm(patches[j])
                )
                similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
        
        # Calculate texture consistency score
        texture_score = np.mean(similarity_matrix)
        
        return texture_score


class ImageNoveltyEvaluator(NoveltyEvaluator):
    """Evaluates novelty in images using CLIP embeddings and domain-specific metrics."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the image novelty evaluator.
        
        Args:
            model_name: Name of the CLIP model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Initialize weights for novelty components
        self.train_weight = 0.4
        self.prior_weight = 0.4
        self.domain_weight = 0.2
    
    def evaluate(
        self,
        image: Image.Image,
        reference_embeddings: Optional[torch.Tensor] = None,
        prior_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the novelty of an image.
        
        Args:
            image: PIL Image to evaluate
            reference_embeddings: Optional tensor of reference image embeddings
            prior_embeddings: Optional tensor of prior generated image embeddings
            
        Returns:
            Tuple of (overall novelty score, component scores)
        """
        # Get image embedding
        image_embedding = self._get_image_embedding(image)
        
        # Calculate novelty components
        train_score = self._calculate_train_novelty(image_embedding, reference_embeddings)
        prior_score = self._calculate_prior_novelty(image_embedding, prior_embeddings)
        domain_score = self._calculate_domain_plausibility(image_embedding)
        
        # Compute weighted average
        novelty_score = (
            self.train_weight * train_score +
            self.prior_weight * prior_score +
            self.domain_weight * domain_score
        )
        
        components = {
            "train_novelty": train_score,
            "prior_novelty": prior_score,
            "domain_plausibility": domain_score
        }
        
        return novelty_score, components
    
    def _get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """Get CLIP embedding for an image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu()
    
    def _calculate_train_novelty(
        self,
        image_embedding: torch.Tensor,
        reference_embeddings: Optional[torch.Tensor]
    ) -> float:
        """Calculate novelty score based on distance from training data."""
        if reference_embeddings is None:
            return 1.0  # No reference data, assume maximum novelty
        
        # Calculate cosine distances
        distances = F.cosine_similarity(
            image_embedding.unsqueeze(0),
            reference_embeddings
        )
        
        # Convert to novelty score (higher distance = higher novelty)
        novelty = 1 - distances.max().item()
        
        return novelty
    
    def _calculate_prior_novelty(
        self,
        image_embedding: torch.Tensor,
        prior_embeddings: Optional[torch.Tensor]
    ) -> float:
        """Calculate novelty score based on distance from prior outputs."""
        if prior_embeddings is None:
            return 1.0  # No prior outputs, assume maximum novelty
        
        # Calculate cosine distances
        distances = F.cosine_similarity(
            image_embedding.unsqueeze(0),
            prior_embeddings
        )
        
        # Convert to novelty score (higher distance = higher novelty)
        novelty = 1 - distances.max().item()
        
        return novelty
    
    def _calculate_domain_plausibility(self, image_embedding: torch.Tensor) -> float:
        """
        Calculate domain plausibility score.
        This is a placeholder implementation - in practice, you would:
        1. Train a discriminator on your artistic domain
        2. Use it to score whether the image belongs to the target style
        """
        # For now, return a constant score
        # In practice, this should be replaced with actual domain classification
        return 0.8 