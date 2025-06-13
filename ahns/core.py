"""
Core module for the Aesthetic Harmony and Novelty Score (AHNS) evaluation system.
This module provides the base classes and interfaces for implementing AHNS evaluations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class AHNSScore:
    """Container for AHNS evaluation results."""
    harmony_score: float
    novelty_score: float
    ahns_score: float
    components: Dict[str, float]
    metadata: Dict[str, Any]


class HarmonyEvaluator(ABC):
    """Abstract base class for evaluating aesthetic harmony."""
    
    @abstractmethod
    def evaluate(self, image: Image.Image) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the aesthetic harmony of an image.
        
        Args:
            image: PIL Image to evaluate
            
        Returns:
            Tuple of (overall harmony score, component scores)
        """
        pass


class NoveltyEvaluator(ABC):
    """Abstract base class for evaluating novelty."""
    
    @abstractmethod
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
        pass


class AHNSEvaluator:
    """Main class for computing AHNS scores."""
    
    def __init__(
        self,
        harmony_evaluator: HarmonyEvaluator,
        novelty_evaluator: NoveltyEvaluator,
        harmony_weight: float = 0.5,
        novelty_weight: float = 0.5
    ):
        """
        Initialize the AHNS evaluator.
        
        Args:
            harmony_evaluator: Implementation of HarmonyEvaluator
            novelty_evaluator: Implementation of NoveltyEvaluator
            harmony_weight: Weight for harmony score in final AHNS (default: 0.5)
            novelty_weight: Weight for novelty score in final AHNS (default: 0.5)
        """
        self.harmony_evaluator = harmony_evaluator
        self.novelty_evaluator = novelty_evaluator
        self.harmony_weight = harmony_weight
        self.novelty_weight = novelty_weight
        
        if not np.isclose(harmony_weight + novelty_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
    
    def evaluate(
        self,
        image: Image.Image,
        reference_embeddings: Optional[torch.Tensor] = None,
        prior_embeddings: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AHNSScore:
        """
        Compute the AHNS score for an image.
        
        Args:
            image: PIL Image to evaluate
            reference_embeddings: Optional tensor of reference image embeddings
            prior_embeddings: Optional tensor of prior generated image embeddings
            metadata: Optional metadata to include in the score
            
        Returns:
            AHNSScore object containing all evaluation results
        """
        # Evaluate harmony
        harmony_score, harmony_components = self.harmony_evaluator.evaluate(image)
        
        # Evaluate novelty
        novelty_score, novelty_components = self.novelty_evaluator.evaluate(
            image, reference_embeddings, prior_embeddings
        )
        
        # Compute final AHNS score
        ahns_score = (
            self.harmony_weight * harmony_score +
            self.novelty_weight * novelty_score
        )
        
        # Combine all components
        components = {
            **harmony_components,
            **novelty_components
        }
        
        return AHNSScore(
            harmony_score=harmony_score,
            novelty_score=novelty_score,
            ahns_score=ahns_score,
            components=components,
            metadata=metadata or {}
        ) 