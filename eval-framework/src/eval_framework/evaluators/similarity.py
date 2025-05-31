"""Semantic similarity evaluator implementation."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from eval_framework.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

class SimilarityMetric(Enum):
    """Types of similarity metrics."""
    COSINE = "cosine"         # Cosine similarity
    EUCLIDEAN = "euclidean"   # Euclidean distance
    DOT = "dot"              # Dot product

@dataclass
class SimilarityConfig:
    """Configuration for similarity evaluation."""
    model_name: str = "all-MiniLM-L6-v2"  # Default sentence-transformer model
    metric: SimilarityMetric = SimilarityMetric.COSINE
    threshold: float = 0.7
    batch_size: int = 32
    normalize: bool = True
    device: Optional[str] = None
    max_length: int = 512
    pooling_strategy: str = "mean"  # mean, max, cls
    cache_dir: Optional[str] = None

class SimilarityEvaluator(BaseEvaluator):
    """Evaluator that uses semantic embeddings for similarity scoring.
    
    This evaluator uses BERT/sentence-transformer embeddings to compute
    semantic similarity between texts, with support for different metrics
    and threshold-based classification.
    """
    
    def __init__(
        self,
        config: Optional[SimilarityConfig] = None,
        **kwargs: Any
    ):
        """Initialize similarity evaluator.
        
        Args:
            config: Evaluation configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or SimilarityConfig(**kwargs)
        
        # Set device
        if self.config.device is None:
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        try:
            self.model = SentenceTransformer(
                self.config.model_name,
                cache_folder=self.config.cache_dir,
                device=self.config.device
            )
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {str(e)}")
            raise
        
        # Set model parameters
        self.model.max_seq_length = self.config.max_length
        
        # Set pooling strategy
        if self.config.pooling_strategy == "mean":
            self.model.pooling = lambda x: torch.mean(x, dim=1)
        elif self.config.pooling_strategy == "max":
            self.model.pooling = lambda x: torch.max(x, dim=1)[0]
        elif self.config.pooling_strategy == "cls":
            self.model.pooling = lambda x: x[:, 0]
        else:
            raise ValueError(f"Invalid pooling strategy: {self.config.pooling_strategy}")
    
    def _compute_similarity(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity scores
        """
        if self.config.metric == SimilarityMetric.COSINE:
            return cos_sim(embeddings1, embeddings2)
        elif self.config.metric == SimilarityMetric.EUCLIDEAN:
            # Convert to distance and then to similarity
            dist = torch.cdist(embeddings1, embeddings2)
            return 1 / (1 + dist)
        else:  # DOT
            return torch.matmul(embeddings1, embeddings2.t())
    
    def _encode_texts(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Tensor of embeddings
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=self.config.normalize
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}")
            raise
    
    def evaluate(
        self,
        prediction: str,
        reference: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Evaluate similarity between prediction and reference.
        
        Args:
            prediction: Model prediction
            reference: Reference text
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        # Encode texts
        embeddings = self._encode_texts([prediction, reference])
        pred_emb, ref_emb = embeddings[0:1], embeddings[1:2]
        
        # Compute similarity
        similarity = self._compute_similarity(pred_emb, ref_emb)
        score = similarity.item()
        
        # Classify based on threshold
        is_similar = score >= self.config.threshold
        
        return {
            "score": score,
            "prediction": prediction,
            "reference": reference,
            "similar": is_similar,
            "threshold": self.config.threshold,
            "metric": self.config.metric.value
        }
    
    def batch_evaluate(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Evaluate batch of predictions.
        
        Args:
            predictions: List of model predictions
            references: List of reference texts
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing batch evaluation results
        """
        # Encode all texts
        all_texts = predictions + references
        embeddings = self._encode_texts(all_texts)
        
        # Split embeddings
        pred_emb = embeddings[:len(predictions)]
        ref_emb = embeddings[len(predictions):]
        
        # Compute similarities
        similarities = self._compute_similarity(pred_emb, ref_emb)
        scores = similarities.diagonal().tolist()
        
        # Classify based on threshold
        similar_count = sum(1 for score in scores if score >= self.config.threshold)
        
        # Calculate metrics
        num_samples = len(predictions)
        avg_score = sum(scores) / num_samples if num_samples > 0 else 0.0
        
        return {
            "average_score": avg_score,
            "total_similar": similar_count,
            "total_samples": num_samples,
            "similarity_rate": similar_count / num_samples if num_samples > 0 else 0.0,
            "threshold": self.config.threshold,
            "metric": self.config.metric.value,
            "scores": scores
        }
    
    def get_embeddings(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """Get embeddings for texts.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Tensor of embeddings
        """
        return self._encode_texts(texts)
    
    def compute_pairwise_similarities(
        self,
        texts1: List[str],
        texts2: List[str]
    ) -> torch.Tensor:
        """Compute pairwise similarities between two sets of texts.
        
        Args:
            texts1: First set of texts
            texts2: Second set of texts
            
        Returns:
            Matrix of similarity scores
        """
        # Encode texts
        emb1 = self._encode_texts(texts1)
        emb2 = self._encode_texts(texts2)
        
        # Compute similarities
        return self._compute_similarity(emb1, emb2) 