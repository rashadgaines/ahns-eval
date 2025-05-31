"""Text evaluation metrics implementation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

@dataclass
class TextMetricConfig:
    """Configuration for text metric computation."""
    # BLEU configuration
    bleu_weights: List[float] = (0.25, 0.25, 0.25, 0.25)  # 1-gram to 4-gram weights
    
    # ROUGE configuration
    rouge_metrics: List[str] = ("rouge1", "rouge2", "rougeL")
    
    # Semantic similarity configuration
    similarity_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    device: Optional[str] = None
    
    # Perplexity configuration
    perplexity_model: str = "gpt2"
    max_length: int = 512
    batch_size: int = 32
    cache_dir: Optional[str] = None

class TextMetrics:
    """Text evaluation metrics implementation.
    
    This class provides implementations of various NLP evaluation metrics including
    BLEU, ROUGE, perplexity, and semantic similarity metrics.
    """
    
    def __init__(
        self,
        config: Optional[TextMetricConfig] = None,
        **kwargs: Any
    ):
        """Initialize text metrics calculator.
        
        Args:
            config: Metric configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or TextMetricConfig(**kwargs)
        
        # Set device
        if self.config.device is None:
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize required models."""
        try:
            # Initialize semantic similarity model
            self.similarity_model = SentenceTransformer(
                self.config.similarity_model,
                device=self.config.device
            )
            
            # Initialize perplexity model
            self.perplexity_tokenizer = AutoTokenizer.from_pretrained(
                self.config.perplexity_model,
                cache_dir=self.config.cache_dir
            )
            self.perplexity_model = AutoModelForCausalLM.from_pretrained(
                self.config.perplexity_model,
                cache_dir=self.config.cache_dir
            ).to(self.config.device)
            
            # Initialize ROUGE scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                self.config.rouge_metrics,
                use_stemmer=True
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise
    
    def compute_bleu(
        self,
        reference: str,
        prediction: str
    ) -> Dict[str, float]:
        """Compute BLEU score.
        
        Args:
            reference: Reference text
            prediction: Predicted text
            
        Returns:
            Dictionary containing BLEU scores
        """
        # Tokenize
        reference_tokens = [reference.split()]
        prediction_tokens = prediction.split()
        
        # Compute BLEU score
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            reference_tokens,
            prediction_tokens,
            weights=self.config.bleu_weights,
            smoothing_function=smoothing
        )
        
        return {
            "bleu": score,
            "weights": self.config.bleu_weights
        }
    
    def compute_rouge(
        self,
        reference: str,
        prediction: str
    ) -> Dict[str, Dict[str, float]]:
        """Compute ROUGE scores.
        
        Args:
            reference: Reference text
            prediction: Predicted text
            
        Returns:
            Dictionary containing ROUGE scores
        """
        # Compute ROUGE scores
        scores = self.rouge_scorer.score(reference, prediction)
        
        # Format scores
        return {
            metric: {
                "precision": score.precision,
                "recall": score.recall,
                "fmeasure": score.fmeasure
            }
            for metric, score in scores.items()
        }
    
    def compute_perplexity(
        self,
        text: str
    ) -> float:
        """Compute perplexity score.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        try:
            # Tokenize
            encodings = self.perplexity_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)
            
            # Compute loss
            with torch.no_grad():
                outputs = self.perplexity_model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss
            
            # Compute perplexity
            perplexity = torch.exp(loss).item()
            
            return perplexity
            
        except Exception as e:
            logger.error(f"Perplexity computation failed: {str(e)}")
            raise
    
    def compute_semantic_similarity(
        self,
        text1: str,
        text2: str
    ) -> Dict[str, float]:
        """Compute semantic similarity score.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary containing similarity scores
        """
        try:
            # Encode texts
            embeddings1 = self.similarity_model.encode(
                text1,
                convert_to_tensor=True
            )
            embeddings2 = self.similarity_model.encode(
                text2,
                convert_to_tensor=True
            )
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                embeddings1.unsqueeze(0),
                embeddings2.unsqueeze(0)
            ).item()
            
            return {
                "similarity": similarity,
                "is_similar": similarity >= self.config.similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {str(e)}")
            raise
    
    def compute_all_metrics(
        self,
        reference: str,
        prediction: str
    ) -> Dict[str, Any]:
        """Compute all text metrics.
        
        Args:
            reference: Reference text
            prediction: Predicted text
            
        Returns:
            Dictionary containing all metrics
        """
        return {
            "bleu": self.compute_bleu(reference, prediction),
            "rouge": self.compute_rouge(reference, prediction),
            "perplexity": self.compute_perplexity(prediction),
            "semantic_similarity": self.compute_semantic_similarity(
                reference,
                prediction
            )
        }
    
    def batch_compute_metrics(
        self,
        references: List[str],
        predictions: List[str]
    ) -> Dict[str, Any]:
        """Compute metrics for a batch of texts.
        
        Args:
            references: List of reference texts
            predictions: List of predicted texts
            
        Returns:
            Dictionary containing batch metrics
        """
        results = []
        total_bleu = 0.0
        total_rouge = {metric: 0.0 for metric in self.config.rouge_metrics}
        total_perplexity = 0.0
        total_similarity = 0.0
        
        # Process each pair
        for ref, pred in zip(references, predictions):
            # Compute metrics
            metrics = self.compute_all_metrics(ref, pred)
            results.append(metrics)
            
            # Update totals
            total_bleu += metrics["bleu"]["bleu"]
            for metric in self.config.rouge_metrics:
                total_rouge[metric] += metrics["rouge"][metric]["fmeasure"]
            total_perplexity += metrics["perplexity"]
            total_similarity += metrics["semantic_similarity"]["similarity"]
        
        # Compute averages
        num_samples = len(references)
        return {
            "average_bleu": total_bleu / num_samples,
            "average_rouge": {
                metric: score / num_samples
                for metric, score in total_rouge.items()
            },
            "average_perplexity": total_perplexity / num_samples,
            "average_similarity": total_similarity / num_samples,
            "results": results
        }
    
    def compare_predictions(
        self,
        reference: str,
        prediction1: str,
        prediction2: str
    ) -> Dict[str, Any]:
        """Compare two predictions against a reference.
        
        Args:
            reference: Reference text
            prediction1: First prediction
            prediction2: Second prediction
            
        Returns:
            Dictionary containing comparison results
        """
        # Compute metrics for both predictions
        metrics1 = self.compute_all_metrics(reference, prediction1)
        metrics2 = self.compute_all_metrics(reference, prediction2)
        
        # Compute improvements
        improvements = {
            "bleu": metrics2["bleu"]["bleu"] - metrics1["bleu"]["bleu"],
            "rouge": {
                metric: (
                    metrics2["rouge"][metric]["fmeasure"] -
                    metrics1["rouge"][metric]["fmeasure"]
                )
                for metric in self.config.rouge_metrics
            },
            "perplexity": metrics1["perplexity"] - metrics2["perplexity"],  # Lower is better
            "similarity": (
                metrics2["semantic_similarity"]["similarity"] -
                metrics1["semantic_similarity"]["similarity"]
            )
        }
        
        return {
            "metrics1": metrics1,
            "metrics2": metrics2,
            "improvements": improvements
        } 