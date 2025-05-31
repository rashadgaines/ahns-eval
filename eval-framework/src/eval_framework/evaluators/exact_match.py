"""Exact match evaluator implementation."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from eval_framework.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

class NormalizationLevel(Enum):
    """Levels of text normalization."""
    NONE = "none"           # No normalization
    MINIMAL = "minimal"     # Basic whitespace normalization
    STANDARD = "standard"   # Standard normalization (whitespace + case)
    AGGRESSIVE = "aggressive"  # Aggressive normalization (whitespace + case + punctuation)

@dataclass
class ExactMatchConfig:
    """Configuration for exact match evaluation."""
    normalization: NormalizationLevel = NormalizationLevel.STANDARD
    case_sensitive: bool = False
    strip_whitespace: bool = True
    remove_punctuation: bool = False
    normalize_unicode: bool = True
    normalize_numbers: bool = False
    normalize_quotes: bool = True
    normalize_hyphens: bool = True
    normalize_spaces: bool = True

class ExactMatchEvaluator(BaseEvaluator):
    """Evaluator for exact string matching with flexible normalization options.
    
    This evaluator provides various levels of string normalization to handle
    common variations in text while still maintaining exact matching semantics.
    """
    
    def __init__(
        self,
        config: Optional[ExactMatchConfig] = None,
        **kwargs: Any
    ):
        """Initialize exact match evaluator.
        
        Args:
            config: Evaluation configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or ExactMatchConfig(**kwargs)
        
        # Compile regex patterns
        self._whitespace_pattern = re.compile(r'\s+')
        self._punctuation_pattern = re.compile(r'[^\w\s]')
        self._quote_pattern = re.compile(r'[''""]')
        self._hyphen_pattern = re.compile(r'[-–—]')
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text according to configuration.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if self.config.normalization == NormalizationLevel.NONE:
            return text
        
        # Convert to string if not already
        text = str(text)
        
        # Unicode normalization
        if self.config.normalize_unicode:
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
        
        # Case normalization
        if not self.config.case_sensitive:
            text = text.lower()
        
        # Whitespace normalization
        if self.config.normalize_spaces:
            text = self._whitespace_pattern.sub(' ', text)
        
        if self.config.strip_whitespace:
            text = text.strip()
        
        # Apply additional normalizations based on level
        if self.config.normalization in [NormalizationLevel.STANDARD, NormalizationLevel.AGGRESSIVE]:
            # Quote normalization
            if self.config.normalize_quotes:
                text = self._quote_pattern.sub("'", text)
            
            # Hyphen normalization
            if self.config.normalize_hyphens:
                text = self._hyphen_pattern.sub('-', text)
        
        if self.config.normalization == NormalizationLevel.AGGRESSIVE:
            # Punctuation removal
            if self.config.remove_punctuation:
                text = self._punctuation_pattern.sub('', text)
            
            # Number normalization
            if self.config.normalize_numbers:
                text = re.sub(r'\d+', '0', text)
        
        return text
    
    def evaluate(
        self,
        prediction: str,
        reference: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Evaluate model prediction against reference text.
        
        Args:
            prediction: Model prediction
            reference: Reference text
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        # Normalize texts
        norm_pred = self._normalize_text(prediction)
        norm_ref = self._normalize_text(reference)
        
        # Calculate exact match
        exact_match = norm_pred == norm_ref
        
        # Calculate character-level metrics
        pred_chars = set(norm_pred)
        ref_chars = set(norm_ref)
        
        # Calculate character overlap
        common_chars = pred_chars.intersection(ref_chars)
        char_overlap = len(common_chars) / len(ref_chars) if ref_chars else 0.0
        
        # Calculate length difference
        length_diff = abs(len(norm_pred) - len(norm_ref))
        length_ratio = min(len(norm_pred), len(norm_ref)) / max(len(norm_pred), len(norm_ref)) if max(len(norm_pred), len(norm_ref)) > 0 else 0.0
        
        return {
            "correct": exact_match,
            "prediction": prediction,
            "reference": reference,
            "normalized_prediction": norm_pred,
            "normalized_reference": norm_ref,
            "exact_match": exact_match,
            "char_overlap": char_overlap,
            "length_difference": length_diff,
            "length_ratio": length_ratio
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
        results = []
        total_correct = 0
        total_char_overlap = 0.0
        total_length_ratio = 0.0
        
        for pred, ref in zip(predictions, references):
            # Evaluate single prediction
            result = self.evaluate(
                prediction=pred,
                reference=ref,
                **kwargs
            )
            
            results.append(result)
            if result["correct"]:
                total_correct += 1
            total_char_overlap += result["char_overlap"]
            total_length_ratio += result["length_ratio"]
        
        # Calculate aggregate metrics
        num_samples = len(predictions)
        accuracy = total_correct / num_samples if num_samples > 0 else 0.0
        avg_char_overlap = total_char_overlap / num_samples if num_samples > 0 else 0.0
        avg_length_ratio = total_length_ratio / num_samples if num_samples > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "average_char_overlap": avg_char_overlap,
            "average_length_ratio": avg_length_ratio,
            "total_correct": total_correct,
            "total_samples": num_samples,
            "results": results
        } 