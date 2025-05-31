"""Regex evaluator implementation."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Union

from eval_framework.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

class MatchType(Enum):
    """Types of regex matching."""
    EXACT = "exact"           # Exact pattern match
    PARTIAL = "partial"       # Partial pattern match
    CAPTURE = "capture"       # Capture group match
    MULTIPLE = "multiple"     # Multiple pattern matches

@dataclass
class ScoringRule:
    """Rule for scoring regex matches."""
    pattern: str
    score: float
    weight: float = 1.0
    required: bool = False
    capture_groups: Optional[List[str]] = None
    transform: Optional[Callable[[str], Any]] = None

@dataclass
class RegexConfig:
    """Configuration for regex evaluation."""
    patterns: List[Union[str, ScoringRule]]
    match_type: MatchType = MatchType.EXACT
    case_sensitive: bool = False
    multiline: bool = False
    dotall: bool = False
    ignore_whitespace: bool = False
    max_matches: Optional[int] = None
    min_matches: Optional[int] = None
    default_score: float = 0.0
    default_weight: float = 1.0

class RegexEvaluator(BaseEvaluator):
    """Evaluator that uses regex patterns to score model outputs.
    
    This evaluator provides flexible pattern matching and scoring based on
    regex patterns, with support for capture groups and custom scoring rules.
    """
    
    def __init__(
        self,
        config: Optional[RegexConfig] = None,
        **kwargs: Any
    ):
        """Initialize regex evaluator.
        
        Args:
            config: Evaluation configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or RegexConfig(**kwargs)
        
        # Convert string patterns to ScoringRules
        self._rules: List[ScoringRule] = []
        for pattern in self.config.patterns:
            if isinstance(pattern, str):
                self._rules.append(ScoringRule(
                    pattern=pattern,
                    score=self.config.default_score,
                    weight=self.config.default_weight
                ))
            else:
                self._rules.append(pattern)
        
        # Compile regex patterns
        self._compiled_patterns: List[Pattern] = []
        flags = 0
        if not self.config.case_sensitive:
            flags |= re.IGNORECASE
        if self.config.multiline:
            flags |= re.MULTILINE
        if self.config.dotall:
            flags |= re.DOTALL
        
        for rule in self._rules:
            try:
                pattern = re.compile(rule.pattern, flags)
                self._compiled_patterns.append(pattern)
            except re.error as e:
                logger.error(f"Invalid regex pattern: {rule.pattern}")
                raise ValueError(f"Invalid regex pattern: {str(e)}")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if self.config.ignore_whitespace:
            # Remove extra whitespace
            text = ' '.join(text.split())
        return text
    
    def _extract_captures(
        self,
        pattern: Pattern,
        text: str,
        group_names: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Extract capture groups from text.
        
        Args:
            pattern: Compiled regex pattern
            text: Text to match
            group_names: Optional list of group names
            
        Returns:
            Dictionary of captured groups
        """
        match = pattern.search(text)
        if not match:
            return {}
        
        # Get all groups
        groups = match.groups()
        
        # If group names provided, use them
        if group_names:
            return {
                name: value
                for name, value in zip(group_names, groups)
                if value is not None
            }
        
        # Otherwise use group numbers
        return {
            str(i + 1): value
            for i, value in enumerate(groups)
            if value is not None
        }
    
    def _apply_transform(
        self,
        value: str,
        transform: Optional[Callable[[str], Any]]
    ) -> Any:
        """Apply transformation to captured value.
        
        Args:
            value: Captured value
            transform: Transformation function
            
        Returns:
            Transformed value
        """
        if transform is None:
            return value
        try:
            return transform(value)
        except Exception as e:
            logger.warning(f"Transform failed: {str(e)}")
            return value
    
    def _score_matches(
        self,
        text: str,
        pattern: Pattern,
        rule: ScoringRule
    ) -> Dict[str, Any]:
        """Score matches for a single pattern.
        
        Args:
            text: Text to match
            pattern: Compiled regex pattern
            rule: Scoring rule
            
        Returns:
            Dictionary containing match results
        """
        # Find all matches
        matches = list(pattern.finditer(text))
        
        if not matches:
            return {
                "matched": False,
                "score": 0.0,
                "weight": rule.weight,
                "required": rule.required,
                "matches": [],
                "captures": {}
            }
        
        # Extract captures if needed
        captures = {}
        if rule.capture_groups:
            for match in matches:
                match_captures = self._extract_captures(
                    pattern,
                    match.group(0),
                    rule.capture_groups
                )
                # Apply transformations
                if rule.transform:
                    match_captures = {
                        k: self._apply_transform(v, rule.transform)
                        for k, v in match_captures.items()
                    }
                captures.update(match_captures)
        
        return {
            "matched": True,
            "score": rule.score,
            "weight": rule.weight,
            "required": rule.required,
            "matches": [m.group(0) for m in matches],
            "captures": captures
        }
    
    def evaluate(
        self,
        prediction: str,
        reference: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Evaluate model prediction using regex patterns.
        
        Args:
            prediction: Model prediction
            reference: Optional reference text
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        # Normalize text
        text = self._normalize_text(prediction)
        
        # Score each pattern
        pattern_results = []
        total_score = 0.0
        total_weight = 0.0
        all_captures = {}
        
        for pattern, rule in zip(self._compiled_patterns, self._rules):
            result = self._score_matches(text, pattern, rule)
            pattern_results.append(result)
            
            if result["matched"]:
                total_score += result["score"] * result["weight"]
                total_weight += result["weight"]
                all_captures.update(result["captures"])
        
        # Calculate weighted score
        weighted_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Check required patterns
        missing_required = any(
            result["required"] and not result["matched"]
            for result in pattern_results
        )
        
        # Check match count constraints
        match_count = sum(len(result["matches"]) for result in pattern_results)
        if self.config.min_matches and match_count < self.config.min_matches:
            weighted_score = 0.0
        if self.config.max_matches and match_count > self.config.max_matches:
            weighted_score *= self.config.max_matches / match_count
        
        return {
            "score": weighted_score,
            "prediction": prediction,
            "reference": reference,
            "matched": not missing_required,
            "total_matches": match_count,
            "pattern_results": pattern_results,
            "captures": all_captures
        }
    
    def batch_evaluate(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Evaluate batch of predictions.
        
        Args:
            predictions: List of model predictions
            references: Optional list of reference texts
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing batch evaluation results
        """
        results = []
        total_score = 0.0
        total_matches = 0
        
        for i, pred in enumerate(predictions):
            # Get reference if available
            ref = references[i] if references else None
            
            # Evaluate single prediction
            result = self.evaluate(
                prediction=pred,
                reference=ref,
                **kwargs
            )
            
            results.append(result)
            total_score += result["score"]
            total_matches += result["total_matches"]
        
        # Calculate aggregate metrics
        num_samples = len(predictions)
        avg_score = total_score / num_samples if num_samples > 0 else 0.0
        avg_matches = total_matches / num_samples if num_samples > 0 else 0.0
        
        return {
            "average_score": avg_score,
            "average_matches": avg_matches,
            "total_score": total_score,
            "total_matches": total_matches,
            "total_samples": num_samples,
            "results": results
        } 