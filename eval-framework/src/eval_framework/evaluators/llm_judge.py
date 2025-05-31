"""LLM judge evaluator implementation."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from eval_framework.evaluators.base import BaseEvaluator
from eval_framework.models.base import BaseModel

logger = logging.getLogger(__name__)

class ScoringMethod(Enum):
    """Methods for scoring model outputs."""
    DIRECT = "direct"           # Direct scoring (e.g., 1-10)
    COMPARATIVE = "comparative" # Comparative scoring (better/worse)
    RUBRIC = "rubric"          # Rubric-based scoring

@dataclass
class LLMJudgeConfig:
    """Configuration for LLM judge evaluation."""
    judge_model: BaseModel
    scoring_method: ScoringMethod = ScoringMethod.DIRECT
    score_range: tuple[int, int] = (1, 10)
    prompt_template: Optional[str] = None
    bias_mitigation: bool = True
    calibration_samples: int = 5
    temperature: float = 0.0
    max_retries: int = 3
    extract_score: bool = True
    normalize_scores: bool = True

class LLMJudgeEvaluator(BaseEvaluator):
    """Evaluator that uses an LLM as a judge to score model outputs.
    
    This evaluator uses a reference model to score model outputs based on
    various criteria, with built-in bias mitigation and robust score extraction.
    """
    
    # Default prompt templates for different scoring methods
    DEFAULT_PROMPTS = {
        ScoringMethod.DIRECT: """
You are an expert evaluator. Please score the following model output on a scale of {min_score} to {max_score}.
Consider the following criteria:
1. Accuracy: Is the information correct and factual?
2. Completeness: Does it fully address the question?
3. Clarity: Is it well-written and easy to understand?
4. Relevance: Is it directly related to the question?

Question: {question}
Reference Answer: {reference}
Model Output: {prediction}

Provide your score as a single number between {min_score} and {max_score}.
Score:""",
        
        ScoringMethod.COMPARATIVE: """
You are an expert evaluator. Please compare the following model output with the reference answer.
Consider the following criteria:
1. Accuracy: Is the information correct and factual?
2. Completeness: Does it fully address the question?
3. Clarity: Is it well-written and easy to understand?
4. Relevance: Is it directly related to the question?

Question: {question}
Reference Answer: {reference}
Model Output: {prediction}

Is the model output:
A) Significantly better than the reference
B) Slightly better than the reference
C) About the same as the reference
D) Slightly worse than the reference
E) Significantly worse than the reference

Provide your evaluation as a single letter (A-E).
Evaluation:""",
        
        ScoringMethod.RUBRIC: """
You are an expert evaluator. Please evaluate the following model output using the provided rubric.
Consider each criterion and provide a score for each.

Question: {question}
Reference Answer: {reference}
Model Output: {prediction}

Rubric:
1. Accuracy (1-10): Is the information correct and factual?
2. Completeness (1-10): Does it fully address the question?
3. Clarity (1-10): Is it well-written and easy to understand?
4. Relevance (1-10): Is it directly related to the question?

Provide your scores as a comma-separated list of numbers (e.g., "8,7,9,8").
Scores:"""
    }
    
    def __init__(
        self,
        config: Optional[LLMJudgeConfig] = None,
        **kwargs: Any
    ):
        """Initialize LLM judge evaluator.
        
        Args:
            config: Evaluation configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or LLMJudgeConfig(**kwargs)
        
        # Set default prompt template if not provided
        if not self.config.prompt_template:
            self.config.prompt_template = self.DEFAULT_PROMPTS[self.config.scoring_method]
        
        # Compile regex patterns for score extraction
        self._score_patterns = {
            ScoringMethod.DIRECT: re.compile(r'(?:score|rating|grade)[\s:]+(\d+)', re.IGNORECASE),
            ScoringMethod.COMPARATIVE: re.compile(r'[A-E]'),
            ScoringMethod.RUBRIC: re.compile(r'(\d+(?:,\s*\d+)*)')
        }
        
        # Initialize calibration if bias mitigation is enabled
        if self.config.bias_mitigation:
            self._calibrate_judge()
    
    def _calibrate_judge(self) -> None:
        """Calibrate the judge model using sample evaluations."""
        logger.info("Calibrating judge model...")
        
        # Sample calibration data
        calibration_data = [
            {
                "question": "What is 2+2?",
                "reference": "4",
                "prediction": "4",
                "expected_score": self.config.score_range[1]  # Perfect score
            },
            {
                "question": "What is the capital of France?",
                "reference": "Paris",
                "prediction": "London",
                "expected_score": self.config.score_range[0]  # Minimum score
            }
        ]
        
        # Add more calibration samples if needed
        while len(calibration_data) < self.config.calibration_samples:
            calibration_data.append({
                "question": "Sample question",
                "reference": "Sample reference",
                "prediction": "Sample prediction",
                "expected_score": (self.config.score_range[0] + self.config.score_range[1]) // 2
            })
        
        # Run calibration evaluations
        self._calibration_scores = []
        for sample in calibration_data:
            score = self._evaluate_single(
                question=sample["question"],
                prediction=sample["prediction"],
                reference=sample["reference"]
            )
            self._calibration_scores.append(score)
        
        logger.info("Judge model calibration complete")
    
    def _extract_score(self, text: str) -> Optional[float]:
        """Extract score from judge's response.
        
        Args:
            text: Judge's response text
            
        Returns:
            Extracted score or None if not found
        """
        if not self.config.extract_score:
            return None
        
        # Get appropriate pattern for scoring method
        pattern = self._score_patterns[self.config.scoring_method]
        
        # Try to find score
        match = pattern.search(text)
        if not match:
            return None
        
        # Extract and process score based on method
        if self.config.scoring_method == ScoringMethod.DIRECT:
            score = float(match.group(1))
        elif self.config.scoring_method == ScoringMethod.COMPARATIVE:
            # Convert letter grade to score
            letter = match.group(0)
            score = {
                'A': self.config.score_range[1],
                'B': (self.config.score_range[0] + self.config.score_range[1]) * 0.75,
                'C': (self.config.score_range[0] + self.config.score_range[1]) * 0.5,
                'D': (self.config.score_range[0] + self.config.score_range[1]) * 0.25,
                'E': self.config.score_range[0]
            }[letter]
        else:  # RUBRIC
            # Calculate average of rubric scores
            scores = [float(s) for s in match.group(1).split(',')]
            score = sum(scores) / len(scores)
        
        # Normalize score if needed
        if self.config.normalize_scores:
            min_score, max_score = self.config.score_range
            score = (score - min_score) / (max_score - min_score)
        
        return score
    
    def _evaluate_single(
        self,
        question: str,
        prediction: str,
        reference: str
    ) -> float:
        """Evaluate a single prediction.
        
        Args:
            question: Question text
            prediction: Model prediction
            reference: Reference answer
            
        Returns:
            Evaluation score
        """
        # Format prompt
        prompt = self.config.prompt_template.format(
            question=question,
            prediction=prediction,
            reference=reference,
            min_score=self.config.score_range[0],
            max_score=self.config.score_range[1]
        )
        
        # Get judge's response
        for attempt in range(self.config.max_retries):
            try:
                response = self.config.judge_model.generate(prompt)
                score = self._extract_score(response)
                
                if score is not None:
                    # Apply bias mitigation if enabled
                    if self.config.bias_mitigation:
                        # Adjust score based on calibration
                        calibration_mean = sum(self._calibration_scores) / len(self._calibration_scores)
                        score = score * (self.config.score_range[1] / calibration_mean)
                    
                    return score
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        raise ValueError("Could not extract valid score from judge's response")
    
    def evaluate(
        self,
        prediction: str,
        reference: str,
        question: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Evaluate model prediction.
        
        Args:
            prediction: Model prediction
            reference: Reference answer
            question: Optional question text
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        # Get score from judge
        score = self._evaluate_single(
            question=question or "",
            prediction=prediction,
            reference=reference
        )
        
        return {
            "score": score,
            "prediction": prediction,
            "reference": reference,
            "question": question,
            "normalized_score": score if self.config.normalize_scores else None
        }
    
    def batch_evaluate(
        self,
        predictions: List[str],
        references: List[str],
        questions: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Evaluate batch of predictions.
        
        Args:
            predictions: List of model predictions
            references: List of reference answers
            questions: Optional list of questions
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing batch evaluation results
        """
        results = []
        total_score = 0.0
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Get question if available
            question = questions[i] if questions else None
            
            # Evaluate single prediction
            result = self.evaluate(
                prediction=pred,
                reference=ref,
                question=question,
                **kwargs
            )
            
            results.append(result)
            total_score += result["score"]
        
        # Calculate aggregate metrics
        num_samples = len(predictions)
        avg_score = total_score / num_samples if num_samples > 0 else 0.0
        
        return {
            "average_score": avg_score,
            "total_score": total_score,
            "total_samples": num_samples,
            "results": results
        } 