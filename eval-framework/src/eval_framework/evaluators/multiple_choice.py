"""Multiple choice evaluator implementation."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from difflib import SequenceMatcher
from thefuzz import fuzz

from eval_framework.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

class AnswerFormat(Enum):
    """Supported answer formats."""
    LETTER = "letter"  # A, B, C, D
    NUMBER = "number"  # 1, 2, 3, 4
    TEXT = "text"      # Full text of answer

@dataclass
class MultipleChoiceConfig:
    """Configuration for multiple choice evaluation."""
    answer_format: AnswerFormat = AnswerFormat.LETTER
    fuzzy_threshold: float = 0.8
    extract_answer: bool = True
    normalize_text: bool = True
    case_sensitive: bool = False

class MultipleChoiceEvaluator(BaseEvaluator):
    """Evaluator for multiple choice questions.
    
    This evaluator handles various answer formats and provides fuzzy matching
    for robust evaluation of model outputs.
    """
    
    def __init__(
        self,
        config: Optional[MultipleChoiceConfig] = None,
        **kwargs: Any
    ):
        """Initialize multiple choice evaluator.
        
        Args:
            config: Evaluation configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or MultipleChoiceConfig(**kwargs)
        
        # Compile regex patterns
        self._letter_pattern = re.compile(r'\b([A-D])\b', re.IGNORECASE)
        self._number_pattern = re.compile(r'\b([1-4])\b')
        self._answer_pattern = re.compile(
            r'(?:answer|correct|right|choice|option)[\s:]+([A-D1-4])',
            re.IGNORECASE
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not self.config.normalize_text:
            return text
        
        # Convert to lowercase if not case sensitive
        if not self.config.case_sensitive:
            text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def _extract_answer(
        self,
        text: str,
        choices: Optional[List[str]] = None
    ) -> Optional[str]:
        """Extract answer from model output.
        
        Args:
            text: Model output text
            choices: Optional list of answer choices
            
        Returns:
            Extracted answer or None if not found
        """
        if not self.config.extract_answer:
            return text
        
        # Try to find explicit answer marker
        answer_match = self._answer_pattern.search(text)
        if answer_match:
            return answer_match.group(1)
        
        # Try to find letter answer
        letter_match = self._letter_pattern.search(text)
        if letter_match:
            return letter_match.group(1).upper()
        
        # Try to find number answer
        number_match = self._number_pattern.search(text)
        if number_match:
            num = int(number_match.group(1))
            if 1 <= num <= 4:
                return chr(ord('A') + num - 1)
        
        # If choices provided, try fuzzy matching
        if choices:
            # Find the choice that appears in the text
            for i, choice in enumerate(choices):
                if choice.lower() in text.lower():
                    return chr(ord('A') + i)
            
            # If no exact match, try fuzzy matching
            best_match = None
            best_score = 0
            
            for i, choice in enumerate(choices):
                score = fuzz.ratio(
                    self._normalize_text(choice),
                    self._normalize_text(text)
                )
                if score > best_score and score >= self.config.fuzzy_threshold * 100:
                    best_score = score
                    best_match = chr(ord('A') + i)
            
            return best_match
        
        return None
    
    def _convert_answer_format(
        self,
        answer: str,
        target_format: AnswerFormat
    ) -> str:
        """Convert answer to target format.
        
        Args:
            answer: Answer to convert
            target_format: Target format
            
        Returns:
            Converted answer
        """
        # Convert to letter format first
        if answer.isdigit() and 1 <= int(answer) <= 4:
            letter = chr(ord('A') + int(answer) - 1)
        elif answer.upper() in 'ABCD':
            letter = answer.upper()
        else:
            return answer
        
        # Convert to target format
        if target_format == AnswerFormat.LETTER:
            return letter
        elif target_format == AnswerFormat.NUMBER:
            return str(ord(letter) - ord('A') + 1)
        else:
            return answer
    
    def evaluate(
        self,
        prediction: str,
        reference: str,
        choices: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Evaluate model prediction against reference answer.
        
        Args:
            prediction: Model prediction
            reference: Reference answer
            choices: Optional list of answer choices
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        # Extract and normalize answers
        pred_answer = self._extract_answer(prediction, choices)
        ref_answer = self._convert_answer_format(
            reference,
            self.config.answer_format
        )
        
        if pred_answer is None:
            logger.warning("Could not extract answer from prediction")
            return {
                "correct": False,
                "prediction": prediction,
                "reference": ref_answer,
                "extracted_prediction": None,
                "confidence": 0.0
            }
        
        # Convert prediction to reference format
        pred_answer = self._convert_answer_format(
            pred_answer,
            self.config.answer_format
        )
        
        # Calculate exact match
        exact_match = pred_answer == ref_answer
        
        # Calculate fuzzy match if not exact
        fuzzy_score = 0.0
        if not exact_match and self.config.answer_format == AnswerFormat.TEXT:
            fuzzy_score = fuzz.ratio(
                self._normalize_text(pred_answer),
                self._normalize_text(ref_answer)
            ) / 100.0
        
        # Calculate confidence
        confidence = 1.0 if exact_match else fuzzy_score
        
        return {
            "correct": exact_match or fuzzy_score >= self.config.fuzzy_threshold,
            "prediction": prediction,
            "reference": ref_answer,
            "extracted_prediction": pred_answer,
            "exact_match": exact_match,
            "fuzzy_score": fuzzy_score,
            "confidence": confidence
        }
    
    def batch_evaluate(
        self,
        predictions: List[str],
        references: List[str],
        choices: Optional[List[List[str]]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Evaluate batch of predictions.
        
        Args:
            predictions: List of model predictions
            references: List of reference answers
            choices: Optional list of answer choices for each question
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing batch evaluation results
        """
        results = []
        total_correct = 0
        total_confidence = 0.0
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Get choices for this question if available
            question_choices = choices[i] if choices else None
            
            # Evaluate single prediction
            result = self.evaluate(
                prediction=pred,
                reference=ref,
                choices=question_choices,
                **kwargs
            )
            
            results.append(result)
            if result["correct"]:
                total_correct += 1
            total_confidence += result["confidence"]
        
        # Calculate aggregate metrics
        num_samples = len(predictions)
        accuracy = total_correct / num_samples if num_samples > 0 else 0.0
        avg_confidence = total_confidence / num_samples if num_samples > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "total_correct": total_correct,
            "total_samples": num_samples,
            "results": results
        } 