"""
Custom evaluator example for the evaluation framework.
This example demonstrates how to:
1. Create a custom evaluator
2. Register and use the custom evaluator
3. Compare results with standard evaluators
"""

import os
from pathlib import Path
from eval_framework.core.engine import EvaluationEngine
from eval_framework.core.base import BaseEvaluator
from eval_framework.evaluators.exact_match import ExactMatchEvaluator
from eval_framework.metrics.rouge import ROUGEMetric

class SemanticSimilarityEvaluator(BaseEvaluator):
    """Custom evaluator that uses semantic similarity for evaluation."""
    
    def __init__(self, similarity_threshold=0.7, **kwargs):
        super().__init__(**kwargs)
        self.similarity_threshold = similarity_threshold
        self.similarity_model = None
    
    def initialize(self):
        """Initialize the similarity model."""
        from sentence_transformers import SentenceTransformer
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate(self, predictions, references):
        """Evaluate predictions using semantic similarity."""
        if not self.similarity_model:
            self.initialize()
        
        # Calculate embeddings
        pred_embeddings = self.similarity_model.encode(predictions)
        ref_embeddings = self.similarity_model.encode(references)
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(pred_embeddings, ref_embeddings).diagonal()
        
        # Calculate metrics
        exact_matches = sum(1 for s in similarities if s >= self.similarity_threshold)
        avg_similarity = similarities.mean()
        
        return {
            "semantic_similarity": avg_similarity,
            "semantic_matches": exact_matches / len(predictions)
        }

def main():
    # Create output directory
    output_dir = Path("output/custom_evaluator")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure evaluation with both standard and custom evaluators
    config = {
        "model": {
            "type": "text",
            "name": "gpt2",
            "batch_size": 4,
            "device": "cpu"
        },
        "dataset": {
            "type": "text",
            "name": "test_dataset",
            "split": "test",
            "max_samples": 100,
            "batch_size": 4
        },
        "evaluators": [
            {
                "type": "exact_match",
                "normalize_text": True,
                "case_sensitive": False
            },
            {
                "type": "semantic_similarity",
                "similarity_threshold": 0.7
            }
        ],
        "metrics": [
            {
                "name": "rouge",
                "type": "rouge",
                "metrics": ["rouge1", "rouge2", "rougeL"],
                "use_stemmer": True
            }
        ],
        "output": {
            "formats": ["json", "html"],
            "save_predictions": True,
            "save_metrics": True,
            "save_error_analysis": True,
            "output_dir": str(output_dir)
        }
    }
    
    # Initialize evaluation engine
    engine = EvaluationEngine()
    
    # Register custom evaluator
    engine.register_evaluator("semantic_similarity", SemanticSimilarityEvaluator)
    
    # Define progress callback
    def progress_callback(current, total, message):
        print(f"Progress: {current}/{total} - {message}")
    
    # Run evaluation
    print("Starting evaluation with custom evaluator...")
    results = engine.evaluate(config, progress_callback=progress_callback)
    
    # Display results
    print("\nEvaluation Results:")
    print("------------------")
    print("\nExact Match Results:")
    for metric_name, score in results["evaluators"]["exact_match"].items():
        print(f"  {metric_name}: {score:.4f}")
    
    print("\nSemantic Similarity Results:")
    for metric_name, score in results["evaluators"]["semantic_similarity"].items():
        print(f"  {metric_name}: {score:.4f}")
    
    print(f"\nTotal samples evaluated: {len(results['predictions'])}")
    print(f"Results saved to: {output_dir}")
    
    # Display sample predictions with both metrics
    print("\nSample Predictions:")
    print("------------------")
    for i, (pred, ref) in enumerate(zip(results["predictions"][:3], results["references"][:3])):
        print(f"\nSample {i+1}:")
        print(f"Reference: {ref}")
        print(f"Prediction: {pred}")
        print(f"Exact Match: {results['evaluators']['exact_match']['exact_match'][i]}")
        print(f"Semantic Similarity: {results['evaluators']['semantic_similarity']['semantic_similarity'][i]:.4f}")

if __name__ == "__main__":
    main() 