import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

from core.evaluator import Evaluator, EvaluationConfig
from metrics.scoring import Scorer, ScoringConfig
from visualization.plotter import ResultPlotter

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_questions(filepath: str) -> List[str]:
    """Load questions from file."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description='Run language model benchmark')
    parser.add_argument('--questions', type=str, required=True,
                       help='Path to questions file')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='List of model names to evaluate')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for generation')
    parser.add_argument('--coherence-threshold', type=float, default=15.0,
                       help='Minimum coherence score threshold')
    parser.add_argument('--novelty-threshold', type=float, default=0.15,
                       help='Minimum novelty score threshold')
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load questions
    questions = load_questions(args.questions)
    logger.info(f"Loaded {len(questions)} questions")
    
    # Initialize components
    eval_config = EvaluationConfig(
        temperature=args.temperature,
        coherence_threshold=args.coherence_threshold,
        novelty_threshold=args.novelty_threshold
    )
    scorer_config = ScoringConfig()
    
    evaluator = Evaluator(eval_config)
    scorer = Scorer(scorer_config)
    plotter = ResultPlotter(args.output_dir)
    
    # Run evaluation for each model
    results = {}
    for model_name in args.models:
        logger.info(f"Evaluating model: {model_name}")
        model_results = evaluator.evaluate_model(model_name, questions)
        results[model_name] = model_results
        
        # Generate visualizations
        plotter.plot_question_breakdown(results, model_name)
    
    # Generate comparison plots
    plotter.plot_model_comparison(results, "total_responses")
    plotter.plot_model_comparison(results, "average_coherence")
    plotter.plot_model_comparison(results, "average_novelty")
    
    # Save results
    plotter.save_results(results)
    logger.info("Evaluation complete")

if __name__ == '__main__':
    main() 