from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
import numpy as np

class ResultPlotter:
    """Handles visualization of benchmark results."""
    
    def __init__(self, output_dir: str = "results/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_model_comparison(self, results: Dict[str, Dict[str, Any]], 
                            metric: str = "total_responses"):
        """
        Create comparison plot across models.
        
        Args:
            results: Dictionary of results by model name
            metric: Metric to plot (total_responses, average_coherence, etc.)
        """
        plt.figure(figsize=(10, 6))
        
        models = list(results.keys())
        values = [results[model][metric] for model in models]
        
        sns.barplot(x=models, y=values)
        plt.title(f"Model Comparison - {metric.replace('_', ' ').title()}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"model_comparison_{metric}.png")
        plt.close()
        
    def plot_question_breakdown(self, results: Dict[str, Dict[str, Any]], 
                              model_name: str):
        """
        Create detailed breakdown for a specific model.
        
        Args:
            results: Results dictionary
            model_name: Name of model to analyze
        """
        model_results = results[model_name]
        
        # Create question-level metrics
        questions = list(model_results.keys())
        response_counts = [r['total_responses'] for r in model_results.values()]
        avg_coherence = [np.mean(r['coherence_scores']) for r in model_results.values()]
        avg_novelty = [np.mean(r['novelty_scores']) for r in model_results.values()]
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot response counts
        sns.barplot(x=questions, y=response_counts, ax=ax1)
        ax1.set_title("Responses per Question")
        ax1.set_xticklabels(questions, rotation=45)
        
        # Plot coherence scores
        sns.barplot(x=questions, y=avg_coherence, ax=ax2)
        ax2.set_title("Average Coherence per Question")
        ax2.set_xticklabels(questions, rotation=45)
        
        # Plot novelty scores
        sns.barplot(x=questions, y=avg_novelty, ax=ax3)
        ax3.set_title("Average Novelty per Question")
        ax3.set_xticklabels(questions, rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"question_breakdown_{model_name}.png")
        plt.close()
        
    def save_results(self, results: Dict[str, Dict[str, Any]], 
                    filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        with open(self.output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2) 