#!/usr/bin/env python3

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

def load_latest_results(results_dir: str = "results") -> pd.DataFrame:
    """Load the most recent evaluation results."""
    results_path = Path(results_dir)
    csv_files = list(results_path.glob("grok_eval_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No evaluation results found")
    
    # Get the most recent file
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    return pd.read_csv(latest_file)

def create_score_distribution_plot(df: pd.DataFrame, output_dir: str):
    """Create distribution plots for AHNS, Harmony, and Novelty scores."""
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create subplots with proper spacing
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Distribution of Image Quality Scores', fontsize=16, y=1.05)
    
    # Define colors and styles
    colors = ['#2ecc71', '#e74c3c', '#3498db']  # Green, Red, Blue
    titles = ['Overall Quality (AHNS)', 'Aesthetic Harmony', 'Creative Novelty']
    scores = ['ahns_score', 'harmony_score', 'novelty_score']
    
    # Plot distributions with improved styling
    for ax, score, title, color in zip([ax1, ax2, ax3], scores, titles, colors):
        # Create histogram with KDE
        sns.histplot(data=df, x=score, ax=ax, kde=True, color=color, edgecolor='black', alpha=0.6)
        
        # Add mean line
        mean_val = df[score].mean()
        std_val = df[score].std()
        ax.axvline(mean_val, color='black', linestyle='--', alpha=0.8)
        
        # Add annotations
        ax.text(0.95, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Improve labels and title
        ax.set_title(title, pad=10, fontsize=12)
        ax.set_xlabel('Score (0-1)', fontsize=10)
        ax.set_ylabel('Number of Images', fontsize=10)
        
        # Set consistent x-axis
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add score interpretation
        if score == 'ahns_score':
            ax.text(0.05, 0.95, 'Higher = Better Overall Quality',
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        elif score == 'harmony_score':
            ax.text(0.05, 0.95, 'Higher = Better Aesthetic Harmony',
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            ax.text(0.05, 0.95, 'Higher = More Creative/Unique',
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score--distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create visualization directory if it doesn't exist
    vis_dir = Path("visualization")
    vis_dir.mkdir(exist_ok=True)
    
    # Create output directory for plots
    output_dir = vis_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Load the latest results
    df = load_latest_results()
    
    # Generate visualization
    create_score_distribution_plot(df, str(output_dir))
    
    print(f"Visualization has been saved to {output_dir}")

if __name__ == "__main__":
    main() 