#!/usr/bin/env python3

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    fig.suptitle('Score Distributions', fontsize=16, y=1.05)
    
    # Plot distributions with consistent styling
    for ax, score, title in zip([ax1, ax2, ax3], 
                              ['ahns_score', 'harmony_score', 'novelty_score'],
                              ['AHNS Score', 'Harmony Score', 'Novelty Score']):
        sns.histplot(data=df, x=score, ax=ax, kde=True, color='skyblue', edgecolor='black')
        ax.set_title(title, pad=10)
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)  # All scores are between 0 and 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_score_correlation_plot(df: pd.DataFrame, output_dir: str):
    """Create correlation plot between different scores."""
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Select score columns and calculate correlation
    score_cols = ["ahns_score", "harmony_score", "novelty_score"]
    corr_matrix = df[score_cols].corr()
    
    # Create figure with proper size
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with improved styling
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap="coolwarm", 
                vmin=-1, 
                vmax=1,
                fmt='.2f',  # Format correlation values to 2 decimal places
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Score Correlations', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_correlations.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_prompt_comparison_plot(df: pd.DataFrame, output_dir: str):
    """Create box plots comparing scores across different prompts."""
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with proper size
    plt.figure(figsize=(12, 8))
    
    # Melt the dataframe
    melted_df = pd.melt(df, 
                        id_vars=["prompt"],
                        value_vars=["ahns_score", "harmony_score", "novelty_score"],
                        var_name="score_type",
                        value_name="score")
    
    # Create box plot with improved styling
    ax = sns.boxplot(data=melted_df, 
                    x="prompt", 
                    y="score", 
                    hue="score_type",
                    palette="Set2")
    
    # Improve readability
    plt.xticks(rotation=45, ha='right')
    plt.title('Score Distribution by Prompt', pad=20)
    plt.xlabel('Prompt')
    plt.ylabel('Score')
    
    # Improve legend
    plt.legend(title='Score Type', 
              bbox_to_anchor=(1.05, 1), 
              loc='upper left')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prompt_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_3d_plot(df: pd.DataFrame, output_dir: str):
    """Create an interactive 3D scatter plot of the scores."""
    fig = px.scatter_3d(df,
                        x="ahns_score",
                        y="harmony_score",
                        z="novelty_score",
                        color="prompt",
                        title="3D Score Distribution",
                        labels={
                            "ahns_score": "AHNS Score",
                            "harmony_score": "Harmony Score",
                            "novelty_score": "Novelty Score"
                        })
    
    fig.write_html(os.path.join(output_dir, "interactive_3d_plot.html"))

def create_summary_dashboard(df: pd.DataFrame, output_dir: str):
    """Create a comprehensive dashboard with multiple plots."""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Score Distributions", "Score Correlations",
                       "Prompt Comparison", "Score Relationships")
    )
    
    # Add histograms
    for i, score in enumerate(["ahns_score", "harmony_score", "novelty_score"]):
        fig.add_trace(
            go.Histogram(x=df[score], name=score),
            row=1, col=1
        )
    
    # Add correlation heatmap
    corr_matrix = df[["ahns_score", "harmony_score", "novelty_score"]].corr()
    fig.add_trace(
        go.Heatmap(z=corr_matrix.values,
                  x=corr_matrix.columns,
                  y=corr_matrix.columns),
        row=1, col=2
    )
    
    # Add box plots
    for i, score in enumerate(["ahns_score", "harmony_score", "novelty_score"]):
        fig.add_trace(
            go.Box(y=df[score], name=score),
            row=2, col=1
        )
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(x=df["ahns_score"],
                  y=df["harmony_score"],
                  mode="markers",
                  name="AHNS vs Harmony"),
        row=2, col=2
    )
    
    fig.update_layout(height=1000, width=1200, title_text="Evaluation Results Dashboard")
    fig.write_html(os.path.join(output_dir, "interactive_dashboard.html"))

def main():
    # Create visualization directory if it doesn't exist
    vis_dir = Path("visualization")
    vis_dir.mkdir(exist_ok=True)
    
    # Create output directory for plots
    output_dir = vis_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Load the latest results
    df = load_latest_results()
    
    # Generate all visualizations
    create_score_distribution_plot(df, str(output_dir))
    create_score_correlation_plot(df, str(output_dir))
    create_prompt_comparison_plot(df, str(output_dir))
    create_interactive_3d_plot(df, str(output_dir))
    create_summary_dashboard(df, str(output_dir))
    
    print(f"Visualizations have been saved to {output_dir}")

if __name__ == "__main__":
    main() 