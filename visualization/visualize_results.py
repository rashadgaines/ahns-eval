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
    plt.figure(figsize=(15, 5))
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot distributions
    sns.histplot(data=df, x="ahns_score", ax=ax1, kde=True)
    sns.histplot(data=df, x="harmony_score", ax=ax2, kde=True)
    sns.histplot(data=df, x="novelty_score", ax=ax3, kde=True)
    
    # Set titles and labels
    ax1.set_title("AHNS Score Distribution")
    ax2.set_title("Harmony Score Distribution")
    ax3.set_title("Novelty Score Distribution")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"))
    plt.close()

def create_score_correlation_plot(df: pd.DataFrame, output_dir: str):
    """Create correlation plot between different scores."""
    # Select score columns
    score_cols = ["ahns_score", "harmony_score", "novelty_score"]
    corr_matrix = df[score_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Score Correlations")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_correlations.png"))
    plt.close()

def create_prompt_comparison_plot(df: pd.DataFrame, output_dir: str):
    """Create box plots comparing scores across different prompts."""
    plt.figure(figsize=(12, 6))
    
    # Melt the dataframe to get it in the right format for seaborn
    melted_df = pd.melt(df, 
                        id_vars=["prompt"],
                        value_vars=["ahns_score", "harmony_score", "novelty_score"],
                        var_name="score_type",
                        value_name="score")
    
    # Create box plot
    sns.boxplot(data=melted_df, x="prompt", y="score", hue="score_type")
    plt.xticks(rotation=45, ha="right")
    plt.title("Score Distribution by Prompt")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prompt_comparison.png"))
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