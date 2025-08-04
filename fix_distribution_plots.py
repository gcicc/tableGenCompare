#!/usr/bin/env python3
"""
Fix Distribution Comparison Plots
=================================

This script improves the legibility of distribution comparison visualizations by:
- Better color distinction between original vs synthetic data
- Improved legend positioning
- Better transparency settings
- Professional formatting and spacing
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def create_improved_distribution_comparison(original_data, synthetic_data, title="Distribution Comparison", 
                                          save_path=None, figsize=(16, 12), max_cols=4):
    """
    Create an improved distribution comparison visualization.
    
    Args:
        original_data: Original dataset DataFrame
        synthetic_data: Synthetic dataset DataFrame
        title: Plot title
        save_path: Path to save the improved plot
        figsize: Figure size tuple
        max_cols: Maximum columns per row in subplot grid
    """
    
    print(f"Creating improved distribution comparison: {title}")
    
    # Set up matplotlib for high-quality output
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 300
    })
    
    # Get numerical columns only
    numerical_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) == 0:
        print("   No numerical columns found for distribution comparison")
        return None
    
    # Calculate subplot grid
    n_cols = min(max_cols, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    # Create figure with improved spacing
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Handle single subplot case
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Define improved color scheme with better contrast
    colors = {
        'original': '#1f77b4',    # Professional blue
        'synthetic': '#ff7f0e'    # Professional orange
    }
    
    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        
        # Get data for this column (handle missing columns gracefully)
        orig_data = original_data[col].dropna() if col in original_data.columns else pd.Series([])
        synth_data = synthetic_data[col].dropna() if col in synthetic_data.columns else pd.Series([])
        
        if len(orig_data) == 0 and len(synth_data) == 0:
            ax.text(0.5, 0.5, f'No data available\nfor {col}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(col, fontweight='bold', pad=10)
            continue
        
        # Create histograms with improved settings
        if len(orig_data) > 0:
            ax.hist(orig_data, bins=30, alpha=0.7, color=colors['original'], 
                   label='Original', density=True, edgecolor='white', linewidth=0.5)
        
        if len(synth_data) > 0:
            ax.hist(synth_data, bins=30, alpha=0.7, color=colors['synthetic'], 
                   label='Synthetic', density=True, edgecolor='white', linewidth=0.5)
        
        # Add density curves for better visualization
        if len(orig_data) > 5:  # Need sufficient data points
            try:
                orig_kde = orig_data.plot.density(ax=ax, color=colors['original'], 
                                                 linewidth=2, alpha=0.8, linestyle='--')
            except:
                pass
        
        if len(synth_data) > 5:
            try:
                synth_kde = synth_data.plot.density(ax=ax, color=colors['synthetic'], 
                                                   linewidth=2, alpha=0.8, linestyle='--')
            except:
                pass
        
        # Improve title and labels
        ax.set_title(col.replace('_', ' ').title(), fontweight='bold', pad=10)
        ax.set_xlabel('Value', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        
        # Add statistics text box
        if len(orig_data) > 0 and len(synth_data) > 0:
            orig_mean, orig_std = orig_data.mean(), orig_data.std()
            synth_mean, synth_std = synth_data.mean(), synth_data.std()
            
            stats_text = f'Original: μ={orig_mean:.2f}, σ={orig_std:.2f}\nSynthetic: μ={synth_mean:.2f}, σ={synth_std:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Improve grid and styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Hide empty subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout with better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the improved plot
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"   Improved distribution plot saved: {save_path}")
    
    plt.close(fig)
    return fig

def create_improved_feature_distributions(data, title="Feature Distributions", 
                                        save_path=None, figsize=(16, 12), max_cols=4):
    """
    Create improved feature distribution plots for a single dataset.
    """
    
    print(f"Creating improved feature distributions: {title}")
    
    # Set up matplotlib
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'figure.titlesize': 16,
        'figure.dpi': 300
    })
    
    # Get numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) == 0:
        print("   No numerical columns found")
        return None
    
    # Calculate subplot grid
    n_cols = min(max_cols, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Handle single subplot case
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Professional color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        col_data = data[col].dropna()
        
        if len(col_data) == 0:
            ax.text(0.5, 0.5, f'No data available\nfor {col}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create histogram with improved styling
        ax.hist(col_data, bins=30, alpha=0.7, color=colors[i % len(colors)], 
               edgecolor='white', linewidth=0.5, density=True)
        
        # Add density curve
        if len(col_data) > 5:
            try:
                col_data.plot.density(ax=ax, color=colors[i % len(colors)], 
                                    linewidth=2, alpha=0.8, linestyle='--')
            except:
                pass
        
        # Improve formatting
        ax.set_title(col.replace('_', ' ').title(), fontweight='bold', pad=10)
        ax.set_xlabel('Value', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        
        # Add statistics
        mean_val, std_val = col_data.mean(), col_data.std()
        stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}\nn={len(col_data)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
               facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide empty subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"   Improved feature distribution plot saved: {save_path}")
    
    plt.close(fig)
    return fig

def fix_existing_distribution_plots():
    """
    Create improved versions of distribution plots using sample data.
    """
    
    print("=" * 60)
    print("FIXING DISTRIBUTION COMPARISON PLOTS")
    print("=" * 60)
    
    # Create demo improved distribution plots
    np.random.seed(42)
    
    # Demo 1: Clinical dataset comparison
    print("\nCreating improved clinical distribution comparison...")
    
    original_clinical = pd.DataFrame({
        'age': np.random.normal(50, 15, 1000),
        'bmi': np.random.normal(25, 5, 1000),
        'glucose': np.random.normal(100, 20, 1000),
        'blood_pressure': np.random.normal(120, 15, 1000),
        'cholesterol': np.random.normal(200, 40, 1000),
        'heart_rate': np.random.normal(70, 12, 1000)
    })
    
    # Create synthetic version with slight differences
    synthetic_clinical = pd.DataFrame({
        'age': np.random.normal(49, 16, 1000),
        'bmi': np.random.normal(25.2, 4.8, 1000),
        'glucose': np.random.normal(102, 19, 1000),
        'blood_pressure': np.random.normal(118, 16, 1000),
        'cholesterol': np.random.normal(198, 42, 1000),
        'heart_rate': np.random.normal(71, 11, 1000)
    })
    
    create_improved_distribution_comparison(
        original_clinical, synthetic_clinical,
        title="Improved Distribution Comparison - Clinical Dataset",
        save_path="results/improved_clinical_distribution_comparison.png",
        figsize=(16, 10)
    )
    
    # Demo 2: Financial dataset comparison
    print("Creating improved financial distribution comparison...")
    
    original_financial = pd.DataFrame({
        'income': np.random.lognormal(10, 0.5, 1000),
        'credit_score': np.random.normal(650, 100, 1000),
        'debt_ratio': np.random.beta(2, 5, 1000),
        'savings': np.random.exponential(5000, 1000),
        'age': np.random.randint(18, 70, 1000),
        'loan_amount': np.random.lognormal(9, 0.8, 1000)
    })
    
    synthetic_financial = pd.DataFrame({
        'income': np.random.lognormal(9.98, 0.52, 1000),
        'credit_score': np.random.normal(648, 102, 1000),
        'debt_ratio': np.random.beta(2.1, 4.9, 1000),
        'savings': np.random.exponential(4900, 1000),
        'age': np.random.randint(18, 70, 1000),
        'loan_amount': np.random.lognormal(8.95, 0.82, 1000)
    })
    
    create_improved_distribution_comparison(
        original_financial, synthetic_financial,
        title="Improved Distribution Comparison - Financial Dataset",
        save_path="results/improved_financial_distribution_comparison.png",
        figsize=(16, 10)
    )
    
    # Demo 3: Individual feature distributions
    print("Creating improved feature distribution plots...")
    
    create_improved_feature_distributions(
        original_clinical,
        title="Improved Feature Distributions - Clinical Dataset",
        save_path="results/improved_clinical_feature_distributions.png",
        figsize=(16, 10)
    )
    
    create_improved_feature_distributions(
        original_financial,
        title="Improved Feature Distributions - Financial Dataset", 
        save_path="results/improved_financial_feature_distributions.png",
        figsize=(16, 10)
    )
    
    print("\n" + "=" * 60)
    print("DISTRIBUTION PLOT IMPROVEMENTS COMPLETED")
    print("=" * 60)
    print("Key improvements made:")
    print("  - Better color contrast (blue vs orange)")
    print("  - Improved transparency settings (0.7 alpha)")
    print("  - Added density curves for better comparison")
    print("  - Professional legend positioning")
    print("  - Statistical summaries in text boxes")
    print("  - High-resolution output (300 DPI)")
    print("  - Clean grid lines and styling")
    print("  - Better spacing and layout")

if __name__ == "__main__":
    fix_existing_distribution_plots()