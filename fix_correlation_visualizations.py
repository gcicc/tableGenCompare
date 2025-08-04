#!/usr/bin/env python3
"""
Fix Correlation Matrix Visualizations
====================================

This script improves the legibility of correlation matrix visualizations by:
- Increasing font sizes for better readability
- Improving color contrast
- Better axis label positioning
- Professional formatting
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import glob
import warnings

warnings.filterwarnings('ignore')

def create_improved_correlation_plot(data, title="Correlation Matrix", save_path=None, figsize=(12, 10)):
    """
    Create an improved correlation matrix visualization with better legibility.
    
    Args:
        data: DataFrame for correlation analysis
        title: Plot title
        save_path: Path to save the improved plot
        figsize: Figure size tuple
    """
    
    print(f"Creating improved correlation plot: {title}")
    
    # Set up matplotlib for high-quality output
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 18,
        'figure.dpi': 300
    })
    
    # Calculate correlation matrix
    if isinstance(data, pd.DataFrame):
        # Select only numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            print("   No numerical columns found for correlation")
            return None
        
        corr_matrix = data[numerical_cols].corr()
    else:
        corr_matrix = data
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create improved heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    
    # Use a better colormap with higher contrast
    cmap = plt.cm.RdBu_r  # Red-blue colormap (reversed)
    
    # Create heatmap with improved settings
    heatmap = sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1, vmax=1,
        square=True,
        annot=True,
        fmt='.2f',
        cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
        annot_kws={'size': 10, 'weight': 'bold', 'color': 'black'},
        linewidths=1,
        linecolor='white',
        ax=ax
    )
    
    # Improve title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=11)
    
    # Improve colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='bold')
    
    # Add grid lines for better readability
    ax.set_facecolor('white')
    
    # Clean up spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('gray')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the improved plot
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"   Improved correlation plot saved: {save_path}")
    
    plt.close(fig)
    return fig

def fix_existing_correlation_matrices():
    """
    Find and fix existing correlation matrices in the results directory.
    """
    
    print("=" * 60)
    print("FIXING CORRELATION MATRIX VISUALIZATIONS")
    print("=" * 60)
    
    # Find all correlation CSV files to regenerate plots
    correlation_files = []
    
    # Search for correlation CSV files
    csv_patterns = [
        "results/**/correlation_matrix.csv",
        "results/**/*correlation*.csv"
    ]
    
    for pattern in csv_patterns:
        correlation_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(correlation_files)} correlation CSV files to process")
    
    for csv_file in correlation_files:
        try:
            print(f"\nProcessing: {csv_file}")
            
            # Read correlation data
            corr_data = pd.read_csv(csv_file, index_col=0)
            
            # Generate improved plot path
            csv_path = Path(csv_file)
            improved_png_path = csv_path.parent / f"improved_{csv_path.stem}.png"
            
            # Get dataset name from path for title
            dataset_name = "Dataset"
            if "diabetes" in str(csv_path):
                dataset_name = "Diabetes Dataset"
            elif "breast" in str(csv_path).lower():
                dataset_name = "Breast Cancer Dataset"
            elif "phase2" in str(csv_path):
                dataset_name = "Phase 2 Dataset"
            
            title = f"Correlation Matrix - {dataset_name}"
            
            # Create improved visualization
            create_improved_correlation_plot(
                data=corr_data,
                title=title,
                save_path=improved_png_path,
                figsize=(12, 10)
            )
            
        except Exception as e:
            print(f"   Error processing {csv_file}: {e}")
            continue
    
    # Also create some demo improved correlation matrices from sample data
    print("\nCreating demo improved correlation matrices...")
    
    # Demo 1: Clinical-style data
    np.random.seed(42)
    clinical_data = pd.DataFrame({
        'age': np.random.randint(20, 80, 500),
        'bmi': np.random.normal(25, 5, 500),
        'glucose': np.random.normal(100, 20, 500),
        'blood_pressure': np.random.normal(120, 15, 500),
        'cholesterol': np.random.normal(200, 40, 500),
        'heart_rate': np.random.normal(70, 12, 500),
        'diagnosis': np.random.choice([0, 1], 500)
    })
    
    create_improved_correlation_plot(
        data=clinical_data,
        title="Improved Correlation Matrix - Clinical Dataset",
        save_path="results/improved_clinical_correlation_demo.png",
        figsize=(10, 8)
    )
    
    # Demo 2: Financial-style data
    financial_data = pd.DataFrame({
        'income': np.random.lognormal(10, 0.5, 500),
        'credit_score': np.random.normal(650, 100, 500),
        'debt_ratio': np.random.beta(2, 5, 500),
        'savings': np.random.exponential(5000, 500),
        'age': np.random.randint(18, 70, 500),
        'loan_amount': np.random.lognormal(9, 0.8, 500),
        'default_risk': np.random.choice([0, 1], 500, p=[0.8, 0.2])
    })
    
    create_improved_correlation_plot(
        data=financial_data,
        title="Improved Correlation Matrix - Financial Dataset",
        save_path="results/improved_financial_correlation_demo.png",
        figsize=(10, 8)
    )
    
    print("\n" + "=" * 60)
    print("CORRELATION MATRIX IMPROVEMENTS COMPLETED")
    print("=" * 60)
    print("Key improvements made:")
    print("  ✓ Increased font sizes (11-16pt)")
    print("  ✓ Better color contrast (RdBu colormap)")
    print("  ✓ Improved axis label rotation (45° x-axis)")
    print("  ✓ Bold correlation values for readability")
    print("  ✓ Professional grid lines and spacing")
    print("  ✓ High-resolution output (300 DPI)")
    print("  ✓ Clean, professional styling")

if __name__ == "__main__":
    fix_existing_correlation_matrices()