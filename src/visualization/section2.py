"""
Section 2 Visualization Functions

This module contains visualization functions for Exploratory Data Analysis (Section 2),
including mixed-association heatmaps and feature distribution plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from src.visualization.colors import REAL_COLOR


# Backward-compatible alias
def create_correlation_heatmap(*args, **kwargs):
    return create_mixed_association_heatmap(*args, **kwargs)


def create_mixed_association_heatmap(association_matrix, results_path,
                                     filename='mixed_association_heatmap.png',
                                     verbose=True,
                                     flagged_columns=None):
    """
    Create mixed-association heatmap with dynamic font sizing.

    Parameters:
    -----------
    association_matrix : pd.DataFrame
        Mixed-association matrix to visualize (Pearson, Cramer's V, eta)
    results_path : str or Path
        Directory to save output
    filename : str
        Output filename (default: 'mixed_association_heatmap.png')
    verbose : bool
        Print progress messages
    flagged_columns : set[str] | None
        Column names whose tick labels (x and y) should render in red to
        highlight that they are involved in a collinearity-reduction
        decision. Unflagged columns stay black.

    Returns:
    --------
    str : Path to saved file
    """
    n_cols = len(association_matrix.columns)
    show_annot = n_cols <= 6

    # Dynamic figure size
    figsize = (max(10, n_cols * 0.6), max(8, n_cols * 0.5))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(association_matrix,
                annot=show_annot,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                ax=ax)

    # Highlight flagged columns on x/y tick labels.
    if flagged_columns:
        flagged_set = set(flagged_columns)
        for lbl in ax.get_xticklabels():
            if lbl.get_text() in flagged_set:
                lbl.set_color('red')
                lbl.set_fontweight('bold')
        for lbl in ax.get_yticklabels():
            if lbl.get_text() in flagged_set:
                lbl.set_color('red')
                lbl.set_fontweight('bold')

    ax.set_title('Feature Mixed-Association Matrix', fontsize=14, fontweight='bold')

    # Footnote explaining metric types and ranges
    footnote = ('Pearson (num\u2013num): [\u22121, 1]  |  '
                'Cram\u00e9r\u2019s V (cat\u2013cat): [0, 1]  |  '
                'Correlation ratio \u03b7 (num\u2013cat): [0, 1]')
    if flagged_columns:
        footnote += '   \u2022   red labels: collinearity-treated pairs'
    fig.text(0.5, -0.02, footnote,
             ha='center', va='top', fontsize=9, style='italic', color='0.4')

    plt.tight_layout()

    output_path = Path(results_path) / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: {filename} (annotations {'on' if show_annot else 'off'})")

    return str(output_path)


def create_feature_distributions(data, target_column, results_path,
                                 plots_per_file=6, verbose=True):
    """
    Create feature distribution plots with multi-file splitting.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with features to plot
    target_column : str
        Target column to exclude from plots
    results_path : str or Path
        Directory to save outputs
    plots_per_file : int
        Number of plots per file (default 6 for 3x2 grid)
    verbose : bool
        Print progress messages

    Returns:
    --------
    list : Paths to saved files
    """
    GRID_COLS, GRID_ROWS = 3, 2

    # Get numeric columns excluding target
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_no_target = [col for col in numeric_cols if col != target_column]

    # Get categorical columns excluding target
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols_no_target = [col for col in categorical_cols if col != target_column]

    # Split numeric columns into chunks
    column_chunks = [numeric_cols_no_target[i:i+plots_per_file]
                     for i in range(0, len(numeric_cols_no_target), plots_per_file)]

    saved_files = []
    total_files = len(column_chunks) + (1 if categorical_cols_no_target else 0)

    if verbose:
        print(f"[VIZ] Creating {total_files} feature distribution file(s)...")

    # Generate numeric feature distribution plots
    for file_idx, cols_subset in enumerate(column_chunks, 1):
        fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(15, 8))
        fig.suptitle(f'Numeric Feature Distributions (Part {file_idx}/{len(column_chunks)})',
                     fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i, col in enumerate(cols_subset):
            axes[i].hist(data[col].dropna(), bins=20, color=REAL_COLOR, edgecolor='black', alpha=0.7)
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel('Value', fontsize=8)
            axes[i].set_ylabel('Frequency', fontsize=8)
            axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(len(cols_subset), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        # File naming (backward compatible)
        if len(column_chunks) > 1:
            filename = f'feature_distributions_part{file_idx}.png'
        else:
            filename = 'feature_distributions.png'

        output_path = Path(results_path) / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        saved_files.append(str(output_path))

        if verbose:
            print(f"[VIZ] Saved: {filename}")

    # Generate categorical feature distribution plots
    if categorical_cols_no_target:
        n_cat_cols = len(categorical_cols_no_target)
        n_cols = min(3, n_cat_cols)
        n_rows = (n_cat_cols + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle('Categorical Feature Distributions', fontsize=16, fontweight='bold')

        # Ensure axes is always iterable
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        for i, col in enumerate(categorical_cols_no_target):
            if i < len(axes):
                value_counts = data[col].value_counts()
                bars = axes[i].bar(range(len(value_counts)), value_counts.values,
                                   alpha=0.7, edgecolor='black')
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
                axes[i].set_title(col, fontsize=10)
                axes[i].set_ylabel('Count', fontsize=8)
                axes[i].grid(True, alpha=0.3, axis='y')

        # Hide unused subplots
        for j in range(len(categorical_cols_no_target), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        cat_filename = 'feature_distributions_categorical.png'
        output_path = Path(results_path) / cat_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(str(output_path))

        if verbose:
            print(f"[VIZ] Saved: {cat_filename}")

    return saved_files
