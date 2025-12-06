"""
Section 3 Visualization Functions

This module contains visualization functions for Model Training evaluation (Section 3),
including correlation comparisons and distribution comparisons between real and synthetic data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def create_correlation_comparison(real_corr, synth_corr, model_name, results_dir,
                                  verbose=True):
    """
    Create side-by-side correlation heatmap comparison with dynamic font sizing.

    Parameters:
    -----------
    real_corr : pd.DataFrame
        Real data correlation matrix
    synth_corr : pd.DataFrame
        Synthetic data correlation matrix
    model_name : str
        Model name for title
    results_dir : Path or str
        Directory to save output
    verbose : bool
        Print messages

    Returns:
    --------
    str : Path to saved file
    """
    n_cols = len(real_corr.columns)

    # Dynamic settings (more conservative for dual display)
    if n_cols <= 8:
        show_annot, font_size, fmt, figsize = True, 9, '.2f', (16, 6)
    elif n_cols <= 12:
        show_annot, font_size, fmt, figsize = True, 7, '.2f', (18, 8)
    elif n_cols <= 18:
        show_annot, font_size, fmt, figsize = True, 5, '.2f', (20, 10)
    else:
        show_annot, font_size, fmt = False, None, '.2f'
        figsize = (max(20, n_cols * 0.8), max(10, n_cols * 0.5))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'{model_name.upper()} - Correlation Structure Comparison',
                 fontsize=16, fontweight='bold')

    # Real data
    sns.heatmap(real_corr, annot=show_annot, cmap='RdBu_r', center=0,
                square=True, ax=axes[0], fmt=fmt,
                annot_kws={'size': font_size} if show_annot else {},
                cbar_kws={'shrink': 0.8})
    axes[0].set_title('Real Data', fontsize=12)

    # Synthetic data
    sns.heatmap(synth_corr, annot=show_annot, cmap='RdBu_r', center=0,
                square=True, ax=axes[1], fmt=fmt,
                annot_kws={'size': font_size} if show_annot else {},
                cbar_kws={'shrink': 0.8})
    axes[1].set_title('Synthetic Data', fontsize=12)

    plt.tight_layout()

    output_file = Path(results_dir) / 'correlation_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: correlation_comparison.png")

    return str(output_file)


def create_distribution_comparison(real_data, synthetic_data, numeric_cols_no_target,
                                   model_name, results_dir, plots_per_file=6,
                                   display_plots=False, verbose=True):
    """
    Create distribution comparison plots with multi-file splitting.

    Parameters:
    -----------
    real_data : pd.DataFrame
        Real dataset
    synthetic_data : pd.DataFrame
        Synthetic dataset
    numeric_cols_no_target : list
        List of numeric column names (excluding target)
    model_name : str
        Model name for titles
    results_dir : Path or str
        Directory to save outputs
    plots_per_file : int
        Number of plots per file (default 6 for 3x2 grid)
    display_plots : bool
        Whether to display plots
    verbose : bool
        Print progress messages

    Returns:
    --------
    tuple : (list of file paths, avg_js_similarity)
    """
    from scipy.spatial.distance import jensenshannon

    GRID_COLS, GRID_ROWS = 3, 2

    column_chunks = [numeric_cols_no_target[i:i+plots_per_file]
                     for i in range(0, len(numeric_cols_no_target), plots_per_file)]

    saved_files = []
    js_scores = []

    for file_idx, cols_subset in enumerate(column_chunks, 1):
        fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(15, 8))
        fig.suptitle(f'{model_name.upper()} - Distribution Comparison (Part {file_idx}/{len(column_chunks)})',
                     fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i, col in enumerate(cols_subset):
            ax = axes[i]

            # Calculate JS divergence
            real_hist, bins = np.histogram(real_data[col].dropna(), bins=20, density=True)
            synth_hist, _ = np.histogram(synthetic_data[col].dropna(), bins=bins, density=True)

            real_hist = real_hist / real_hist.sum() if real_hist.sum() > 0 else real_hist
            synth_hist = synth_hist / synth_hist.sum() if synth_hist.sum() > 0 else synth_hist

            js_div = jensenshannon(real_hist, synth_hist)
            js_similarity = 1 - js_div
            js_scores.append(js_similarity)

            # Plot overlaid histograms
            ax.hist(real_data[col].dropna(), bins=20, alpha=0.7, label='Real',
                   density=True, color='blue', edgecolor='black')
            ax.hist(synthetic_data[col].dropna(), bins=20, alpha=0.7, label='Synthetic',
                   density=True, color='orange', edgecolor='black')
            ax.set_title(f'{col}\nJS Sim: {js_similarity:.3f}', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(len(cols_subset), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        # File naming
        if len(column_chunks) > 1:
            filename = f'distribution_comparison_part{file_idx}.png'
        else:
            filename = 'distribution_comparison.png'

        output_file = Path(results_dir) / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        saved_files.append(str(output_file))

        if display_plots:
            plt.show()
        else:
            plt.close()

    avg_js_similarity = np.mean(js_scores) if js_scores else 0

    if verbose:
        print(f"[VIZ] Generated {len(column_chunks)} distribution comparison file(s)")
        print(f"[VIZ] Average JS Similarity: {avg_js_similarity:.3f}")

    return saved_files, avg_js_similarity
