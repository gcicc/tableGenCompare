"""
Section 3 Visualization Functions

This module contains visualization functions for Model Training evaluation (Section 3),
including correlation comparisons and distribution comparisons between real and synthetic data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from src.visualization.colors import REAL_COLOR, SYNTH_COLOR, get_model_color


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
    show_annot = n_cols <= 6

    # Dynamic figure size
    if n_cols <= 8:
        figsize = (16, 6)
    elif n_cols <= 12:
        figsize = (18, 8)
    elif n_cols <= 18:
        figsize = (20, 10)
    else:
        figsize = (max(20, n_cols * 0.8), max(10, n_cols * 0.5))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'{model_name.upper()} - Association Structure Comparison',
                 fontsize=16, fontweight='bold')

    # Real data
    sns.heatmap(
        real_corr,
        annot=show_annot,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes[0],
        cbar_kws={'shrink': 0.8}
    )
    axes[0].set_title('Real Data', fontsize=12)

    # Synthetic data
    sns.heatmap(
        synth_corr,
        annot=show_annot,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes[1],
        cbar_kws={'shrink': 0.8}
    )
    axes[1].set_title('Synthetic Data', fontsize=12)

    # Footnote explaining metric types and ranges
    fig.text(0.5, -0.02,
             'Pearson (num\u2013num): [\u22121, 1]  |  Cram\u00e9r\u2019s V (cat\u2013cat): [0, 1]  |  '
             'Correlation ratio \u03b7 (num\u2013cat): [0, 1]',
             ha='center', va='top', fontsize=9, style='italic', color='0.4')

    plt.tight_layout()

    output_file = Path(results_dir) / 'correlation_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: correlation_comparison.png (annotations {'on' if show_annot else 'off'})")

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
                   density=True, color=REAL_COLOR, edgecolor='black')
            ax.hist(synthetic_data[col].dropna(), bins=20, alpha=0.7, label='Synthetic',
                   density=True, color=SYNTH_COLOR, edgecolor='black')
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


def create_mode_collapse_visualization(mode_collapse_df, model_name, results_dir, verbose=True):
    """
    Create mode collapse summary visualization.

    Parameters:
    -----------
    mode_collapse_df : pd.DataFrame
        Mode collapse analysis results
    model_name : str
        Model name for title
    results_dir : Path or str
        Directory to save output
    verbose : bool
        Print messages

    Returns:
    --------
    str or None : Path to saved file (None if no data)
    """
    if mode_collapse_df.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name.upper()} - Mode Collapse Analysis',
                 fontsize=16, fontweight='bold')

    # Left: Category coverage per column
    colors_coverage = ['red' if x < 0.5 else 'orange' if x < 0.8 else 'green'
                       for x in mode_collapse_df['category_coverage']]

    ax1.barh(mode_collapse_df['column'], mode_collapse_df['category_coverage'],
             color=colors_coverage)
    ax1.axvline(x=0.8, color='orange', linestyle='--', linewidth=2, label='Mild threshold')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Moderate threshold')
    ax1.set_xlabel('Category Coverage (higher is better)')
    ax1.set_title('Category Coverage by Feature')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, 1.0)

    # Right: Distribution similarity
    colors_dist = ['red' if x < 0.5 else 'orange' if x < 0.7 else 'green'
                   for x in mode_collapse_df['distribution_similarity']]

    ax2.barh(mode_collapse_df['column'], mode_collapse_df['distribution_similarity'],
             color=colors_dist)
    ax2.set_xlabel('Distribution Similarity (higher is better)')
    ax2.set_title('Categorical Distribution Similarity')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, 1.0)

    plt.tight_layout()

    output_file = Path(results_dir) / 'mode_collapse_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: mode_collapse_summary.png")

    return str(output_file)


def create_mi_comparison(mi_real, mi_synth, mi_features, mi_correlation, model_name,
                        results_dir, verbose=True):
    """
    Create mutual information comparison visualization.

    Parameters:
    -----------
    mi_real : np.array
        Mutual information scores for real data
    mi_synth : np.array
        Mutual information scores for synthetic data
    mi_features : list
        List of feature names
    mi_correlation : float
        Correlation between real and synthetic MI scores
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
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(mi_features))
    width = 0.35

    ax.bar(x - width/2, mi_real, width, label='Real Data', alpha=0.8, color=REAL_COLOR)
    ax.bar(x + width/2, mi_synth, width, label='Synthetic Data', alpha=0.8, color=SYNTH_COLOR)

    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Mutual Information with Target', fontsize=12)
    ax.set_title(f'{model_name.upper()} - MI Preservation (Correlation: {mi_correlation:.3f})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mi_features, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_file = Path(results_dir) / 'mutual_information_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: mutual_information_comparison.png")

    return str(output_file)


def create_loss_plot(loss_history, model_name, results_dir, verbose=True):
    """
    Create training loss visualization.

    Parameters:
    -----------
    loss_history : dict or list
        Loss values over training epochs
        - If dict: {'generator': [...], 'discriminator': [...]}
        - If list: single loss sequence
    model_name : str
        Model name for title
    results_dir : Path or str
        Directory to save output
    verbose : bool
        Print messages

    Returns:
    --------
    str or None : Path to saved file (None if no data)
    """
    if loss_history is None or (isinstance(loss_history, (list, dict)) and len(loss_history) == 0):
        if verbose:
            print(f"[VIZ] No loss history available for {model_name}")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    # Handle different loss formats
    if isinstance(loss_history, dict):
        # GAN models with generator and discriminator losses
        for loss_type, values in loss_history.items():
            if len(values) > 0:
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, label=loss_type.capitalize(), alpha=0.8, linewidth=2)

        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=10)
        title = f'{model_name.upper()} - Training Loss Over Epochs'

    elif isinstance(loss_history, list):
        # Single loss sequence (e.g., TVAE)
        epochs = range(1, len(loss_history) + 1)
        ax.plot(epochs, loss_history, label='Loss', alpha=0.8, linewidth=2, color=REAL_COLOR)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=10)
        title = f'{model_name.upper()} - Training Loss Over Epochs'

    else:
        if verbose:
            print(f"[VIZ] Unrecognized loss format for {model_name}")
        return None

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add vertical line at convergence point if detectable
    if isinstance(loss_history, list) and len(loss_history) > 10:
        # Simple convergence detection: where loss stabilizes
        losses = np.array(loss_history)
        if len(losses) > 20:
            window = 10
            rolling_std = pd.Series(losses).rolling(window=window).std()
            convergence_idx = rolling_std.idxmin()
            if not np.isnan(convergence_idx) and convergence_idx > window:
                ax.axvline(x=convergence_idx, color='red', linestyle='--',
                          linewidth=1, alpha=0.5, label=f'Convergence (~epoch {int(convergence_idx)})')
                ax.legend(fontsize=10)

    plt.tight_layout()

    output_file = Path(results_dir) / 'training_loss.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: training_loss.png")

    return str(output_file)


def create_multi_model_loss_comparison(loss_histories, results_path, verbose=True):
    """
    Create comparative loss plot for all models.

    Parameters:
    -----------
    loss_histories : dict
        {model_name: loss_history, ...}
    results_path : str or Path
        Section 3 results directory
    verbose : bool
        Print messages

    Returns:
    --------
    str or None : Path to saved file (None if no data)
    """
    if not loss_histories or len(loss_histories) == 0:
        return None

    fig, ax = plt.subplots(figsize=(14, 7))

    for model_name, loss_history in loss_histories.items():
        color = get_model_color(model_name)
        if loss_history is None:
            continue

        # Extract primary loss (generator loss for GANs, or single loss)
        if isinstance(loss_history, dict) and 'generator' in loss_history:
            values = loss_history['generator']
            label = f'{model_name} (G)'
        elif isinstance(loss_history, dict) and len(loss_history) > 0:
            # Take first available loss type
            values = list(loss_history.values())[0]
            label = model_name
        elif isinstance(loss_history, list):
            values = loss_history
            label = model_name
        else:
            continue

        if len(values) > 0:
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=label, alpha=0.7, linewidth=2, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison - All Models', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(results_path) / "training_loss_comparison_all_models.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: training_loss_comparison_all_models.png")

    return str(output_path)
