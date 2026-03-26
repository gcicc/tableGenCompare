"""
Section 4 Visualization Functions

This module contains visualization functions for Hyperparameter Optimization (Section 4),
including Optuna study visualizations.
"""

import os
from pathlib import Path


def _save_plotly_figure(fig, output_path, verbose=True):
    """
    Save a Plotly figure with Kaleido fallback to HTML.

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    output_path : str or Path
        Output file path (should end with .png)
    verbose : bool
        Print progress messages

    Returns:
    --------
    str or None : Path to saved file (may be .html if Kaleido fails)
    """
    output_path = Path(output_path)

    try:
        fig.write_image(str(output_path))
        return str(output_path)
    except Exception as e:
        error_str = str(e).lower()
        if 'kaleido' in error_str or 'chrome' in error_str or 'orca' in error_str:
            # Fallback: save as HTML instead
            html_path = output_path.with_suffix('.html')
            try:
                fig.write_html(str(html_path))
                if verbose:
                    print(f"[WARNING] Saved as HTML (Kaleido unavailable): {html_path.name}")
                return str(html_path)
            except Exception as html_error:
                if verbose:
                    print(f"[WARNING] HTML fallback also failed: {html_error}")
                return None
        else:
            raise


def _apply_plotly_theme(fig, model_name=None):
    """Apply consistent Plotly theme matching the matplotlib style."""
    from src.visualization.colors import get_model_color
    model_color = get_model_color(model_name) if model_name else '#1f77b4'

    fig.update_layout(
        template='plotly_white',
        font=dict(family='Helvetica Neue, Helvetica, Arial, sans-serif', size=12),
        title_font=dict(size=14),
        colorway=[model_color],
    )
    return fig


def create_optuna_visualizations(study, model_name, results_path, verbose=True):
    """
    Create standardized Optuna visualization exports for a completed study.
    Outputs are saved as PNG (requires kaleido); falls back to HTML if unavailable.

    Parameters:
    -----------
    study : optuna.Study
        Completed Optuna study object
    model_name : str
        Model name (e.g., 'CTGAN', 'CTABGAN', etc.)
    results_path : str or Path
        Directory to save visualization outputs
    verbose : bool
        Print progress messages

    Returns:
    --------
    dict : Paths to generated files {'optimization_history': path, 'param_importance': path, 'parallel_coordinate': path}
    """
    try:
        import optuna.visualization as vis
    except ImportError:
        if verbose:
            print(f"[WARNING] Optuna visualization module not available")
        return {}

    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    generated_files = {}

    try:
        # 1. Optimization History Plot
        try:
            fig1 = vis.plot_optimization_history(study)
            _apply_plotly_theme(fig1, model_name)
            history_path = results_path / f'optim_history_{model_name}.png'
            saved_path = _save_plotly_figure(fig1, history_path, verbose)
            if saved_path:
                generated_files['optimization_history'] = saved_path
                if verbose and saved_path.endswith('.png'):
                    print(f"[VIZ] Saved: optim_history_{model_name}.png")
        except Exception as e:
            if verbose:
                print(f"[WARNING] Optimization history plot failed for {model_name}: {e}")

        # 2. Parameter Importance Plot
        try:
            fig2 = vis.plot_param_importances(study)
            _apply_plotly_theme(fig2, model_name)
            importance_path = results_path / f'param_importance_{model_name}.png'
            saved_path = _save_plotly_figure(fig2, importance_path, verbose)
            if saved_path:
                generated_files['param_importance'] = saved_path
                if verbose and saved_path.endswith('.png'):
                    print(f"[VIZ] Saved: param_importance_{model_name}.png")
        except Exception as e:
            if verbose:
                print(f"[WARNING] Parameter importance plot failed for {model_name}: {e}")

        # 3. Parallel Coordinate Plot (limit to top 5 parameters for readability)
        try:
            params = list(study.best_params.keys())
            # Limit to 5 parameters for better visualization
            params_to_plot = params[:min(5, len(params))]
            if params_to_plot:
                fig3 = vis.plot_parallel_coordinate(study, params=params_to_plot)
                _apply_plotly_theme(fig3, model_name)
                # Unify colorscale across all models for visual comparability
                if fig3.data and hasattr(fig3.data[0], 'line'):
                    fig3.data[0].line.colorscale = 'Viridis'
                parallel_path = results_path / f'parallel_coord_{model_name}.png'
                saved_path = _save_plotly_figure(fig3, parallel_path, verbose)
                if saved_path:
                    generated_files['parallel_coordinate'] = saved_path
                    if verbose and saved_path.endswith('.png'):
                        print(f"[VIZ] Saved: parallel_coord_{model_name}.png")
        except Exception as e:
            if verbose:
                print(f"[WARNING] Parallel coordinate plot failed for {model_name}: {e}")

        if verbose and generated_files:
            print(f"[OK] Generated {len(generated_files)} Optuna visualization(s) for {model_name}")

        return generated_files

    except Exception as e:
        if verbose:
            print(f"[ERROR] Optuna visualization failed for {model_name}: {e}")
        return {}


def create_all_models_optuna_summary(studies_dict, results_path, verbose=True):
    """
    Create a summary visualization comparing Optuna studies across all models.

    Parameters:
    -----------
    studies_dict : dict
        Dictionary mapping model names to their Optuna study objects
        e.g., {'CTGAN': study1, 'CTABGAN': study2, ...}
    results_path : str or Path
        Directory to save visualization output
    verbose : bool
        Print progress messages

    Returns:
    --------
    str or None : Path to saved summary file (None if generation failed)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        if verbose:
            print("[WARNING] matplotlib/numpy not available for summary plot")
        return None

    if not studies_dict or len(studies_dict) == 0:
        if verbose:
            print("[INFO] No studies provided for summary visualization")
        return None

    # Extract best values for each model
    model_names = []
    best_values = []
    n_trials = []

    for model_name, study in studies_dict.items():
        if study is not None and study.best_trial is not None:
            model_names.append(model_name)
            best_values.append(study.best_value)
            n_trials.append(len(study.trials))

    if not model_names:
        if verbose:
            print("[INFO] No valid studies to summarize")
        return None

    from src.visualization.colors import get_model_colors_for_list

    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Hyperparameter Optimization Summary - All Models',
                 fontsize=16, fontweight='bold')

    # Left: Best objective values
    colors = get_model_colors_for_list(model_names)
    ax1.barh(model_names, best_values, color=colors)
    ax1.set_xlabel('Best Objective Value (higher is better)', fontsize=11)
    ax1.set_title('Best Performance by Model', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')

    # Right: Number of trials
    ax2.barh(model_names, n_trials, color=colors)
    ax2.set_xlabel('Number of Trials', fontsize=11)
    ax2.set_title('Optimization Effort', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    results_path = Path(results_path)
    output_path = results_path / 'optuna_summary_all_models.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: optuna_summary_all_models.png")

    return str(output_path)
