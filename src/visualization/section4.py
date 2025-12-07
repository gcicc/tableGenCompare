"""
Section 4 Visualization Functions

This module contains visualization functions for Hyperparameter Optimization (Section 4),
including Optuna study visualizations.
"""

import os
from pathlib import Path


def create_optuna_visualizations(study, model_name, results_path, verbose=True):
    """
    Create standardized Optuna visualization exports for a completed study.

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
            history_path = results_path / f'optim_history_{model_name}.png'
            fig1.write_image(str(history_path))
            generated_files['optimization_history'] = str(history_path)
            if verbose:
                print(f"[VIZ] Saved: optim_history_{model_name}.png")
        except Exception as e:
            if verbose:
                print(f"[WARNING] Optimization history plot failed for {model_name}: {e}")

        # 2. Parameter Importance Plot
        try:
            fig2 = vis.plot_param_importances(study)
            importance_path = results_path / f'param_importance_{model_name}.png'
            fig2.write_image(str(importance_path))
            generated_files['param_importance'] = str(importance_path)
            if verbose:
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
                parallel_path = results_path / f'parallel_coord_{model_name}.png'
                fig3.write_image(str(parallel_path))
                generated_files['parallel_coordinate'] = str(parallel_path)
                if verbose:
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

    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Hyperparameter Optimization Summary - All Models',
                 fontsize=16, fontweight='bold')

    # Left: Best objective values
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
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
