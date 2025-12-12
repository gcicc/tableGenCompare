"""
Hyperparameter optimization evaluation and analysis functions.

This module provides comprehensive analysis tools for Optuna hyperparameter
optimization results, including batch evaluation, visualization, and reporting.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from src.utils.paths import get_results_path, extract_dataset_identifier
from src.config import DATASET_IDENTIFIER


def evaluate_hyperparameter_optimization_results(section_number=4, scope=None, target_column=None):
    """
    Batch evaluation of hyperparameter optimization results for all available models.

    Replaces individual CHUNK_041, CHUNK_043, CHUNK_045, CHUNK_047, CHUNK_049, CHUNK_051
    Following CHUNK_018 pattern for Section 3.

    Parameters:
    - section_number: Section number for file organization (default 4)
    - scope: Notebook scope (globals()) to access study variables
    - target_column: Target column name for analysis

    Returns:
    - Dictionary with analysis results for all models
    """

    if scope is None:
        scope = globals()

    # Use global TARGET_COLUMN if not provided
    if target_column is None and 'TARGET_COLUMN' in scope:
        target_column = scope['TARGET_COLUMN']

    # Get dataset identifier - try multiple sources for robustness
    dataset_id = None

    # First try: scope (notebook globals) - this is what works for Section 2 & 3
    if 'DATASET_IDENTIFIER' in scope and scope['DATASET_IDENTIFIER']:
        dataset_id = scope['DATASET_IDENTIFIER']
        print(f"[LOCATION] Using DATASET_IDENTIFIER from scope: {dataset_id}")

    # Second try: setup module global variable
    elif DATASET_IDENTIFIER:
        dataset_id = DATASET_IDENTIFIER
        print(f"[LOCATION] Using DATASET_IDENTIFIER from setup module: {dataset_id}")

    # Fallback: extract from any available data files in scope
    else:
        print("[WARNING]  DATASET_IDENTIFIER not found! Attempting extraction from scope...")
        # Look for common data file variables in notebook scope
        for var_name in ['data_file', 'DATA_FILE', 'current_data_file']:
            if var_name in scope and scope[var_name]:
                dataset_id = extract_dataset_identifier(scope[var_name])
                print(f"[LOCATION] Extracted DATASET_IDENTIFIER from {var_name}: {dataset_id}")
                break

        if not dataset_id:
            dataset_id = 'unknown-dataset'
            print(f"[WARNING]  Using fallback DATASET_IDENTIFIER: {dataset_id}")

    print(f"[TARGET] Final DATASET_IDENTIFIER for Section {section_number}: {dataset_id}")

    # Get base results directory for Section 4
    base_results_dir = get_results_path(dataset_id, section_number)

    print(f"\n{'='*80}")
    print(f"SECTION {section_number} - HYPERPARAMETER OPTIMIZATION BATCH ANALYSIS")
    print(f"{'='*80}")
    print(f"[FOLDER] Base results directory: {base_results_dir}")
    print(f"[TARGET] Target column: {target_column}")
    print(f"[CHART] Dataset identifier: {dataset_id}")
    print()

    # Define model configurations with directory names for 1:1 correspondence with Section 3
    # All 6 models should have matching directories: CTGAN, CTABGAN, CTABGANPLUS, GANERAID, COPULAGAN, TVAE
    model_configs = [
        {'name': 'CTGAN', 'study_var': 'ctgan_study', 'model_name': 'ctgan', 'section': '4.1.1', 'dir_name': 'CTGAN'},
        {'name': 'CTAB-GAN', 'study_var': 'ctabgan_study', 'model_name': 'ctabgan', 'section': '4.2.1', 'dir_name': 'CTABGAN'},
        {'name': 'CTAB-GAN+', 'study_var': 'ctabganplus_study', 'model_name': 'ctabganplus', 'section': '4.3.1', 'dir_name': 'CTABGANPLUS'},
        {'name': 'GANerAid', 'study_var': 'ganeraid_study', 'model_name': 'ganeraid', 'section': '4.4.1', 'dir_name': 'GANERAID'},
        {'name': 'CopulaGAN', 'study_var': 'copulagan_study', 'model_name': 'copulagan', 'section': '4.5.1', 'dir_name': 'COPULAGAN'},
        {'name': 'TVAE', 'study_var': 'tvae_study', 'model_name': 'tvae', 'section': '4.6.1', 'dir_name': 'TVAE'}
    ]

    analysis_results = {}
    summary_data = []

    for config in model_configs:
        model_name = config['name']
        study_var = config['study_var']
        model_key = config['model_name']
        section = config['section']
        dir_name = config['dir_name']

        print(f"\n[SEARCH] {section}: {model_name} Hyperparameter Optimization Analysis")
        print("-" * 60)

        try:
            # Check if study exists in scope
            if study_var in scope and scope[study_var] is not None:
                study_results = scope[study_var]

                print(f"[OK] {model_name} optimization study found")

                # Create model-specific subdirectory (matching Section 3 structure)
                model_results_dir = f"{base_results_dir}/{dir_name}"
                os.makedirs(model_results_dir, exist_ok=True)
                print(f"[FOLDER] Model directory: {model_results_dir}")

                # Run hyperparameter analysis with model-specific directory
                analysis_result = analyze_hyperparameter_optimization(
                    study_results=study_results,
                    model_name=model_key,
                    target_column=target_column,
                    results_dir=model_results_dir,  # Use model-specific directory
                    export_figures=True,  # Export all figures to files
                    export_tables=True,   # Export all tables to CSV
                    display_plots=False   # Don't display inline - save to files only
                )

                analysis_results[model_key] = analysis_result

                # Collect summary statistics
                if hasattr(study_results, 'best_trial'):
                    best_trial = study_results.best_trial
                    completed_trials = [t for t in study_results.trials if hasattr(t, 'state') and
                                      t.state.name == 'COMPLETE']

                    summary_data.append({
                        'model': model_name,
                        'section': section,
                        'best_score': best_trial.value if best_trial else None,
                        'total_trials': len(study_results.trials),
                        'completed_trials': len(completed_trials),
                        'best_trial_number': best_trial.number if best_trial else None,
                        'study_variable': study_var
                    })

                    print(f"[OK] {model_name} analysis completed - files exported to {model_results_dir}")

            else:
                print(f"[WARNING]  {model_name} optimization study not found (variable: {study_var})")
                print(f"   Skipping {model_name} analysis")

        except Exception as e:
            print(f"[ERROR] {model_name} analysis failed: {str(e)}")
            print(f"[SEARCH] Error details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            continue

    # Create comprehensive summary
    print(f"\n{'='*60}")
    print("HYPERPARAMETER OPTIMIZATION SUMMARY")
    print(f"{'='*60}")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        print(f"[CHART] Models analyzed: {len(summary_data)}")
        print(f"[STATS] Total optimization trials: {summary_df['total_trials'].sum()}")
        print(f"[OK] Successful trials: {summary_df['completed_trials'].sum()}")
        print()

        # Display summary table
        print("[INFO] OPTIMIZATION RESULTS SUMMARY:")
        print(summary_df.to_string(index=False))

        # Export summary to CSV in base results directory
        summary_file = f"{base_results_dir}/hyperparameter_optimization_summary.csv"
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        summary_df.to_csv(summary_file, index=False)
        print(f"\n[SAVE] Summary exported to: {summary_file}")

        # Find best performing model
        valid_scores = summary_df.dropna(subset=['best_score'])
        if not valid_scores.empty:
            best_model = valid_scores.loc[valid_scores['best_score'].idxmax()]
            print(f"\n[BEST] BEST PERFORMING MODEL: {best_model['model']}")
            print(f"   - Score: {best_model['best_score']:.4f}")
            print(f"   - Section: {best_model['section']}")
            print(f"   - Trials completed: {best_model['completed_trials']}")

    else:
        print("[WARNING]  No optimization results found")
        print("   Run hyperparameter optimization first (CHUNK_040, CHUNK_042, etc.)")

    print(f"\n[OK] Section {section_number} hyperparameter optimization batch analysis completed!")
    print(f"[FOLDER] All figures and tables exported to: {base_results_dir}")
    print(f"[CHART] Model-specific results in subdirectories: {[config['dir_name'] for config in model_configs]}")

    return {
        'analysis_results': analysis_results,
        'summary_data': summary_data,
        'results_dir': base_results_dir
    }


def analyze_hyperparameter_optimization(study_results, model_name,
                                       target_column, results_dir=None,
                                       export_figures=True, export_tables=True,
                                       display_plots=True):
    """
    Comprehensive hyperparameter optimization analysis with file output.
    Reusable across all model sections in Section 4.

    Enhanced following Section 3 lessons learned:
    - Model-specific subdirectories for clean organization
    - Professional dataframe display for all tables
    - Consistent display + file output for all models
    - High-quality graphics with proper styling

    Parameters:
    - study_results: Optuna study object or trial results dataframe
    - model_name: str, model identifier (ctgan, ctabgan, etc.)
    - target_column: str, target column name for context
    - results_dir: str, base results directory (creates model subdirectories)
    - export_figures: bool, save graphics to files
    - export_tables: bool, save tables to CSV files
    - display_plots: bool, show plots and dataframes in notebook

    Returns:
    - Dictionary with analysis results and file paths
    """

    # Helper function to safely convert parameters for plotting
    def safe_plot_parameter(param_col, trials_df):
        """Convert parameter values to plottable numeric format."""
        param_data = trials_df[param_col]

        # Handle TimeDelta64 types (convert to seconds)
        if pd.api.types.is_timedelta64_dtype(param_data):
            return param_data.dt.total_seconds()

        # Handle datetime types (convert to timestamp)
        elif pd.api.types.is_datetime64_dtype(param_data):
            return pd.to_numeric(param_data)

        # Handle list/tuple parameters (convert to string representation)
        elif param_data.apply(lambda x: isinstance(x, (list, tuple))).any():
            return param_data.astype(str)

        # Handle object/categorical types
        elif pd.api.types.is_object_dtype(param_data) or pd.api.types.is_categorical_dtype(param_data):
            try:
                # Try to convert to numeric
                return pd.to_numeric(param_data, errors='coerce')
            except:
                # If conversion fails, use string representation
                return param_data.astype(str)

        # For numeric types, return as-is
        else:
            return param_data

    # Enhanced Setup - Use provided results_dir directly (batch function provides model-specific directory)
    if results_dir is None:
        results_dir = Path('./results')
    elif isinstance(results_dir, str):
        results_dir = Path(results_dir)
    # If results_dir is already a Path object, use it directly

    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SEARCH] ANALYZING {model_name.upper()} HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    try:
        # 1. EXTRACT AND PROCESS TRIAL DATA
        print("[CHART] 1. TRIAL DATA EXTRACTION AND PROCESSING")
        print("-" * 40)

        # Handle both Optuna Study objects and DataFrames
        if hasattr(study_results, 'trials_dataframe'):
            trials_df = study_results.trials_dataframe()
        elif hasattr(study_results, 'trials'):
            # Convert Optuna study to DataFrame manually
            trial_data = []
            for trial in study_results.trials:
                trial_dict = {
                    'number': trial.number,
                    'value': trial.value,
                    'datetime_start': trial.datetime_start,
                    'datetime_complete': trial.datetime_complete,
                    'duration': trial.duration,
                    'state': trial.state.name
                }
                # Add parameters with 'params_' prefix
                for key, value in trial.params.items():
                    trial_dict[f'params_{key}'] = value
                trial_data.append(trial_dict)
            trials_df = pd.DataFrame(trial_data)
        else:
            # Assume it's already a DataFrame
            trials_df = study_results.copy()

        if trials_df.empty:
            print("[ERROR] No trial data available for analysis")
            return {"error": "No trial data available"}

        print(f"[OK] Extracted {len(trials_df)} trials for analysis")

        # Get parameter columns (those starting with 'params_')
        param_cols = [col for col in trials_df.columns if col.startswith('params_')]
        objective_col = 'value'

        if objective_col not in trials_df.columns:
            print(f"[ERROR] Objective column '{objective_col}' not found")
            return {"error": f"Objective column '{objective_col}' not found"}

        print(f"[CHART] 2. PARAMETER SPACE EXPLORATION ANALYSIS")
        print("-" * 40)
        print(f"   - Found {len(param_cols)} hyperparameters: {param_cols}")

        # Filter out completed trials for analysis
        completed_trials = trials_df[trials_df['state'] == 'COMPLETE'] if 'state' in trials_df.columns else trials_df

        if completed_trials.empty:
            print("[ERROR] No completed trials available for analysis")
            return {"error": "No completed trials available"}

        print(f"   - Using {len(completed_trials)} completed trials")

        # ENHANCED PARAMETER VS PERFORMANCE VISUALIZATION (WITH DTYPE FIX)
        if param_cols and (display_plots or export_figures):
            print(f"[STATS] Creating parameter vs performance visualizations...")

            # Limit parameters for readability
            n_params = min(6, len(param_cols))  # Limit to 6 for visualization
            if n_params > 0:
                n_cols = 3
                n_rows = (n_params + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                if n_params == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes
                else:
                    axes = axes.flatten()

                fig.suptitle(f'{model_name.upper()} - Parameter vs Performance Analysis',
                             fontsize=16, fontweight='bold')

                for i, param_col in enumerate(param_cols[:n_params]):
                    if param_col in completed_trials.columns:
                        try:
                            # CRITICAL FIX: Use safe parameter conversion for plotting
                            param_data = safe_plot_parameter(param_col, completed_trials)
                            objective_data = completed_trials[objective_col]

                            # Create scatter plot with converted data
                            axes[i].scatter(param_data, objective_data, alpha=0.6, s=50)
                            axes[i].set_xlabel(param_col)
                            axes[i].set_ylabel(f'{objective_col}')
                            axes[i].set_title(f'{param_col} vs Performance', fontweight='bold')
                            axes[i].grid(True, alpha=0.3)

                            # Add correlation coefficient if possible
                            try:
                                if pd.api.types.is_numeric_dtype(param_data):
                                    corr = param_data.corr(objective_data)
                                    axes[i].text(0.05, 0.95, f'Corr: {corr:.3f}',
                                               transform=axes[i].transAxes,
                                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                            except Exception as corr_error:
                                print(f"[WARNING] Could not calculate correlation for {param_col}: {corr_error}")

                        except Exception as plot_error:
                            print(f"[WARNING] Could not plot {param_col}: {plot_error}")
                            axes[i].text(0.5, 0.5, f'Plot Error\n{param_col}',
                                       transform=axes[i].transAxes, ha='center', va='center')
                            axes[i].set_title(f'{param_col} (Plot Error)', fontweight='bold')

                # Remove empty subplots
                for j in range(n_params, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()

                if export_figures:
                    param_plot_path = results_dir / f'{model_name}_parameter_analysis.png'
                    plt.savefig(param_plot_path, dpi=300, bbox_inches='tight')
                    print(f"   [FOLDER] Parameter analysis plot saved: {param_plot_path}")

                if display_plots:
                    plt.show()
                else:
                    plt.close()

        # BEST TRIAL ANALYSIS
        print(f"[CHART] 3. BEST TRIAL ANALYSIS")
        print("-" * 40)

        best_trial = completed_trials.loc[completed_trials[objective_col].idxmax()]
        print(f"[OK] Best Trial #{best_trial.get('number', 'Unknown')}")
        print(f"   - Best Score: {best_trial[objective_col]:.4f}")

        if 'duration' in best_trial:
            duration = best_trial['duration']
            if pd.isna(duration):
                print(f"   - Duration: Not available")
            else:
                try:
                    if isinstance(duration, pd.Timedelta):
                        print(f"   - Duration: {duration.total_seconds():.1f} seconds")
                    else:
                        print(f"   - Duration: {duration}")
                except:
                    print(f"   - Duration: {duration}")

        # Best parameters
        best_params = {col.replace('params_', ''): best_trial[col]
                      for col in param_cols if col in best_trial.index}

        print(f"   - Best Parameters:")
        for param, value in best_params.items():
            if isinstance(value, float):
                print(f"     - {param}: {value:.4f}")
            else:
                print(f"     - {param}: {value}")

        # CONVERGENCE ANALYSIS
        print(f"[CHART] 4. CONVERGENCE ANALYSIS")
        print("-" * 40)

        # FIXED: Handle convergence plots for both single and multiple trials
        if len(completed_trials) >= 1 and (display_plots or export_figures):
            if len(completed_trials) == 1:
                # Special handling for single trial case
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                # Single trial result
                trial_score = completed_trials[objective_col].iloc[0]
                trial_number = completed_trials['number'].iloc[0]

                ax.bar([trial_number], [trial_score], alpha=0.7, color='blue', width=0.5)
                ax.set_xlabel('Trial Number')
                ax.set_ylabel('Objective Value')
                ax.set_title(f'{model_name.upper()} - Single Trial Result\n(Only 1 trial completed successfully)', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(trial_number - 1, trial_number + 1)

                # Add value annotation
                ax.annotate(f'{trial_score:.4f}', (trial_number, trial_score),
                           textcoords="offset points", xytext=(0,10), ha='center')

                print(f"   [WARNING] Only {len(completed_trials)} trial completed - limited convergence analysis")

            else:
                # Original multi-trial analysis
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

                # Trial progression
                ax1.plot(completed_trials['number'], completed_trials[objective_col], 'o-', alpha=0.7)
                ax1.set_xlabel('Trial Number')
                ax1.set_ylabel('Objective Value')
                ax1.set_title(f'{model_name.upper()} - Trial Progression', fontweight='bold')
                ax1.grid(True, alpha=0.3)

                # Best value progression (cumulative best)
                cumulative_best = completed_trials[objective_col].cummax()
                ax2.plot(completed_trials['number'], cumulative_best, 'g-', linewidth=2, label='Best So Far')
                ax2.fill_between(completed_trials['number'], cumulative_best, alpha=0.3, color='green')
                ax2.set_xlabel('Trial Number')
                ax2.set_ylabel('Best Objective Value')
                ax2.set_title(f'{model_name.upper()} - Convergence', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if export_figures:
                convergence_plot_path = results_dir / f'{model_name}_convergence_analysis.png'
                plt.savefig(convergence_plot_path, dpi=300, bbox_inches='tight')
                print(f"   [FOLDER] Convergence plot saved: {convergence_plot_path}")

            if display_plots:
                plt.show()
            else:
                plt.close()

        elif len(completed_trials) == 0:
            print(f"   [ERROR] No completed trials for {model_name} - cannot generate convergence plot")
            print(f"   [HINT] Check hyperparameter optimization logs for {model_name} training failures")
        else:
            print(f"   [INFO] Convergence plot generation skipped (display_plots={display_plots}, export_figures={export_figures})")

        # STATISTICAL SUMMARY
        print(f"[CHART] 5. STATISTICAL SUMMARY")
        print("-" * 40)

        summary_stats = completed_trials[objective_col].describe()
        print(f"[OK] Performance Statistics:")
        print(f"   - Mean Score: {summary_stats['mean']:.4f}")
        print(f"   - Std Dev: {summary_stats['std']:.4f}")
        print(f"   - Min Score: {summary_stats['min']:.4f}")
        print(f"   - Max Score: {summary_stats['max']:.4f}")
        print(f"   - Median Score: {summary_stats['50%']:.4f}")

        # EXPORT RESULTS TABLES
        files_generated = []

        if export_tables:
            # Export trial results
            trials_export_path = results_dir / f'{model_name}_trial_results.csv'
            completed_trials.to_csv(trials_export_path, index=False)
            files_generated.append(str(trials_export_path))
            print(f"   [FOLDER] Trial results saved: {trials_export_path}")

            # Export summary statistics
            summary_export_path = results_dir / f'{model_name}_optimization_summary.csv'
            summary_df = pd.DataFrame({
                'Metric': ['Best Score', 'Mean Score', 'Std Dev', 'Min Score', 'Max Score', 'Trials Completed'],
                'Value': [best_trial[objective_col], summary_stats['mean'],
                         summary_stats['std'], summary_stats['min'],
                         summary_stats['max'], len(completed_trials)]
            })
            summary_df.to_csv(summary_export_path, index=False)
            files_generated.append(str(summary_export_path))
            print(f"   [FOLDER] Summary statistics saved: {summary_export_path}")

        # PREPARE RETURN DATA
        analysis_results = {
            'best_score': float(best_trial[objective_col]),
            'best_params': best_params,
            'n_trials': len(completed_trials),
            'mean_score': float(summary_stats['mean']),
            'std_score': float(summary_stats['std']),
            'trials_df': completed_trials,
            'files_generated': files_generated,
            'output_dir': str(results_dir)
        }

        print(f"[OK] {model_name.upper()} optimization analysis completed successfully!")
        print(f"[FOLDER] Results saved to: {results_dir}")

        return analysis_results

    except Exception as e:
        print(f"[ERROR] Error in {model_name} hyperparameter optimization analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


print("[OK] Hyperparameter optimization evaluation functions loaded from src/evaluation/hyperparameters.py")
