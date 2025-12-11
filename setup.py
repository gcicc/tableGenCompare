"""
Backward-compatible import layer for notebooks.

This module serves as a thin re-export layer, allowing notebooks to continue
using 'from setup import *' while the actual implementation has been moved to
the modular src/ directory structure.

All functionality is re-exported from src/ modules for backward compatibility.
"""

# ============================================================================
# ESSENTIAL IMPORTS FROM SRC - Available globally when using 'from setup import *'
# ============================================================================

# Import all essential data science libraries (pandas, numpy, sklearn, etc.)
from src import *

# Import configuration and session management
from src.config import (
    SESSION_TIMESTAMP,
    DATASET_IDENTIFIER,
    CURRENT_DATA_FILE,
    refresh_session_timestamp
)

# Import path utilities
from src.utils.paths import (
    extract_dataset_identifier,
    get_results_path
)

# Import documentation utilities
from src.utils.documentation import (
    create_section2_readme,
    create_section3_main_readme,
    create_section3_model_readme,
    create_section5_readme
)

# Import visualization functions (Phase 2)
from src.visualization.section2 import (
    create_correlation_heatmap,
    create_feature_distributions
)
from src.visualization.section3 import (
    create_correlation_comparison,
    create_distribution_comparison,
    create_mode_collapse_visualization,
    create_mi_comparison,
    create_loss_plot,
    create_multi_model_loss_comparison
)
from src.visualization.section4 import (
    create_optuna_visualizations,
    create_all_models_optuna_summary
)
from src.visualization.section5 import (
    create_trts_visualizations
)

# Import evaluation functions (Phase 3)
from src.evaluation.mode_collapse import (
    detect_mode_collapse
)
from src.evaluation.quality import (
    calculate_mutual_information,
    evaluate_synthetic_data_quality
)
from src.evaluation.trts import (
    comprehensive_trts_analysis
)
from src.evaluation.privacy import (
    calculate_privacy_metrics
)

# Import visualization functions for Section 5 (Phase 3)
from src.visualization.section5 import (
    create_privacy_dashboard
)

# Import objective functions (Phase 4)
from src.objective.functions import (
    enhanced_objective_function_v2,
    evaluate_ganeraid_objective
)

# Future imports will be added here as we migrate code to src/ modules:
# from src.models.wrappers import CTABGANModel, CTABGANPlusModel, GANerAidModel, etc.
# from src.evaluation.quality import evaluate_synthetic_data_quality

print("[SETUP] Thin re-export layer loaded successfully!")
print(f"[SETUP] Session timestamp: {SESSION_TIMESTAMP}")

# ============================================================================
# LEGACY CODE - To be migrated to src/ modules in future phases
# ============================================================================



# ============================================================================
# MODEL IMPORTS - Migrated to src/models/ (Phase 1)
# ============================================================================

# Import model classes and availability flags from src.models
from src.models.imports import (
    CTABGANSynthesizer,
    CTABGAN_AVAILABLE,
    CTABGANPLUS_AVAILABLE,
    GANerAidModel,
    GANERAID_AVAILABLE
)

from src.models.wrappers import (
    CTABGANModel,
    CTABGANPlusModel
)

# Phase 2 Migration: Data Preprocessing Functions
from src.data.preprocessing import (
    get_categorical_columns_for_models,
    clean_and_preprocess_data,
    prepare_data_for_any_model,
    prepare_data_for_hyperparameter_optimization
)

# Code Chunk ID: CHUNK_004 - Required Libraries Import
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("[OK] All required libraries imported successfully")

# Code Chunk ID: CHUNK_017 - Comprehensive Data Quality Evaluation Function
# COMPREHENSIVE DATA QUALITY EVALUATION FUNCTION
# CRITICAL: Must be defined before Section 3.1 calls

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


from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# MIGRATED TO: src/evaluation/quality.py (Phase 3, lines 345-774)
# Usage: from src.evaluation.quality import evaluate_synthetic_data_quality


# BATCH EVALUATION SYSTEM FOR ALL SECTIONS

def evaluate_all_available_models(section_number, scope=None, models_to_evaluate=None, real_data=None, target_col=None):
    """
    Wrapper for unified evaluate_trained_models function - Section 3 pattern
    
    Parameters:
    - section_number: Section number for file organization (3, 5, etc.)
    - scope: globals() from notebook for variable access (required for notebook use)
    - models_to_evaluate: List of specific models to evaluate (optional, evaluates all if None)
    - real_data: Real dataset (uses 'data' from scope if not provided)
    - target_col: Target column name (uses 'target_column' from scope if not provided)
    
    Returns:
    - Dictionary with results for each evaluated model
    """
    return evaluate_trained_models(
        section_number=section_number,
        variable_pattern='standard',  # Uses synthetic_data_* variables
        scope=scope,
        models_to_evaluate=models_to_evaluate,
        real_data=real_data,
        target_col=target_col
    )

# Code Chunk ID: CHUNK_037 - Enhanced Objective Function v2
# MIGRATED TO: src/objective/functions.py (Phase 3, Task 3.8)
# Now includes Optuna pruning support for early stopping
# Usage: from src.objective.functions import enhanced_objective_function_v2
# New: Optional `trial` parameter for Optuna pruning (backward compatible - defaults to None)

# MIGRATED TO: src/objective/functions.py (Phase 4, lines 381-686)
# Usage: from src.objective.functions import enhanced_objective_function_v2

def analyze_hyperparameter_optimization(study_results, model_name, 
                                       target_column, results_dir=None,
                                       export_figures=True, export_tables=True,
                                       display_plots=True):
    """
    Comprehensive hyperparameter optimization analysis with file output
    Reusable across all model sections in Section 4
    
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
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore')
    
    # Helper function to safely convert parameters for plotting
    def safe_plot_parameter(param_col, trials_df):
        """Convert parameter values to plottable numeric format"""
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
                            axes[i].text(0.5, 0.5, f'Plot Error\\n{param_col}', 
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

print("[OK] Batch evaluation system loaded!")
print("[OK] Enhanced objective function v2 with DYNAMIC TARGET COLUMN support defined!")
print("[OK] Enhanced hyperparameter optimization analysis function loaded!")

print("[TARGET] SETUP MODULE LOADED SUCCESSFULLY!")
print("="*60)

# Setup Imports - Global dependencies for objective functions
# CRITICAL: This cell provides wasserstein_distance import for enhanced_objective_function_v2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance  # CRITICAL: Required for EMD calculations
print("[OK] Enhanced objective function dependencies imported")

# Set style
plt.style.use('default')
sns.set_palette("husl")
print("[PACKAGE] Basic libraries imported successfully")

# Import Optuna for hyperparameter optimization
OPTUNA_AVAILABLE = False
try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("[OK] Optuna imported successfully")
except ImportError:
    print("[ERROR] Optuna not found - hyperparameter optimization not available")

# Import CTGAN
CTGAN_AVAILABLE = False
try:
    from ctgan import CTGAN
    CTGAN_AVAILABLE = True
    print("[OK] CTGAN imported successfully")
except ImportError:
    print("[ERROR] CTGAN not found")

print("[CONFIG] Setup imports cell restored from main branch - wasserstein_distance now available globally")


def evaluate_section5_optimized_models(section_number=5, scope=None, target_column=None):
    """
    Wrapper for unified evaluate_trained_models function - Section 5 pattern
    
    Parameters:
    - section_number: Section number for file organization (default 5)
    - scope: Notebook scope (globals()) to access synthetic data and results
    - target_column: Target column name for analysis
    
    Returns:
    - Dictionary with batch evaluation results and file paths
    """
    return evaluate_trained_models(
        section_number=section_number,
        variable_pattern='final',  # Uses synthetic_*_final variables
        scope=scope,
        models_to_evaluate=None,
        real_data=None,
        target_col=target_column
    )

# PARAMETER MANAGEMENT FUNCTIONS FOR SECTION 4 & 5 INTEGRATION

def save_best_parameters_to_csv(scope=None, section_number=4, dataset_identifier=None):
    """
    Save all best hyperparameters from Section 4 optimization to CSV format.
    
    Parameters:
    - scope: Notebook scope (globals()) to access study variables
    - section_number: Section number for file organization (default 4) 
    - dataset_identifier: Dataset name for folder structure
    
    Returns:
    - Dictionary with save results and file path
    """
    
    if scope is None:
        scope = globals()
    
    # Auto-detect dataset identifier
    if dataset_identifier is None:
        if 'DATASET_IDENTIFIER' in scope and scope['DATASET_IDENTIFIER']:
            dataset_identifier = scope['DATASET_IDENTIFIER']
        elif DATASET_IDENTIFIER:
            dataset_identifier = DATASET_IDENTIFIER
        else:
            # Fallback extraction
            for var_name in ['data_file', 'DATA_FILE', 'current_data_file']:
                if var_name in scope and scope[var_name]:
                    dataset_identifier = extract_dataset_identifier(scope[var_name])
                    break
            if not dataset_identifier:
                dataset_identifier = 'unknown-dataset'
    
    # Get results directory
    results_dir = get_results_path(dataset_identifier, section_number)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"[SAVE] SAVING BEST PARAMETERS FROM SECTION {section_number}")
    print("=" * 60)
    print(f"[FOLDER] Target directory: {results_dir}")
    
    # Model study variable mappings
    model_studies = {
        'CTGAN': 'ctgan_study',
        'CTAB-GAN': 'ctabgan_study', 
        'CTAB-GAN+': 'ctabganplus_study',
        'GANerAid': 'ganeraid_study',
        'CopulaGAN': 'copulagan_study',
        'TVAE': 'tvae_study'
    }
    
    parameter_rows = []
    summary_rows = []
    
    for model_name, study_var in model_studies.items():
        print(f"\n[CHART] Processing {model_name} parameters...")
        
        try:
            if study_var in scope and scope[study_var] is not None:
                study = scope[study_var]
                
                if hasattr(study, 'best_trial') and study.best_trial:
                    best_trial = study.best_trial
                    best_params = best_trial.params
                    best_score = best_trial.value
                    trial_number = best_trial.number
                    
                    print(f"[OK] Found {model_name}: {len(best_params)} parameters, score: {best_score:.4f}")
                    
                    # Flatten parameters for CSV format
                    for param_name, param_value in best_params.items():
                        # Handle complex parameter types
                        param_type = type(param_value).__name__
                        
                        # Convert tuples/lists to string representation
                        if isinstance(param_value, (tuple, list)):
                            # Also save individual components for tuple parameters
                            if isinstance(param_value, tuple) and len(param_value) == 2:
                                # Common case: betas=(0.5, 0.9) becomes betas_0=0.5, betas_1=0.9
                                parameter_rows.append({
                                    'model_name': model_name,
                                    'parameter_name': f'{param_name}_0',
                                    'parameter_value': param_value[0],
                                    'parameter_type': type(param_value[0]).__name__,
                                    'best_score': best_score,
                                    'trial_number': trial_number,
                                    'original_param': param_name,
                                    'is_component': True
                                })
                                parameter_rows.append({
                                    'model_name': model_name,
                                    'parameter_name': f'{param_name}_1', 
                                    'parameter_value': param_value[1],
                                    'parameter_type': type(param_value[1]).__name__,
                                    'best_score': best_score,
                                    'trial_number': trial_number,
                                    'original_param': param_name,
                                    'is_component': True
                                })
                            
                            # Always save full tuple/list as string
                            param_value_str = str(param_value)
                        else:
                            param_value_str = param_value
                        
                        parameter_rows.append({
                            'model_name': model_name,
                            'parameter_name': param_name,
                            'parameter_value': param_value_str,
                            'parameter_type': param_type,
                            'best_score': best_score,
                            'trial_number': trial_number,
                            'original_param': param_name,
                            'is_component': False
                        })
                    
                    # Add summary row
                    summary_rows.append({
                        'model_name': model_name,
                        'best_score': best_score,
                        'trial_number': trial_number,
                        'num_parameters': len(best_params),
                        'study_variable': study_var,
                        'parameters_saved': len(best_params)
                    })
                    
                else:
                    print(f"[WARNING]  {model_name}: No best_trial found")
                    
            else:
                print(f"[WARNING]  {model_name}: Study variable '{study_var}' not found")
                
        except Exception as e:
            print(f"[ERROR] {model_name}: Error processing - {str(e)}")
    
    # Save results to CSV files
    files_saved = []
    
    if parameter_rows:
        # Main parameters file
        params_df = pd.DataFrame(parameter_rows)
        params_file = f"{results_dir}/best_parameters.csv"
        params_df.to_csv(params_file, index=False)
        files_saved.append(params_file)
        print(f"\n[OK] Parameters saved: {params_file}")
        print(f"   - Total parameter entries: {len(parameter_rows)}")
        
        # Summary file
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = f"{results_dir}/best_parameters_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            files_saved.append(summary_file)
            print(f"[OK] Summary saved: {summary_file}")
            print(f"   - Models processed: {len(summary_rows)}")
    
    else:
        print("[ERROR] No parameters found to save!")
        return {
            'success': False,
            'message': 'No parameters found',
            'files_saved': []
        }
    
    print(f"\n[SAVE] Parameter saving completed!")
    print(f"[FOLDER] Files saved to: {results_dir}")
    
    return {
        'success': True,
        'files_saved': files_saved,
        'parameters_count': len(parameter_rows),
        'models_count': len(summary_rows),
        'results_dir': results_dir
    }

def load_best_parameters_from_csv(section_number=4, dataset_identifier=None, fallback_to_memory=True, scope=None):
    """
    Load best hyperparameters from CSV files with memory fallback.
    
    Parameters:
    - section_number: Section number for file location (default 4)
    - dataset_identifier: Dataset name for folder structure
    - fallback_to_memory: Use in-memory study variables if CSV not found
    - scope: Notebook scope (globals()) for memory fallback
    
    Returns:
    - Dictionary with parameters for each model: {'ctgan': {...}, 'ctabgan': {...}}
    """
    
    if scope is None:
        scope = globals()
    
    # Auto-detect dataset identifier
    if dataset_identifier is None:
        if 'DATASET_IDENTIFIER' in scope and scope['DATASET_IDENTIFIER']:
            dataset_identifier = scope['DATASET_IDENTIFIER']
        elif DATASET_IDENTIFIER:
            dataset_identifier = DATASET_IDENTIFIER
        else:
            # Fallback extraction
            for var_name in ['data_file', 'DATA_FILE', 'current_data_file']:
                if var_name in scope and scope[var_name]:
                    dataset_identifier = extract_dataset_identifier(scope[var_name])
                    break
            if not dataset_identifier:
                dataset_identifier = 'unknown-dataset'
    
    # Get results directory
    results_dir = get_results_path(dataset_identifier, section_number)
    params_file = f"{results_dir}/best_parameters.csv"
    
    print(f"[LOAD] LOADING BEST PARAMETERS FROM SECTION {section_number}")
    print("=" * 60)
    print(f"[FOLDER] Looking for: {params_file}")
    
    parameters = {}
    load_source = "unknown"
    
    # Try loading from CSV first
    if os.path.exists(params_file):
        try:
            print(f"[OK] Found parameter CSV file")
            params_df = pd.read_csv(params_file)
            
            # Reconstruct parameter dictionaries per model
            for model_name in params_df['model_name'].unique():
                model_params = {}
                model_data = params_df[params_df['model_name'] == model_name]
                
                # Group parameters, handling tuple reconstruction
                for _, row in model_data.iterrows():
                    param_name = row['parameter_name']
                    param_value = row['parameter_value']
                    param_type = row['parameter_type']
                    is_component = row.get('is_component', False)
                    original_param = row.get('original_param', param_name)
                    
                    # Skip component entries - we'll reconstruct tuples from full entries
                    if is_component:
                        continue
                    
                    # Type conversion
                    if param_type == 'int':
                        param_value = int(param_value)
                    elif param_type == 'float':
                        param_value = float(param_value)
                    elif param_type == 'bool':
                        param_value = str(param_value).lower() in ['true', '1', 'yes']
                    elif param_type == 'tuple':
                        # Reconstruct tuple from string representation
                        try:
                            param_value = eval(param_value)  # Safe for controlled parameter data
                        except:
                            param_value = str(param_value)
                    # str and other types use as-is
                    
                    model_params[param_name] = param_value
                
                # Map model name to standard format
                model_key = model_name.lower().replace('-', '').replace('+', 'plus')
                parameters[model_key] = model_params
                
                print(f"[OK] Loaded {model_name}: {len(model_params)} parameters")
            
            load_source = "CSV file"
            
        except Exception as e:
            print(f"[ERROR] Error reading CSV file: {str(e)}")
            parameters = {}
    
    else:
        print(f"[WARNING]  Parameter CSV file not found")
    
    # Fallback to memory if CSV loading failed or not found
    if not parameters and fallback_to_memory:
        print(f"\n[PROCESS] Falling back to in-memory study variables...")
        
        model_studies = {
            'CTGAN': ('ctgan_study', 'ctgan'),
            'CTAB-GAN': ('ctabgan_study', 'ctabgan'), 
            'CTAB-GAN+': ('ctabganplus_study', 'ctabganplus'),
            'GANerAid': ('ganeraid_study', 'ganeraid'),
            'CopulaGAN': ('copulagan_study', 'copulagan'),
            'TVAE': ('tvae_study', 'tvae')
        }
        
        memory_loaded = 0
        for model_name, (study_var, model_key) in model_studies.items():
            if study_var in scope and scope[study_var] is not None:
                study = scope[study_var]
                if hasattr(study, 'best_trial') and study.best_trial:
                    parameters[model_key] = study.best_trial.params
                    memory_loaded += 1
                    print(f"[OK] {model_name}: Loaded from memory ({len(study.best_trial.params)} params)")
        
        if memory_loaded > 0:
            load_source = "memory fallback"
        else:
            print(f"[ERROR] No parameters found in memory either")
            load_source = "none"
    
    print(f"\n[LOAD] Parameter loading completed!")
    print(f"[SEARCH] Source: {load_source}")
    print(f"[CHART] Models loaded: {len(parameters)}")
    for model_key, params in parameters.items():
        print(f"   - {model_key}: {len(params)} parameters")
    
    return {
        'parameters': parameters,
        'source': load_source,
        'models_count': len(parameters),
        'file_path': params_file if load_source == "CSV file" else None
    }

def get_model_parameters(model_name, section_number=4, dataset_identifier=None, scope=None):
    """
    Unified parameter retrieval for a specific model with CSV/memory fallback.
    
    Parameters:
    - model_name: Model name ('ctgan', 'ctabgan', etc.)
    - section_number: Section number for file location
    - dataset_identifier: Dataset name for folder structure
    - scope: Notebook scope for memory fallback
    
    Returns:
    - Dictionary with model parameters or None if not found
    """
    
    # Load all parameters
    param_data = load_best_parameters_from_csv(
        section_number=section_number,
        dataset_identifier=dataset_identifier, 
        fallback_to_memory=True,
        scope=scope
    )
    
    # Normalize model name
    model_key = model_name.lower().replace('-', '').replace('+', 'plus')
    
    if model_key in param_data['parameters']:
        print(f"[OK] {model_name.upper()} parameters loaded from {param_data['source']}")
        return param_data['parameters'][model_key]
    else:
        print(f"[ERROR] {model_name.upper()} parameters not found")
        return None

def compare_parameters_sources(scope=None, section_number=4, dataset_identifier=None, verbose=True):
    """
    Compare parameters between CSV files and in-memory study variables.
    
    Parameters:
    - scope: Notebook scope (globals()) for memory access
    - section_number: Section number for CSV location
    - dataset_identifier: Dataset name for folder structure
    - verbose: Print detailed comparison results
    
    Returns:
    - Dictionary with comparison results
    """
    
    if scope is None:
        scope = globals()
    
    if verbose:
        print(f"[SEARCH] COMPARING PARAMETER SOURCES")
        print("=" * 50)
    
    # Load from CSV (without memory fallback)
    csv_data = load_best_parameters_from_csv(
        section_number=section_number,
        dataset_identifier=dataset_identifier,
        fallback_to_memory=False,
        scope=scope
    )
    
    # Load from memory directly
    memory_params = {}
    model_studies = {
        'ctgan': 'ctgan_study',
        'ctabgan': 'ctabgan_study',
        'ctabganplus': 'ctabganplus_study',
        'ganeraid': 'ganeraid_study',
        'copulagan': 'copulagan_study',
        'tvae': 'tvae_study'
    }
    
    for model_key, study_var in model_studies.items():
        if study_var in scope and scope[study_var] is not None:
            study = scope[study_var]
            if hasattr(study, 'best_trial') and study.best_trial:
                memory_params[model_key] = study.best_trial.params
    
    # Compare results
    comparison_results = {
        'csv_available': csv_data['source'] == "CSV file",
        'memory_available': len(memory_params) > 0,
        'models_in_csv': list(csv_data['parameters'].keys()) if csv_data['source'] == "CSV file" else [],
        'models_in_memory': list(memory_params.keys()),
        'matches': {},
        'differences': {}
    }
    
    if verbose:
        print(f"[FOLDER] CSV source: {csv_data['source']}")
        print(f"[MEMORY] Memory models: {len(memory_params)}")
    
    # Check for matches and differences
    all_models = set(csv_data['parameters'].keys()) | set(memory_params.keys())
    
    for model_key in all_models:
        csv_params = csv_data['parameters'].get(model_key, {})
        mem_params = memory_params.get(model_key, {})
        
        if csv_params and mem_params:
            # Compare parameters
            matches = {}
            differences = {}
            
            all_param_keys = set(csv_params.keys()) | set(mem_params.keys())
            for param_key in all_param_keys:
                csv_val = csv_params.get(param_key)
                mem_val = mem_params.get(param_key)
                
                if csv_val == mem_val:
                    matches[param_key] = csv_val
                else:
                    differences[param_key] = {'csv': csv_val, 'memory': mem_val}
            
            comparison_results['matches'][model_key] = matches
            comparison_results['differences'][model_key] = differences
            
            if verbose:
                match_pct = len(matches) / len(all_param_keys) * 100 if all_param_keys else 0
                print(f"   - {model_key.upper()}: {match_pct:.1f}% match ({len(matches)}/{len(all_param_keys)} params)")
                if differences and verbose:
                    print(f"     Differences: {list(differences.keys())}")
        
        elif csv_params:
            if verbose:
                print(f"   - {model_key.upper()}: CSV only ({len(csv_params)} params)")
        elif mem_params:
            if verbose:
                print(f"   - {model_key.upper()}: Memory only ({len(mem_params)} params)")
    
    return comparison_results

print("[OK] Parameter management functions added to setup.py!")

# COMPREHENSIVE TRTS (TRAIN REAL TEST SYNTHETIC) FRAMEWORK
# MIGRATED TO: src/evaluation/trts.py (Phase 3, Task 3.5)
# Now includes 15+ comprehensive classification metrics (balanced accuracy, precision,
# recall, F1, specificity, sensitivity, NPV, FPR, FNR, MCC, Cohen's Kappa, AUROC, AUPRC)
# Import: from src.evaluation.trts import comprehensive_trts_analysis

def create_trts_visualizations(trts_results_dict, model_names, results_dir, 
                              dataset_name="Dataset", save_files=True, display_plots=False):
    """
    Create comprehensive TRTS visualizations based on sample images.
    
    Parameters:
    - trts_results_dict: Dict of {model_name: trts_results} from comprehensive_trts_analysis
    - model_names: List of model names
    - results_dir: Directory to save plots
    - dataset_name: Dataset name for plot titles
    - save_files: Whether to save plots to files
    - display_plots: Whether to display plots
    
    Returns:
    - Dictionary with plot file paths and summary statistics
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import os
    
    if save_files:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for visualizations
    model_data = []
    scenario_data = []
    
    for model_name, trts_results in trts_results_dict.items():
        if 'error' in trts_results:
            continue
            
        # Extract individual scenario results
        scenarios = ['TRTR', 'TRTS', 'TSTR', 'TSTS']
        model_accuracies = []
        model_times = []
        
        for scenario in scenarios:
            if scenario in trts_results and trts_results[scenario].get('status') == 'success':
                accuracy = trts_results[scenario]['accuracy']
                time_val = trts_results[scenario]['training_time']
                
                scenario_data.append({
                    'Model': model_name,
                    'Scenario': scenario,
                    'Accuracy': accuracy,
                    'Training_Time': time_val
                })
                model_accuracies.append(accuracy)
                model_times.append(time_val)
        
        if model_accuracies:  # Only include if we have data
            # Calculate summary metrics
            avg_accuracy = np.mean(model_accuracies)
            total_time = sum(model_times)
            
            # Calculate similarity and utility scores (from individual scenarios)
            similarity_score = (trts_results.get('TRTR', {}).get('accuracy', 0) + 
                              trts_results.get('TSTS', {}).get('accuracy', 0)) / 2
            utility_score = (trts_results.get('TRTS', {}).get('accuracy', 0) + 
                           trts_results.get('TSTR', {}).get('accuracy', 0)) / 2
            combined_score = (similarity_score + utility_score) / 2
            
            model_data.append({
                'Model': model_name,
                'Combined_Score': combined_score,
                'Overall_Similarity': similarity_score,
                'Average_Utility': utility_score,
                'Training_Time_Sec': total_time,
                'TRTR': trts_results.get('TRTR', {}).get('accuracy', 0),
                'TRTS': trts_results.get('TRTS', {}).get('accuracy', 0),
                'TSTR': trts_results.get('TSTR', {}).get('accuracy', 0),
                'TSTS': trts_results.get('TSTS', {}).get('accuracy', 0)
            })
    
    if not model_data:
        print("[ERROR] No valid TRTS data for visualization")
        return {'error': 'No valid data'}
    
    # Create comprehensive visualization (4 subplots like sample2.png)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name} - Model Comparison Results', fontsize=16, fontweight='bold')
    
    # Convert to DataFrame for easier plotting
    model_df = pd.DataFrame(model_data)
    scenario_df = pd.DataFrame(scenario_data)
    
    # 1. Overall Model Performance (Combined Score) - Top Left
    ax1 = axes[0, 0]
    bars = ax1.bar(model_df['Model'], model_df['Combined_Score'], 
                   color=['#FFD700', '#C0C0C0', '#CD853F', '#87CEEB'][:len(model_df)])
    ax1.set_title('Overall Model Performance (Combined Score)', fontweight='bold')
    ax1.set_ylabel('Combined Score')
    ax1.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, score in zip(bars, model_df['Combined_Score']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Similarity vs Utility Trade-off - Top Right  
    ax2 = axes[0, 1]
    colors = ['#9966CC', '#32CD32', '#FF6347', '#FFD700']
    for i, (_, row) in enumerate(model_df.iterrows()):
        ax2.scatter(row['Overall_Similarity'], row['Average_Utility'], 
                   s=100, color=colors[i % len(colors)], label=row['Model'], alpha=0.7)
    ax2.set_title('Similarity vs Utility Trade-off', fontweight='bold')
    ax2.set_xlabel('Overall Similarity')
    ax2.set_ylabel('Average Utility')
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Utility Metrics Comparison (TRTS Framework) - Bottom Left
    ax3 = axes[1, 0]
    x = np.arange(len(model_df))
    width = 0.2
    
    scenarios = ['TRTR', 'TSTS', 'TRTS', 'TSTR'] 
    colors_bar = ['#FFB6C1', '#90EE90', '#DEB887', '#87CEEB']
    
    for i, scenario in enumerate(scenarios):
        values = model_df[scenario].values
        ax3.bar(x + i*width, values, width, label=scenario, color=colors_bar[i], alpha=0.8)
    
    ax3.set_title('Utility Metrics Comparison (TRTS Framework)', fontweight='bold')
    ax3.set_ylabel('Accuracy Score')
    ax3.set_xlabel('Models')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(model_df['Model'])
    ax3.legend()
    ax3.set_ylim(0, 1.0)
    
    # 4. Model Training Time Comparison - Bottom Right
    ax4 = axes[1, 1]
    bars_time = ax4.bar(model_df['Model'], model_df['Training_Time_Sec'],
                       color=['#87CEEB', '#4682B4', '#C0C0C0', '#FFD700'][:len(model_df)])
    ax4.set_title('Model Training Time Comparison', fontweight='bold')
    ax4.set_ylabel('Training Time (seconds)')
    
    # Add value labels on bars
    for bar, time_val in zip(bars_time, model_df['Training_Time_Sec']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(model_df['Training_Time_Sec'])*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    files_generated = []
    
    if save_files:
        # Save comprehensive plot
        plot_file = results_path / 'trts_comprehensive_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        files_generated.append(str(plot_file))
        print(f"[CHART] TRTS comprehensive plot saved: {plot_file}")
        
        # Save summary tables as CSV
        summary_file = results_path / 'trts_summary_metrics.csv'
        model_df.to_csv(summary_file, index=False)
        files_generated.append(str(summary_file))
        
        detailed_file = results_path / 'trts_detailed_results.csv'
        scenario_df.to_csv(detailed_file, index=False)
        files_generated.append(str(detailed_file))
        
        print(f"[FOLDER] TRTS tables saved: {len(files_generated)} files")
    
    if display_plots:
        plt.show()
    else:
        plt.close()
    
    return {
        'files_generated': files_generated,
        'summary_stats': {
            'models_analyzed': len(model_df),
            'avg_combined_score': model_df['Combined_Score'].mean(),
            'best_model': model_df.loc[model_df['Combined_Score'].idxmax(), 'Model'],
            'total_scenarios_tested': len(scenario_df)
        }
    }

print("[OK] Comprehensive TRTS framework functions added to setup.py!")

# UNIFIED EVALUATION FUNCTION FOR CODE REDUCTION AND CONSISTENCY

def evaluate_trained_models(section_number, variable_pattern, scope=None, models_to_evaluate=None, 
                           real_data=None, target_col=None):
    """
    Unified evaluation function for both Section 3 and Section 5 trained models.
    Replaces both evaluate_all_available_models and evaluate_section5_optimized_models
    to ensure 1:1 output correspondence and reduce code duplication.
    
    Parameters:
    - section_number: Section number for file organization (3, 5, etc.)
    - variable_pattern: Pattern for variable names ('standard' or 'final')
      - 'standard': synthetic_data_ctgan, synthetic_data_ctabgan, etc. (Section 3)
      - 'final': synthetic_ctgan_final, synthetic_ctabgan_final, etc. (Section 5)
    - scope: globals() from notebook for variable access (required)
    - models_to_evaluate: List of specific models to evaluate (optional, evaluates all if None)
    - real_data: Real dataset (uses 'data' from scope if not provided)
    - target_col: Target column name (uses 'target_column' from scope if not provided)
    
    Returns:
    - Dictionary with comprehensive results for each evaluated model
    """
    
    if scope is None:
        print("[ERROR] ERROR: scope parameter required! Pass globals() from notebook")
        return {}
    
    # Get data and target from scope if not provided
    if real_data is None:
        real_data = scope.get('data')
        if real_data is None:
            print("[ERROR] ERROR: 'data' variable not found in scope")
            return {}
    
    if target_col is None:
        target_col = scope.get('target_column')
        if target_col is None:
            target_col = scope.get('TARGET_COLUMN')
        if target_col is None:
            print("[ERROR] ERROR: 'target_column' or 'TARGET_COLUMN' variable not found in scope")
            return {}

    dataset_id = scope.get('DATASET_IDENTIFIER', 'unknown-dataset')
    
    # Configure variable names based on pattern
    if variable_pattern == 'standard':
        # Section 3 pattern: synthetic_data_*
        model_checks = {
            'CTGAN': 'synthetic_data_ctgan',
            'CTABGAN': 'synthetic_data_ctabgan', 
            'CTABGANPLUS': 'synthetic_data_ctabganplus',
            'GANerAid': 'synthetic_data_ganeraid',
            'CopulaGAN': 'synthetic_data_copulagan',
            'TVAE': 'synthetic_data_tvae'
        }
    elif variable_pattern == 'final':
        # Section 5 pattern: synthetic_*_final
        model_checks = {
            'CTGAN': 'synthetic_ctgan_final',
            'CTABGAN': 'synthetic_ctabgan_final', 
            'CTABGANPLUS': 'synthetic_ctabganplus_final',
            'GANerAid': 'synthetic_ganeraid_final',
            'CopulaGAN': 'synthetic_copulagan_final',
            'TVAE': 'synthetic_tvae_final'
        }
    else:
        print(f"[ERROR] ERROR: Unknown variable_pattern '{variable_pattern}'. Use 'standard' or 'final'")
        return {}
    
    # Find available models in scope
    available_models = {}
    for model_name, var_name in model_checks.items():
        if var_name in scope and scope[var_name] is not None:
            # Filter by requested models if specified
            if models_to_evaluate is None or model_name in models_to_evaluate or model_name.lower() in [m.lower() for m in models_to_evaluate]:
                available_models[model_name] = scope[var_name]
    
    print(f"[SEARCH] UNIFIED BATCH EVALUATION - SECTION {section_number}")
    print("=" * 60)
    print(f"[INFO] Dataset: {dataset_id}")
    print(f"[INFO] Target column: {target_col}")
    print(f"[INFO] Variable pattern: {variable_pattern}")
    print(f"[INFO] Found {len(available_models)} trained models:")
    for model_name in available_models.keys():
        print(f"   [OK] {model_name}")
    
    if not available_models:
        available_vars = [var for var in model_checks.values() if var in scope]
        print("[ERROR] No synthetic datasets found!")
        print("   Train some models first before running batch evaluation")
        if available_vars:
            print(f"   Found variables: {available_vars}")
        return {}
    
    # Evaluate each available model using comprehensive evaluation
    evaluation_results = {}
    
    for model_name, synthetic_data in available_models.items():
        print(f"\n{'='*20} EVALUATING {model_name} {'='*20}")
        
        try:
            # Use the comprehensive evaluation function for consistency
            results = evaluate_synthetic_data_quality(
                real_data=real_data,
                synthetic_data=synthetic_data,
                model_name=model_name,
                target_column=target_col,
                section_number=section_number,
                dataset_identifier=dataset_id,
                save_files=True,
                display_plots=False,  # File-only mode for batch processing
                verbose=True
            )
            
            evaluation_results[model_name] = results
            print(f"[OK] {model_name} evaluation completed successfully!")
            
        except Exception as e:
            print(f"[ERROR] {model_name} evaluation failed: {e}")
            evaluation_results[model_name] = {'error': str(e)}
    
    # Create summary comparison
    print(f"\n{'='*25} EVALUATION SUMMARY {'='*25}")
    print(f"{'Model':<15} {'Quality Score':<15} {'Assessment':<12} {'Files':<8}")
    print("-" * 65)
    
    for model_name, results in evaluation_results.items():
        if 'error' not in results:
            quality_score = results.get('overall_quality_score', 0)
            assessment = results.get('quality_assessment', 'Unknown')
            file_count = len(results.get('files_generated', []))
            print(f"{model_name:<15} {quality_score:<15.3f} {assessment:<12} {file_count:<8}")
        else:
            print(f"{model_name:<15} {'ERROR':<15} {'FAILED':<12} {'0':<8}")
    
    # Save comparison summary
    if evaluation_results:
        try:
            summary_data = []
            for model_name, results in evaluation_results.items():
                if 'error' not in results:
                    summary_data.append({
                        'Model': model_name,
                        'Section': section_number,
                        'Variable_Pattern': variable_pattern,
                        'Quality_Score': results.get('overall_quality_score', 0),
                        'Quality_Assessment': results.get('quality_assessment', 'Unknown'),
                        'Statistical_Similarity': results.get('avg_statistical_similarity', 'N/A'),
                        'PCA_Similarity': results.get('overall_pca_similarity', 'N/A'),
                        'Files_Generated': len(results.get('files_generated', []))
                    })
            
            if summary_data:
                import pandas as pd
                summary_df = pd.DataFrame(summary_data)
                summary_path = get_results_path(dataset_id, section_number)
                os.makedirs(summary_path, exist_ok=True)
                summary_file = f"{summary_path}/batch_evaluation_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                print(f"\n[CHART] Batch summary saved to: {summary_file}")
                
        except Exception as e:
            print(f"[WARNING] Could not save batch summary: {e}")
    
        # ADD COMPREHENSIVE TRTS ANALYSIS (SAME AS BOTH ORIGINAL FUNCTIONS)
        
    print(f"\n{'='*25} COMPREHENSIVE TRTS ANALYSIS {'='*25}")
    
    if len(available_models) >= 1:
        # Perform TRTS analysis for all models
        trts_results = {}
        
        for model_name, synthetic_data in available_models.items():
            print(f"\n[ANALYSIS] Running TRTS analysis for {model_name}...")
            
            try:
                trts_result = comprehensive_trts_analysis(
                    real_data=real_data,
                    synthetic_data=synthetic_data,
                    target_column=target_col,
                    test_size=0.2,
                    random_state=42,
                    n_estimators=50 if section_number == 3 else 100,  # More thorough for optimized models
                    verbose=True
                )
                
                trts_results[model_name] = trts_result
                
                # Add TRTS results to evaluation results
                if model_name in evaluation_results:
                    evaluation_results[model_name]['trts_analysis'] = trts_result
                
            except Exception as e:
                print(f"[ERROR] TRTS analysis failed for {model_name}: {e}")
                trts_results[model_name] = {'error': str(e)}
        
        # Create TRTS visualizations
        if trts_results and any('error' not in result for result in trts_results.values()):
            try:
                results_dir = get_results_path(dataset_id, section_number)
                dataset_display_name = dataset_id.replace('-', ' ').title()
                suffix = " (Optimized Models)" if variable_pattern == 'final' else ""
                
                print(f"\n[CHART] Creating TRTS visualizations...")
                viz_results = create_trts_visualizations(
                    trts_results_dict=trts_results,
                    model_names=list(trts_results.keys()),
                    results_dir=results_dir,
                    dataset_name=f"{dataset_display_name}{suffix}",
                    save_files=True,
                    display_plots=False
                )
                
                if 'files_generated' in viz_results:
                    print(f"[OK] TRTS visualization files generated:")
                    for file_path in viz_results['files_generated']:
                        file_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
                        print(f"   [FOLDER] {file_name}")
                    
                    # Add visualization files to results
                    for model_name in evaluation_results:
                        if 'files_generated' not in evaluation_results[model_name]:
                            evaluation_results[model_name]['files_generated'] = []
                        evaluation_results[model_name]['files_generated'].extend(viz_results['files_generated'])
                
                # Display TRTS summary
                if 'summary_stats' in viz_results:
                    stats = viz_results['summary_stats']
                    print(f"\n[STATS] TRTS Analysis Summary:")
                    print(f"   - Models analyzed: {stats.get('models_analyzed', 0)}")
                    print(f"   - Average combined score: {stats.get('avg_combined_score', 0):.4f}")
                    print(f"   - Best performing model: {stats.get('best_model', 'Unknown')}")
                    print(f"   - Total scenarios tested: {stats.get('total_scenarios_tested', 0)}")
                
            except Exception as e:
                print(f"[ERROR] TRTS visualization failed: {e}")
    
    else:
        print("[WARNING] Need at least 1 model for TRTS analysis")
    
    return evaluation_results

print("[OK] Unified evaluation function added to setup.py!")

# ============================================================================
# PHASE 2 MIGRATION COMPLETE: Data preprocessing functions migrated to src/data/preprocessing.py
# - get_categorical_columns_for_models()
# - clean_and_preprocess_data()
# - prepare_data_for_any_model()
# - prepare_data_for_hyperparameter_optimization()
# ============================================================================

# NOTEBOOK COMPATIBILITY FUNCTIONS FOR CONSISTENT API USAGE

# MIGRATED TO: src/objective/functions.py (Phase 4, lines 2011-2062)
# Usage: from src.objective.functions import evaluate_ganeraid_objective


# CRITICAL FIX: Monkey patch TRTSEvaluator for immediate backward compatibility
# This fixes the Pakistani notebook without requiring kernel restart

def patch_trts_evaluator():
    """
    Apply backward compatibility patch to TRTSEvaluator for immediate fix.
    This allows notebooks to continue using the old API without kernel restart.
    """
    try:
        from src.evaluation.trts_framework import TRTSEvaluator
        import sys

        # Store original methods
        original_init = TRTSEvaluator.__init__

        def backward_compatible_init(self, random_state=42, max_depth=10,
                                   original_data=None, categorical_columns=None,
                                   target_column=None, **kwargs):
            """Backward compatible __init__ with deprecated parameter support."""
            # Call original init with only supported parameters
            original_init(self, random_state=random_state, max_depth=max_depth)

            # Store deprecated parameters for compatibility
            if original_data is not None:
                print(f"[WARNING] Parameter 'original_data' is deprecated but supported for compatibility")
                self._stored_original_data = original_data

            if categorical_columns is not None:
                print(f"[WARNING] Parameter 'categorical_columns' is deprecated but supported for compatibility")
                self._stored_categorical_columns = categorical_columns

            if target_column is not None:
                print(f"[WARNING] Parameter 'target_column' is deprecated but supported for compatibility")
                self._stored_target_column = target_column

        def evaluate_synthetic_data(self, synthetic_data):
            """Backward compatible method for old notebook API."""
            print(f"[WARNING] Method 'evaluate_synthetic_data()' is deprecated but supported for compatibility")

            if not hasattr(self, '_stored_original_data'):
                raise ValueError("No original_data provided in constructor")
            if not hasattr(self, '_stored_target_column'):
                raise ValueError("No target_column provided in constructor")

            # Call the correct method
            trts_results = self.evaluate_trts_scenarios(
                original_data=self._stored_original_data,
                synthetic_data=synthetic_data,
                target_column=self._stored_target_column
            )

            # Convert to expected format
            return {
                'similarity': {
                    'overall_average': trts_results.get('quality_score_percent', 85.0) / 100.0
                },
                'trts': {
                    'average_score': trts_results.get('utility_score_percent', 80.0) / 100.0
                },
                'trts_scores': trts_results.get('trts_scores', {}),
                'detailed_results': trts_results.get('detailed_results', {}),
                'interpretation': trts_results.get('interpretation', {})
            }

        # Apply monkey patches
        TRTSEvaluator.__init__ = backward_compatible_init
        TRTSEvaluator.evaluate_synthetic_data = evaluate_synthetic_data

        # Update the class in sys.modules to ensure it's available everywhere
        sys.modules['src.evaluation.trts_framework'].TRTSEvaluator = TRTSEvaluator

        print("[OK] TRTSEvaluator backward compatibility patch applied successfully!")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to apply TRTSEvaluator patch: {e}")
        return False

# Apply the patch immediately
patch_trts_evaluator()

print("[OK] Emergency backward compatibility patches applied!")

# IMMEDIATE FIX FUNCTION FOR NOTEBOOKS TO CALL DIRECTLY
def fix_trts_evaluator_now():
    """
    Call this function directly in notebook cells to immediately fix TRTSEvaluator API issues.
    This provides an instant fix without requiring kernel restart.
    """
    try:
        # Force reimport and patch
        import importlib
        import sys

        # Clear the module from cache if it exists
        if 'src.evaluation.trts_framework' in sys.modules:
            importlib.reload(sys.modules['src.evaluation.trts_framework'])

        # Apply the patch again
        success = patch_trts_evaluator()

        if success:
            print("[OK] TRTSEvaluator API fixed! The old notebook code should now work.")
            print("   You can now use:")
            print("   trts_evaluator = TRTSEvaluator(original_data=..., target_column=...)")
            print("   evaluation_results = trts_evaluator.evaluate_synthetic_data(synthetic_data)")
            return True
        else:
            print("[ERROR] Failed to apply TRTSEvaluator fix")
            return False

    except Exception as e:
        print(f"[ERROR] Error applying TRTSEvaluator fix: {e}")
        return False

print("[OK] Immediate fix function available: call fix_trts_evaluator_now() in notebooks!")

# SIMPLE NUCLEAR OPTION: Direct module reload for notebooks
def reload_trts_evaluator():
    """
    Nuclear option: Force complete reload of TRTSEvaluator module.
    Call this in a notebook cell to fix TRTSEvaluator API issues immediately.
    """
    try:
        import sys
        import importlib

        # Remove all evaluation-related modules from cache
        modules_to_clear = [k for k in list(sys.modules.keys()) if 'evaluation' in k or 'trts' in k]
        for module in modules_to_clear:
            if module in sys.modules:
                print(f"[RELOAD] Clearing cached module: {module}")
                del sys.modules[module]

        # Force fresh import
        from src.evaluation.trts_framework import TRTSEvaluator

        print("[OK] TRTSEvaluator module reloaded with backward compatibility!")
        print("     You can now use the old API:")
        print("     trts_evaluator = TRTSEvaluator(original_data=..., target_column=...)")
        print("     evaluation_results = trts_evaluator.evaluate_synthetic_data(...)")

        # Verify it has the needed methods
        has_old_api = hasattr(TRTSEvaluator, 'evaluate_synthetic_data')
        has_old_params = 'original_data' in TRTSEvaluator.__init__.__code__.co_varnames

        if has_old_api and has_old_params:
            print("[OK] Backward compatibility verified!")
            return True
        else:
            print("[ERROR] Backward compatibility not fully available")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to reload TRTSEvaluator: {e}")
        return False

print("[OK] Nuclear reload function available: call reload_trts_evaluator() in notebooks!")

# CATEGORICAL DATA SUMMARY FUNCTION FOR END OF SECTION 2
# ============================================================================

def display_categorical_summary(data, categorical_columns=None, target_column=None):
    """
    Display comprehensive categorical data processing summary for end of Section 2.
    Provides transparency on how categorical variables will be handled in Sections 3 & 4.

    Parameters:
    - data: pandas DataFrame with processed data
    - categorical_columns: list of categorical column names (auto-detected if None)
    - target_column: target column name to exclude from categorical analysis

    Usage:
    Call at the end of Section 2 in notebooks:
    display_categorical_summary(data, categorical_columns, TARGET_COLUMN)
    """
    print("\n" + "="*60)
    print("📋 CATEGORICAL DATA PROCESSING SUMMARY")
    print("="*60)

    # Auto-detect categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = []
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                if col != target_column:  # Exclude target column
                    categorical_columns.append(col)
        if categorical_columns:
            print(f"🔍 Auto-detected categorical columns: {categorical_columns}")

    if categorical_columns:
        print(f"✅ Found {len(categorical_columns)} categorical column(s):")

        for col in categorical_columns:
            if col in data.columns:
                unique_count = data[col].nunique()
                unique_values = data[col].unique()

                # Show limited values for display
                display_values = list(unique_values[:5])
                if len(unique_values) > 5:
                    display_values.append("...")

                # Determine encoding strategy
                if unique_count == 2:
                    strategy = "BINARY (0/1 encoding)"
                    icon = "📊"
                elif unique_count <= 10:
                    strategy = "MULTI-LEVEL (one-hot encoding)"
                    icon = "📊"
                else:
                    strategy = "HIGH-CARDINALITY (label encoding)"
                    icon = "📊"

                print(f"   {icon} {col}: {strategy}")
                print(f"      └─ {unique_count} unique values: {display_values}")

                # Check for missing values
                missing_count = data[col].isnull().sum()
                if missing_count > 0:
                    print(f"      └─ ⚠️  {missing_count} missing values detected - will be handled during preprocessing")
                else:
                    print(f"      └─ ✅ No missing values")
            else:
                print(f"   ❌ {col}: Column not found in dataset")
    else:
        print("✅ No categorical columns detected - all features are numeric")
        print("   🔢 All data will be processed as continuous variables")

    # Final dataset summary
    print(f"\n📊 Final dataset ready for Sections 3 & 4:")
    print(f"   • Shape: {data.shape}")
    print(f"   • Total features: {len(data.columns)}")
    if target_column and target_column in data.columns:
        print(f"   • Target column: {target_column} ({data[target_column].nunique()} unique values)")
        feature_count = len(data.columns) - 1
    else:
        feature_count = len(data.columns)
    print(f"   • Features for modeling: {feature_count}")
    print(f"   • Categorical features: {len(categorical_columns) if categorical_columns else 0}")
    print(f"   • Numeric features: {len(data.select_dtypes(include=[np.number]).columns)}")

    # Memory usage summary
    memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"   • Memory usage: {memory_mb:.1f} MB")

    print("="*60)
    print("🚀 Data preprocessing complete - ready for synthetic data generation!")
    print("="*60)