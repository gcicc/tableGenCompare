"""
Backward-compatible import layer for notebooks.

This module serves as a thin re-export layer, allowing notebooks to continue
using 'from setup import *' while the actual implementation has been moved to
the modular src/ directory structure.

All functionality is re-exported from src/ modules for backward compatibility.

Migrated to src/ modules in Phase 4, Task 4.3 - reducing setup.py to <100 lines.
"""
import os

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
    refresh_session_timestamp,
    # Notebook config schema (Phase 1)
    NOTEBOOK_CONFIG_DEFAULTS,
    get_default_config,
    validate_config,
)

# Import path utilities
from src.utils.paths import (
    extract_dataset_identifier,
    get_results_path,
    ensure_results_dir,
    save_model_params,
    save_evaluation_summary,
    save_synthetic_sample,
    save_comparison_table,
    get_latest_results_path
)

# Import documentation utilities
from src.utils.documentation import (
    create_section2_readme,
    create_section3_main_readme,
    create_section3_model_readme,
    create_section5_readme
)

# Import parameter management utilities (Phase 4)
from src.utils.parameters import (
    save_best_parameters_to_csv,
    load_best_parameters_from_csv,
    get_model_parameters,
    compare_parameters_sources
)

# Import visualization functions
from src.visualization.section2 import (
    create_correlation_heatmap,  # backward-compatible alias
    create_mixed_association_heatmap,
    create_feature_distributions
)
from src.visualization.section3 import (
    create_correlation_comparison,  # backward-compatible alias
    create_association_comparison,
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
    create_trts_visualizations,
    create_privacy_dashboard
)

# Import evaluation functions
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
from src.evaluation.hyperparameters import (
    evaluate_hyperparameter_optimization_results,
    analyze_hyperparameter_optimization
)

# Import objective functions
from src.objective.functions import (
    enhanced_objective_function_v2,
    evaluate_ganeraid_objective
)

# Import model classes and wrappers
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

from src.models.model_factory import (
    ModelFactory
)

# Import batch training and optimization functions (Phase 5)
from src.models.batch_training import (
    train_models_batch,
    extract_synthetic_data_to_globals,
    train_models_batch_with_best_params,
    extract_final_synthetic_to_globals,
)

from src.models.search_spaces import (
    get_search_space,
    get_pruner_config,
    list_supported_models,
)

from src.models.batch_optimization import (
    optimize_models_batch,
    extract_studies_to_globals,
)

from src.models.staged_optimization import flush_previous_runs

# Import checkpoint system for notebook resume capability
from src.utils.checkpoint import SectionCheckpoint

# Import model registry and selection helpers (Phase 4)
from src.models.registry import (
    AVAILABLE_MODELS,
    get_available_model_names,
    resolve_models,
    get_models_to_run,
    get_model_display_name,
    is_model_available,
    get_tuning_config,
    get_n_trials as get_n_trials_from_registry
)

# Import data preprocessing functions
from src.data.preprocessing import (
    get_categorical_columns_for_models,
    clean_and_preprocess_data,
    prepare_data_for_any_model,
    prepare_data_for_hyperparameter_optimization,
    # Config-driven preprocessing (Phase 3)
    preprocess_dataset,
    load_and_preprocess_from_config
)

# Import EDA functions (Phase 5 - streamlined)
from src.data.eda import (
    run_comprehensive_eda,
)

# ============================================================================
# BACKWARD COMPATIBILITY - Thin wrapper functions
# ============================================================================

def evaluate_all_available_models(section_number, scope=None, models_to_evaluate=None, real_data=None, target_col=None,
                                  protected_col=None, compute_mia=False):
    """
    Wrapper for unified evaluate_trained_models function - Section 3 pattern.

    Parameters:
    - section_number: Section number for file organization (3, 5, etc.)
    - scope: globals() from notebook for variable access (required for notebook use)
    - models_to_evaluate: List of specific models to evaluate (optional, evaluates all if None)
    - real_data: Real dataset (uses 'data' from scope if not provided)
    - target_col: Target column name (uses 'target_column' from scope if not provided)
    - protected_col: Protected attribute column for fairness metrics (None = skip fairness)
    - compute_mia: Whether to run MIA evaluation (expensive, default False)

    Returns:
    - Dictionary with results for each evaluated model
    """
    from src.evaluation.batch import evaluate_trained_models
    return evaluate_trained_models(
        section_number=section_number,
        variable_pattern='standard',  # Uses synthetic_data_* variables
        scope=scope,
        models_to_evaluate=models_to_evaluate,
        real_data=real_data,
        target_col=target_col,
        protected_col=protected_col,
        compute_mia=compute_mia
    )


def evaluate_section5_optimized_models(section_number=5, scope=None, target_column=None,
                                       protected_col=None, compute_mia=False):
    """
    Wrapper for unified evaluate_trained_models function - Section 5 pattern.

    Parameters:
    - section_number: Section number for file organization (default 5)
    - scope: Notebook scope (globals()) to access synthetic data and results
    - target_column: Target column name for analysis
    - protected_col: Protected attribute column for fairness metrics (None = skip fairness)
    - compute_mia: Whether to run MIA evaluation (expensive, default False)

    Returns:
    - Dictionary with batch evaluation results including:
      - models_processed: Count of successfully processed models
      - results_dir: Path to results directory
      - evaluation_summaries: Per-model summary dict
      - raw_results: Full evaluation results for backward compatibility
    """
    from src.evaluation.batch import evaluate_trained_models
    from src.utils.paths import get_results_path

    results = evaluate_trained_models(
        section_number=section_number,
        variable_pattern='final',  # Uses synthetic_*_final variables
        scope=scope,
        models_to_evaluate=None,
        real_data=None,
        target_col=target_column,
        protected_col=protected_col,
        compute_mia=compute_mia
    )

    # Restructure for notebook compatibility
    successful = {k: v for k, v in results.items() if 'error' not in v}
    dataset_id = scope.get('DATASET_IDENTIFIER', 'unknown') if scope else 'unknown'

    return {
        'models_processed': len(successful),
        'results_dir': get_results_path(dataset_id, section_number),
        'evaluation_summaries': {
            model: {
                'synthetic_samples': res.get('synthetic_data_shape', (0,))[0] if res.get('synthetic_data_shape') else 0,
                'overall_score': res.get('overall_quality_score', 0)
            }
            for model, res in successful.items()
        },
        'raw_results': results  # Keep full results for backward compat
    }


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
    from src.data.summary import display_categorical_data_summary
    return display_categorical_data_summary(data, categorical_columns, target_column)


# ============================================================================
# BACKWARD COMPATIBILITY PATCHES - Import from compat module
# ============================================================================

from src.compat import (
    patch_trts_evaluator,
    fix_trts_evaluator_now,
    reload_trts_evaluator
)

# Apply backward compatibility patches on import
patch_trts_evaluator()

# ============================================================================
# INITIALIZATION
# ============================================================================

print("[SETUP] Thin re-export layer loaded successfully!")
print(f"[SETUP] Session timestamp: {SESSION_TIMESTAMP}")
print(f"[SETUP] All functions imported from modular src/ structure")
print(f"[SETUP] Backward compatibility maintained for notebooks")