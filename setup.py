"""
Backward-compatible import layer for notebooks.

This module serves as a thin re-export layer, allowing notebooks to continue
using 'from setup import *' while the actual implementation has been moved to
the modular src/ directory structure.

All functionality is re-exported from src/ modules for backward compatibility.

Migrated to src/ modules in Phase 4, Task 4.3 - reducing setup.py to <100 lines.
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

# Import parameter management utilities (Phase 4)
from src.utils.parameters import (
    save_best_parameters_to_csv,
    load_best_parameters_from_csv,
    get_model_parameters,
    compare_parameters_sources
)

# Import visualization functions
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

# Import data preprocessing functions
from src.data.preprocessing import (
    get_categorical_columns_for_models,
    clean_and_preprocess_data,
    prepare_data_for_any_model,
    prepare_data_for_hyperparameter_optimization
)

# ============================================================================
# BACKWARD COMPATIBILITY - Thin wrapper functions
# ============================================================================

def evaluate_all_available_models(section_number, scope=None, models_to_evaluate=None, real_data=None, target_col=None):
    """
    Wrapper for unified evaluate_trained_models function - Section 3 pattern.

    Parameters:
    - section_number: Section number for file organization (3, 5, etc.)
    - scope: globals() from notebook for variable access (required for notebook use)
    - models_to_evaluate: List of specific models to evaluate (optional, evaluates all if None)
    - real_data: Real dataset (uses 'data' from scope if not provided)
    - target_col: Target column name (uses 'target_column' from scope if not provided)

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
        target_col=target_col
    )


def evaluate_section5_optimized_models(section_number=5, scope=None, target_column=None):
    """
    Wrapper for unified evaluate_trained_models function - Section 5 pattern.

    Parameters:
    - section_number: Section number for file organization (default 5)
    - scope: Notebook scope (globals()) to access synthetic data and results
    - target_column: Target column name for analysis

    Returns:
    - Dictionary with batch evaluation results and file paths
    """
    from src.evaluation.batch import evaluate_trained_models
    return evaluate_trained_models(
        section_number=section_number,
        variable_pattern='final',  # Uses synthetic_*_final variables
        scope=scope,
        models_to_evaluate=None,
        real_data=None,
        target_col=target_column
    )


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