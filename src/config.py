"""
Global Configuration and Session Management

This module manages global configuration state for the Clinical Synthetic
Data Generation Framework, including session timestamps, dataset identifiers,
and notebook configuration schema.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import warnings

# Session timestamp (captured at import time)
SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d")

# Global variables to be set when data is loaded
DATASET_IDENTIFIER = None
CURRENT_DATA_FILE = None


# ============================================================================
# NOTEBOOK CONFIGURATION SCHEMA
# ============================================================================

NOTEBOOK_CONFIG_DEFAULTS: Dict[str, Any] = {
    # Dataset
    "data_file": None,                      # path to CSV (required)
    "dataset_name": None,                   # display name only
    "dataset_identifier_override": None,    # override auto-detected identifier

    # Target
    "target_column": None,                  # target column name (required)
    "categorical_columns": [],              # optional override; else auto-detect
    "task_type": "auto",                    # auto | classification | regression

    # Subsetting
    "use_row_subset": True,                 # True -> sample to sample_n rows
    "sample_n": 500,                        # number of rows to sample
    "sample_random_state": 42,              # random state for reproducibility

    # Missingness strategy
    "missing_strategy": "none",             # none | drop | median | mode | mice | indicator_onehot
    "mice_max_iter": 10,                    # max iterations for MICE imputation

    # Encoding
    "encoding_strategy": "auto",            # auto | onehot | ordinal

    # Models selection
    "models_to_run": "all",                 # "all" or list like ["ctgan", "ctabganplus"]

    # Tuning
    "tuning_mode": "smoke",                 # smoke | full
    "n_trials_smoke": 5,                    # trials for smoke testing
    "n_trials_full": 50,                    # trials for full optimization
    "timeout_seconds": None,                # optional timeout per study
}

# Valid values for enum-like fields
_VALID_TASK_TYPES = {"auto", "classification", "regression"}
_VALID_MISSING_STRATEGIES = {"none", "drop", "median", "mode", "mice", "indicator_onehot"}
_VALID_ENCODING_STRATEGIES = {"auto", "onehot", "ordinal"}
_VALID_TUNING_MODES = {"smoke", "full"}


def get_default_config() -> Dict[str, Any]:
    """
    Return a fresh copy of the default notebook configuration.

    Returns:
        Dict with all default configuration values
    """
    return NOTEBOOK_CONFIG_DEFAULTS.copy()


def validate_config(config: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """
    Validate and normalize a notebook configuration dictionary.

    Fills in missing keys with defaults, validates enum values, and checks
    required fields.

    Parameters:
        config: User-provided configuration dictionary
        strict: If True, raise errors for invalid values; if False, warn and use defaults

    Returns:
        Validated configuration dictionary with all keys present

    Raises:
        ValueError: If strict=True and validation fails
    """
    # Start with defaults and overlay user config
    validated = get_default_config()
    validated.update(config)

    errors = []
    warnings_list = []

    # Check required fields
    if validated["data_file"] is None:
        errors.append("'data_file' is required but not provided")

    if validated["target_column"] is None:
        errors.append("'target_column' is required but not provided")

    # Validate task_type
    if validated["task_type"] not in _VALID_TASK_TYPES:
        msg = f"Invalid task_type '{validated['task_type']}'. Must be one of {_VALID_TASK_TYPES}"
        if strict:
            errors.append(msg)
        else:
            warnings_list.append(msg + ". Using 'auto'.")
            validated["task_type"] = "auto"

    # Validate missing_strategy
    if validated["missing_strategy"] not in _VALID_MISSING_STRATEGIES:
        msg = f"Invalid missing_strategy '{validated['missing_strategy']}'. Must be one of {_VALID_MISSING_STRATEGIES}"
        if strict:
            errors.append(msg)
        else:
            warnings_list.append(msg + ". Using 'none'.")
            validated["missing_strategy"] = "none"

    # Validate encoding_strategy
    if validated["encoding_strategy"] not in _VALID_ENCODING_STRATEGIES:
        msg = f"Invalid encoding_strategy '{validated['encoding_strategy']}'. Must be one of {_VALID_ENCODING_STRATEGIES}"
        if strict:
            errors.append(msg)
        else:
            warnings_list.append(msg + ". Using 'auto'.")
            validated["encoding_strategy"] = "auto"

    # Validate tuning_mode
    if validated["tuning_mode"] not in _VALID_TUNING_MODES:
        msg = f"Invalid tuning_mode '{validated['tuning_mode']}'. Must be one of {_VALID_TUNING_MODES}"
        if strict:
            errors.append(msg)
        else:
            warnings_list.append(msg + ". Using 'smoke'.")
            validated["tuning_mode"] = "smoke"

    # Validate models_to_run
    if validated["models_to_run"] != "all" and not isinstance(validated["models_to_run"], list):
        msg = f"'models_to_run' must be 'all' or a list of model names"
        if strict:
            errors.append(msg)
        else:
            warnings_list.append(msg + ". Using 'all'.")
            validated["models_to_run"] = "all"

    # Validate numeric fields
    if not isinstance(validated["sample_n"], int) or validated["sample_n"] <= 0:
        msg = "'sample_n' must be a positive integer"
        if strict:
            errors.append(msg)
        else:
            warnings_list.append(msg + ". Using 500.")
            validated["sample_n"] = 500

    if not isinstance(validated["n_trials_smoke"], int) or validated["n_trials_smoke"] <= 0:
        msg = "'n_trials_smoke' must be a positive integer"
        if strict:
            errors.append(msg)
        else:
            warnings_list.append(msg + ". Using 5.")
            validated["n_trials_smoke"] = 5

    if not isinstance(validated["n_trials_full"], int) or validated["n_trials_full"] <= 0:
        msg = "'n_trials_full' must be a positive integer"
        if strict:
            errors.append(msg)
        else:
            warnings_list.append(msg + ". Using 50.")
            validated["n_trials_full"] = 50

    # Validate timeout_seconds if provided
    if validated["timeout_seconds"] is not None:
        if not isinstance(validated["timeout_seconds"], (int, float)) or validated["timeout_seconds"] <= 0:
            msg = "'timeout_seconds' must be a positive number or None"
            if strict:
                errors.append(msg)
            else:
                warnings_list.append(msg + ". Setting to None.")
                validated["timeout_seconds"] = None

    # Issue warnings
    for w in warnings_list:
        warnings.warn(w, UserWarning)

    # Raise errors if strict mode
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return validated


def get_n_trials(config: Dict[str, Any]) -> int:
    """
    Get the number of trials based on tuning_mode.

    Parameters:
        config: Validated configuration dictionary

    Returns:
        Number of trials to use for optimization
    """
    if config.get("tuning_mode") == "full":
        return config.get("n_trials_full", 50)
    return config.get("n_trials_smoke", 5)


def refresh_session_timestamp():
    """Refresh the session timestamp to current date"""
    global SESSION_TIMESTAMP
    SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d")
    print(f"Session timestamp refreshed to: {SESSION_TIMESTAMP}")
    return SESSION_TIMESTAMP


print(f"[CONFIG] Session timestamp: {SESSION_TIMESTAMP}")
