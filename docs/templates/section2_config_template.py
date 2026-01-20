# ============================================================================
# SECTION 2: CONFIG-DRIVEN DATA LOADING AND PREPROCESSING
# ============================================================================
#
# This template demonstrates the new January 2026 config-driven workflow.
# Copy this into your notebook's Section 2 to use the standardized approach.
#
# BEFORE USING: Run Section 1 to import setup.py
# ============================================================================

# --- 2.1 NOTEBOOK CONFIGURATION ---
# Edit ONLY this block for your dataset. All other sections will use this config.

NOTEBOOK_CONFIG = {
    # ========== REQUIRED: Dataset Settings ==========
    "data_file": "data/Breast_cancer_data.csv",  # Path to your CSV file
    "target_column": "diagnosis",                 # Target/outcome column name

    # ========== OPTIONAL: Dataset Metadata ==========
    "dataset_name": "Breast Cancer Dataset",      # Display name (optional)
    "dataset_identifier_override": None,          # Override auto-detected identifier (or None)

    # ========== OPTIONAL: Column Configuration ==========
    "categorical_columns": [],                    # List of categorical columns, or [] for auto-detect
    "task_type": "auto",                          # "auto" | "classification" | "regression"

    # ========== OPTIONAL: Data Subsetting ==========
    "use_row_subset": True,                       # True to sample rows for faster testing
    "sample_n": 500,                              # Number of rows to sample
    "sample_random_state": 42,                    # Random seed for reproducibility

    # ========== OPTIONAL: Missing Data Handling ==========
    "missing_strategy": "none",                   # "none" | "drop" | "median" | "mode" | "mice" | "indicator_onehot"
    "mice_max_iter": 10,                          # Max iterations for MICE imputation

    # ========== OPTIONAL: Model Selection ==========
    "models_to_run": "all",                       # "all" or list like ["ctgan", "tvae", "ctabganplus"]

    # ========== OPTIONAL: Tuning Configuration ==========
    "tuning_mode": "smoke",                       # "smoke" (5 trials) | "full" (50 trials)
    "n_trials_smoke": 5,                          # Trials for smoke testing
    "n_trials_full": 50,                          # Trials for full optimization
    "timeout_seconds": None,                      # Optional timeout per study
}

# --- 2.2 VALIDATE AND LOAD DATA ---
# This uses the config-driven preprocessing pipeline

# Validate configuration
NOTEBOOK_CONFIG = validate_config(NOTEBOOK_CONFIG)

# Load and preprocess data using the config
(
    data,                  # Processed DataFrame
    original_data,         # Copy for reference
    target_column,         # Target column name (cleaned)
    DATASET_IDENTIFIER,    # Dataset identifier for results paths
    categorical_columns,   # List of categorical columns
    metadata               # Full preprocessing metadata
) = load_and_preprocess_from_config(NOTEBOOK_CONFIG)

# Set aliases for backward compatibility
TARGET_COLUMN = target_column

# --- 2.3 DISPLAY CONFIGURATION SUMMARY ---
print("\n" + "="*60)
print("CONFIGURATION SUMMARY")
print("="*60)
print(f"Dataset: {NOTEBOOK_CONFIG.get('dataset_name', DATASET_IDENTIFIER)}")
print(f"Identifier: {DATASET_IDENTIFIER}")
print(f"Data shape: {data.shape}")
print(f"Target column: {target_column}")
print(f"Task type: {metadata['task_type']}")
print(f"Categorical columns: {categorical_columns}")
print(f"Missing strategy: {NOTEBOOK_CONFIG['missing_strategy']}")
print(f"Models to run: {NOTEBOOK_CONFIG['models_to_run']}")
print(f"Tuning mode: {NOTEBOOK_CONFIG['tuning_mode']} ({get_n_trials(NOTEBOOK_CONFIG)} trials)")
print(f"Session timestamp: {SESSION_TIMESTAMP}")
print(f"Results path: {get_results_path(DATASET_IDENTIFIER, 2)}")
print("="*60 + "\n")

# --- 2.4 PREVIEW DATA ---
print("Data Preview:")
print(data.head())

print(f"\nTarget Distribution ({target_column}):")
print(data[target_column].value_counts())

# Check for missing values
missing = data.isnull().sum()
if missing.sum() > 0:
    print(f"\nMissing Values:")
    print(missing[missing > 0])
else:
    print(f"\nNo missing values in processed data.")

# --- 2.5 GET MODELS TO RUN ---
# This resolves the models_to_run config to actual model list
models_to_run = get_models_to_run(NOTEBOOK_CONFIG)
print(f"\nModels that will be run: {models_to_run}")

# Get tuning configuration
tuning_config = get_tuning_config(NOTEBOOK_CONFIG)
print(f"Tuning config: {tuning_config}")


# ============================================================================
# SECTION 2 COMPLETE
# ============================================================================
# The following variables are now available for Sections 3-5:
#   - data: Processed DataFrame
#   - original_data: Copy of data
#   - target_column / TARGET_COLUMN: Target column name
#   - DATASET_IDENTIFIER: For results paths
#   - categorical_columns: List of categorical columns
#   - NOTEBOOK_CONFIG: Full configuration dict
#   - models_to_run: List of models to run
#   - tuning_config: Tuning settings
# ============================================================================
