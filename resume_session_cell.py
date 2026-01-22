# ============================================================================
# RESUME SESSION - Load persisted state from Sections 1-4
# ============================================================================
# Run this cell to restore notebook state and continue with Section 5
#
# This cell loads:
#   1. Setup imports and configuration
#   2. Data preprocessing from Section 2
#   3. Best parameters from CSV files (Section 4 HPO results)
# ============================================================================

print("=" * 80)
print("🔄 RESUMING SESSION FROM PERSISTED STATE")
print("=" * 80)

# ---- Step 1: Core imports and setup ----
print("\n📦 Step 1: Loading core imports...")
from setup import *
import os
import pandas as pd

# ---- Step 2: Notebook configuration (same as Section 1) ----
print("\n⚙️  Step 2: Loading notebook configuration...")

NOTEBOOK_CONFIG = {
    "data_file": "data/Breast_cancer_data.csv",
    "target_column": "diagnosis",
    "dataset_name": "Breast Cancer Dataset",
    "dataset_identifier_override": None,
    "categorical_columns": [],
    "task_type": "auto",
    "use_row_subset": False,
    "sample_n": 500,
    "sample_random_state": 42,
}

# ---- Step 3: Load and preprocess data (same as Section 2) ----
print("\n📊 Step 3: Loading and preprocessing data...")

(
    data,
    original_data,
    target_column,
    DATASET_IDENTIFIER,
    categorical_columns,
    metadata
) = load_and_preprocess_from_config(NOTEBOOK_CONFIG)

TARGET_COLUMN = target_column

print(f"   ✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"   ✅ Target column: {TARGET_COLUMN}")
print(f"   ✅ Dataset identifier: {DATASET_IDENTIFIER}")

# ---- Step 4: Configure persist paths ----
print("\n📁 Step 4: Configuring persist directory paths...")

PERSIST_BASE = "/home/ec2-user/SageMaker/persist"
PERSIST_RESULTS = f"{PERSIST_BASE}/results/2026-01-21"

print(f"   • Results dir: {PERSIST_RESULTS}")

# ---- Step 5: Load best parameters from CSV ----
print("\n📋 Step 5: Loading best parameters from CSV...")

params_csv = f"{PERSIST_RESULTS}/Section-4/best_parameters.csv"
summary_csv = f"{PERSIST_RESULTS}/Section-4/hyperparameter_optimization_summary.csv"

LOADED_BEST_PARAMS = {}

if os.path.exists(params_csv):
    params_df = pd.read_csv(params_csv)

    # Parse parameters by model
    for model_name in params_df['model_name'].unique():
        model_df = params_df[params_df['model_name'] == model_name]
        model_params = {}

        for _, row in model_df.iterrows():
            if row.get('is_component', False):
                continue  # Skip component entries

            param_name = row['parameter_name']
            param_value = row['parameter_value']
            param_type = row['parameter_type']

            # Type conversion
            if param_type == 'int':
                param_value = int(param_value)
            elif param_type == 'float':
                param_value = float(param_value)
            elif param_type == 'tuple':
                param_value = eval(param_value)
            elif param_type == 'bool':
                param_value = str(param_value).lower() in ['true', '1']

            model_params[param_name] = param_value

        model_key = model_name.lower().replace('-', '').replace('+', 'plus').replace(' ', '')
        LOADED_BEST_PARAMS[model_key] = model_params
        print(f"   ✅ {model_name}: {len(model_params)} parameters")

    # Display summary
    if os.path.exists(summary_csv):
        print("\n   📊 HPO Summary (from Section 4):")
        print("   " + "-" * 65)
        summary_df = pd.read_csv(summary_csv)
        for _, row in summary_df.iterrows():
            model = row['model']
            score = row['best_score']
            completed = row['completed_trials']
            total = row['total_trials']
            print(f"   {model:12s} | Score: {score:.4f} | Trials: {completed}/{total}")
else:
    print(f"   ⚠️  Parameters CSV not found at {params_csv}")

# ---- Step 6: Create mock study objects for Section 5 compatibility ----
print("\n🔧 Step 6: Creating study-like objects for Section 5 compatibility...")

class MockTrial:
    """Mock trial object that mimics Optuna's best_trial interface."""
    def __init__(self, params, value, number):
        self.params = params
        self.value = value
        self.number = number

class MockStudy:
    """Mock study object that mimics Optuna's study interface."""
    def __init__(self, model_name, params, score, trial_number):
        self.study_name = model_name
        self.best_trial = MockTrial(params, score, trial_number)
        self.trials = [self.best_trial]  # Minimal trials list

    @property
    def best_params(self):
        return self.best_trial.params

    @property
    def best_value(self):
        return self.best_trial.value

# Load scores from summary CSV
scores_by_model = {}
trial_numbers = {}
if os.path.exists(summary_csv):
    summary_df = pd.read_csv(summary_csv)
    for _, row in summary_df.iterrows():
        model_key = row['model'].lower().replace('-', '').replace('+', 'plus').replace(' ', '')
        scores_by_model[model_key] = row['best_score']
        trial_numbers[model_key] = row['best_trial_number']

# Create mock studies
STUDY_VARS = {
    'ctgan': 'ctgan_study',
    'ctabgan': 'ctabgan_study',
    'ctabganplus': 'ctabganplus_study',
    'ganeraid': 'ganeraid_study',
    'copulagan': 'copulagan_study',
    'tvae': 'tvae_study',
    'pategan': 'pategan_study',
    'medgan': 'medgan_study',
}

for model_key, var_name in STUDY_VARS.items():
    if model_key in LOADED_BEST_PARAMS:
        params = LOADED_BEST_PARAMS[model_key]
        score = scores_by_model.get(model_key, 0.0)
        trial_num = trial_numbers.get(model_key, 0)
        globals()[var_name] = MockStudy(model_key, params, score, trial_num)
        print(f"   ✅ {var_name} created (score={score:.4f})")
    else:
        globals()[var_name] = None
        print(f"   ⚠️  {var_name} = None (no params found)")

# ---- Step 7: Helper function to get parameters ----
def get_best_params(model_name):
    """
    Get best parameters for a model from loaded state.

    Usage:
        params = get_best_params('ctgan')
        params = get_best_params('CTAB-GAN+')
    """
    key = model_name.lower().replace('-', '').replace('+', 'plus').replace(' ', '')
    return LOADED_BEST_PARAMS.get(key, {})

# ---- Step 8: Summary ----
print("\n" + "=" * 80)
print("✅ SESSION RESTORED SUCCESSFULLY!")
print("=" * 80)

print("\n📌 Available variables:")
print("   • data, original_data     - Preprocessed DataFrames")
print("   • TARGET_COLUMN           - Target column name")
print("   • DATASET_IDENTIFIER      - Dataset identifier for paths")
print("   • *_study variables       - Mock study objects with best_trial.params")
print("   • LOADED_BEST_PARAMS      - Dict of all best parameters by model")
print("   • get_best_params(name)   - Helper to get params for a model")

print("\n📌 Example usage for Section 5:")
print('   # Option 1: Use study object (like original code)')
print('   best_params = ctgan_study.best_trial.params')
print('   best_score = ctgan_study.best_trial.value')
print('')
print('   # Option 2: Use helper function')
print('   best_params = get_best_params("ctgan")')
print('')
print('   # Option 3: Direct access')
print('   best_params = LOADED_BEST_PARAMS["ctgan"]')

print("\n📌 Model rankings by score:")
ranked = sorted(scores_by_model.items(), key=lambda x: x[1], reverse=True)
for i, (model, score) in enumerate(ranked, 1):
    print(f"   {i}. {model:15s}: {score:.4f}")

print("\n🚀 Ready to continue with Section 5!")
