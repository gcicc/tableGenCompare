# Implementation Workflow

Task checklist for January 2026 framework improvements.
See [January2026-Reference.md](January2026-Reference.md) for detailed specs.

---

## Phase 1: Config System

- [x] **1.1** Create `src/config.py` with NOTEBOOK_CONFIG schema
  - Default values for all config keys
  - Type hints for IDE support

- [x] **1.2** Add `validate_config(config)` function
  - Check required fields (data_file, target_column)
  - Validate enum values (task_type, missing_strategy, tuning_mode)
  - Return validated config with defaults filled

- [x] **1.3** Re-export in setup.py
  - `from src.config import NOTEBOOK_CONFIG_DEFAULTS, get_default_config, validate_config, get_n_trials`

---

## Phase 2: Target Integrity (Bug Fixes)

**Context**: CTABGAN produces continuous targets (breaks classification). TVAE produces NaN/inf (breaks correlation).

- [x] **2.1** Create `src/data/target_integrity.py`
  - `enforce_target_schema(real_df, synth_df, target_column, task_type) -> synth_df_fixed`
    - Classification: clamp/round to valid class labels
    - Binary: ensure {0, 1}
    - Regression: cast to float
  - `sanitize_numeric(df) -> df`
    - Replace inf/-inf with NaN
    - Impute NaN in numeric columns with median
  - `sanitize_synthetic_data()` - convenience function combining both

- [x] **2.2** Integrate into model wrappers
  - Added `sanitize_output()` method to base_model.py
  - Added `sanitize_numeric()` call in TVAE `_postprocess_data()`
  - Added `sanitize_numeric()` call in CTABGAN `generate()`

- [x] **2.3** Add defensive call in batch evaluation
  - Added `sanitize_synthetic_data()` before `evaluate_synthetic_data_quality()`
  - Added `sanitize_synthetic_data()` before `comprehensive_trts_analysis()`

---

## Phase 3: Preprocessing Pipeline

- [x] **3.1** Create `src/data/preprocessing.py` (extended existing file)
  - `preprocess_dataset(df, config) -> (df_processed, metadata)`
  - Metadata includes: dataset_identifier, target_column, categorical_columns, task_type, transform_log
  - Auto-detects categorical columns and task type
  - Standardizes column names

- [x] **3.2** Implement core missing strategies
  - `none`: pass through
  - `drop`: drop rows with NA
  - `median`: fill numeric with median
  - `mode`: fill categorical with mode
  - `mice`: IterativeImputer for numeric, mode for categorical

- [x] **3.3** Implement `indicator_onehot` strategy (new)
  - Add `<col>__is_missing` as 0/1 for columns with any NA
  - Impute remaining NA (median for numeric, mode for categorical)
  - One-hot encode categorical columns (NOT the target)

- [x] **3.4** Add `load_and_preprocess_from_config(config)` helper
  - Returns: data, original_data, target_column, DATASET_IDENTIFIER, categorical_columns, metadata
  - Re-exported via setup.py

---

## Phase 4: Model Selection

- [x] **4.1** Create `src/models/registry.py`
  - AVAILABLE_MODELS dict with 6 models: ctgan, tvae, ctabgan, ctabganplus, copulagan, ganeraid
  - MODEL_ALIASES for flexible naming (ctab-gan, ctabgan+, etc.)
  - `get_available_model_names()`, `is_model_available()`, `get_model_display_name()`

- [x] **4.2** Add `resolve_models(models_to_run)` in registry.py
  - `"all"` returns all available models
  - List returns intersection, warns on unknown/unavailable names
  - Normalizes aliases automatically

- [x] **4.3** Add `get_models_to_run(config)` helper
  - Main entry point for config-driven model selection
  - Re-exported via setup.py

- [x] **4.4** Wire smoke/full trial selection
  - `get_tuning_config(config)` returns {n_trials, timeout_seconds, tuning_mode}
  - `get_n_trials(config)` returns appropriate trial count
  - Re-exported via setup.py as `get_tuning_config`, `get_n_trials_from_registry`

---

## Phase 5: New Models

- [x] **5.1** Create `src/models/implementations/pategan_model.py`
  - Implement train/generate interface
  - Log privacy budget/accountant parameters
  - Tuning params: epochs, batch_size, learning rates, privacy budget

- [x] **5.2** Create `src/models/implementations/medgan_model.py`
  - Implement train/generate interface
  - Handle discrete/encoded input requirements
  - Tuning params: epochs, batch_size, architecture dims

- [x] **5.3** Register in registry.py and model_factory.py
  - Add to AVAILABLE_MODELS dict
  - Wire up in factory pattern

- [x] **5.4** Add Optuna objective functions in `src/objective/`
  - Define search spaces for PATE-GAN and MEDGAN
  - Integrate with existing objective infrastructure

---

## Phase 6: Integration

- [x] **6.1** Create `get_results_path()` utility
  - `get_results_path(dataset_identifier, section_number, model_name=None)`
  - Returns: `results/<dataset_identifier>/<YYYY-MM-DD>/Section-<N>/[model_name/]`
  - Re-export via setup.py

- [ ] **6.2** Test end-to-end on breast-cancer dataset
  - Run all 5 notebook sections
  - Verify config block works
  - Verify model subset selection
  - Verify smoke/full trials
  - Verify results saved correctly

---

## Done Checklist

- [x] Existing notebooks run without modification (`from setup import *` works)
- [ ] New config block works on at least one dataset notebook end-to-end
- [x] Subset model selection works (e.g., only CTGAN + PATE-GAN)
- [x] Smoke vs full trials works
- [ ] All missingness strategies run (including indicator_onehot)
- [ ] Target integrity prevents CTABGAN continuous-target failures
- [ ] NaN/inf sanitization prevents correlation failures
- [ ] Results saved under standard folder structure
