# Framework Reference Guide

Reference documentation for the Clinical Synthetic Data Generation Framework. Read this before executing tasks in January2026-Workflow.md.

---

## Principles

### Backward Compatibility
- Existing notebooks must work with `from setup import *`
- Prefer adding functionality under `src/` and re-exporting via setup.py

### Reproducibility
- All runs must use `random_state`/seeds
- All results saved under: `results/<dataset_identifier>/<YYYY-MM-DD>/Section-<N>/...`

### Clear User Knobs
Users should be able to choose:
- Which models to run (subset)
- Smoke vs full tuning trials
- Full dataset vs 500-row sample
- Missing-data strategy: none, median/mode, MICE, indicator_onehot

---

## Architecture

```
setup.py           # Re-export layer for notebooks (thin)
src/
  config.py        # Session management, global config
  compat.py        # Backward compatibility patches
  utils/           # Paths, documentation, parameter management
  data/            # Preprocessing and summary functions
  models/          # Model wrappers, factory, implementations
  objective/       # Optuna objective functions
  evaluation/      # Quality, TRTS, privacy, batch evaluation
  visualization/   # Section-specific plots
```

---

## Config Schema

```python
NOTEBOOK_CONFIG = {
    # Dataset
    "data_file": None,                 # path to CSV
    "dataset_name": None,              # display only
    "dataset_identifier_override": None,

    # Target
    "target_column": None,
    "categorical_columns": [],         # optional override; else auto-detect
    "task_type": "auto",               # auto | classification | regression

    # Subsetting
    "use_row_subset": True,            # True -> sample_n rows
    "sample_n": 500,
    "sample_random_state": 42,

    # Missingness strategy
    "missing_strategy": "none",        # none | drop | median | mode | mice | indicator_onehot
    "mice_max_iter": 10,

    # Encoding
    "encoding_strategy": "auto",       # auto | onehot | ordinal

    # Models selection
    "models_to_run": "all",            # "all" or list like ["ctgan","ctabganplus"]

    # Tuning
    "tuning_mode": "smoke",            # smoke | full
    "n_trials_smoke": 5,
    "n_trials_full": 50,
    "timeout_seconds": None,           # optional
}
```

---

## Missingness Strategies

| Strategy | Behavior |
|----------|----------|
| `none` | Do nothing |
| `drop` | Drop rows with NA |
| `median` | Fill numeric with median |
| `mode` | Fill categorical with mode |
| `mice` | IterativeImputer for numeric, mode for categorical |
| `indicator_onehot` | Add `<col>__is_missing` indicators, impute median/mode, one-hot encode categoricals |

**Important**: Never one-hot encode the supervised target column.

---

## Model Wrapper Interface

Each model wrapper must implement:

```python
class ModelWrapper:
    def train(self, df, **params):
        """Train on dataframe with hyperparameters."""
        pass

    def generate(self, n) -> pd.DataFrame:
        """Generate n synthetic rows."""
        pass

    def set_config(self, params):
        """Optional: set configuration."""
        pass
```

Requirements:
- Store `random_state`
- Respect `categorical_columns` if relevant

---

## Notebook Contract

After Section 2 runs, these globals must exist for Sections 3-5:

| Variable | Description |
|----------|-------------|
| `data` | Processed DataFrame |
| `original_data` | Copy before modeling |
| `target_column` | Target column name |
| `TARGET_COLUMN` | Same as target_column |
| `DATASET_IDENTIFIER` | Dataset identifier string |
| `categorical_columns` | List of categorical column names |
| `NOTEBOOK_CONFIG` | Configuration dict |

Produce via: `load_and_preprocess_from_config(NOTEBOOK_CONFIG)`

---

## Target Integrity Rules

For `enforce_target_schema(real_df, synth_df, target_column, task_type)`:

| Condition | Action |
|-----------|--------|
| Classification or unique count <= 20 | Cast to numeric, clamp/round to valid classes |
| Binary | Clamp to {0, 1} |
| Multiclass | Map to nearest valid class label |
| Regression | Cast to float, allow continuous |

Always align dtype with real target dtype.

For `sanitize_numeric(df)`:
- Replace inf/-inf with NaN
- Impute NaN with median
- Optionally drop rows with remaining NaN

---

## Results Path Convention

```python
get_results_path(dataset_identifier, section_number, model_name=None)
# Returns: results/<dataset_identifier>/<YYYY-MM-DD>/Section-<N>/[model_name/]
```

Every section/model must write:
- Parameters used (json/csv)
- Generated synthetic sample (optional; at least summary stats)
- Evaluation summaries
