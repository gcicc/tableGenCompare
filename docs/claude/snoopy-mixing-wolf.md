# Plan: Reorganize January2026.txt into Structured Workflow

## Summary
Split January2026.txt into two files:
1. **Reference doc** - Principles, architecture, detailed specs
2. **Workflow doc** - Numbered tasks optimized for Claude execution

## Files to Create/Modify

| File | Action |
|------|--------|
| `/home/ec2-user/SageMaker/tableGenCompare/docs/January2026-Reference.md` | CREATE - Reference material |
| `/home/ec2-user/SageMaker/tableGenCompare/docs/January2026-Workflow.md` | CREATE - Task workflow |
| `/home/ec2-user/SageMaker/tableGenCompare/docs/January2026.txt` | KEEP - Original preserved |

---

## File 1: January2026-Reference.md

Contains background context Claude should read before executing tasks:

```markdown
# Framework Reference Guide

## Principles
- Backward compatibility: `from setup import *` must work
- Reproducibility: random_state/seeds everywhere
- Results path: results/<dataset>/<YYYY-MM-DD>/Section-<N>/

## Architecture
- setup.py: re-export layer for notebooks
- src/data/: preprocessing
- src/models/: wrappers + factory
- src/objective/: Optuna objectives
- src/evaluation/: quality, TRTS, privacy
- src/visualization/: plots

## Config Schema (for reference)
[Include NOTEBOOK_CONFIG dict from original]

## Model Wrapper Interface
- train(df, **params)
- generate(n) -> pd.DataFrame
- set_config(params)

## Notebook Contract (Section 2 outputs)
- data, original_data, target_column, DATASET_IDENTIFIER, categorical_columns, NOTEBOOK_CONFIG
```

---

## File 2: January2026-Workflow.md

Streamlined task list with clear structure:

```markdown
# Implementation Workflow

## Phase 1: Config System
- [ ] 1.1 Create `src/config.py` with NOTEBOOK_CONFIG schema
- [ ] 1.2 Add `validate_config()` function
- [ ] 1.3 Re-export in setup.py

## Phase 2: Target Integrity (Bug Fixes)
- [ ] 2.1 Create `src/data/target_integrity.py`
  - `enforce_target_schema(real_df, synth_df, target_column, task_type)`
  - `sanitize_numeric(df)` - replace inf with NaN, impute median
- [ ] 2.2 Integrate into model wrappers
- [ ] 2.3 Add defensive call in batch evaluation

## Phase 3: Preprocessing Pipeline
- [ ] 3.1 Create `src/data/preprocess.py`
  - `preprocess_dataset(df, config) -> (df_processed, metadata)`
- [ ] 3.2 Implement missing strategies: none, drop, median/mode, mice
- [ ] 3.3 Implement `indicator_onehot` strategy (new)
- [ ] 3.4 Add `load_and_preprocess_from_config()` helper

## Phase 4: Model Selection
- [ ] 4.1 Create `src/models/registry.py` with AVAILABLE_MODELS dict
- [ ] 4.2 Add `resolve_models(models_to_run)` in model_factory.py
- [ ] 4.3 Add `get_models_to_run(config)` helper
- [ ] 4.4 Wire smoke/full trial selection from config

## Phase 5: New Models
- [ ] 5.1 Create `src/models/implementations/pategan_model.py`
- [ ] 5.2 Create `src/models/implementations/medgan_model.py`
- [ ] 5.3 Register in registry.py and model_factory.py
- [ ] 5.4 Add Optuna objective functions in src/objective/

## Phase 6: Integration
- [ ] 6.1 Create `get_results_path()` utility
- [ ] 6.2 Test end-to-end on breast-cancer dataset

## Done Checklist
- [ ] Existing notebooks run without modification
- [ ] Config block works on one dataset end-to-end
- [ ] Subset model selection works
- [ ] Smoke vs full trials works
- [ ] All missingness strategies run
- [ ] Target integrity prevents failures
- [ ] NaN/inf sanitization works
- [ ] Results saved to standard folder structure
```

---

## Implementation Steps

1. Create `January2026-Reference.md` with principles, architecture, and specs
2. Create `January2026-Workflow.md` with numbered task checklist
3. Keep original `January2026.txt` unchanged as source of truth

## Verification
- Both new files created in `/home/ec2-user/SageMaker/tableGenCompare/docs/`
- Workflow contains all actionable items from original
- Reference contains all background context
- Original file preserved
