# Task 4.1: Optuna Visualization Integration Guide

## Overview

This document provides instructions for adding standardized Optuna visualizations to all 7 notebooks in Section 4.

## Implementation Status

✅ **Core Infrastructure Complete:**
- Created `src/visualization/section4.py` with helper functions
- Added exports to `setup.py` for notebook access
- Functions available: `create_optuna_visualizations()` and `create_all_models_optuna_summary()`

## Notebooks to Update

1. `SynthethicTableGenerator-Alzheimer.ipynb`
2. `SynthethicTableGenerator-BreastCancer.ipynb`
3. `SynthethicTableGenerator-Liver.ipynb`
4. `SynthethicTableGenerator-Pakistani.ipynb`
5. `STG-BreastCancerV2.ipynb`
6. `STG-LiverV2.ipynb`
7. `STG-PakistaniV2.ipynb`

## Code to Add

### After Each Model's `study.optimize()` Call

Add this code block immediately after each of the 6 models' optimization:

```python
# Generate Optuna visualizations
from src.visualization.section4 import create_optuna_visualizations

create_optuna_visualizations(
    study=<STUDY_NAME>,  # Replace with actual study variable
    model_name='<MODEL_NAME>',  # Replace with model name
    results_path=results_path,
    verbose=True
)
```

### Model-Specific Examples

#### 1. CTGAN (after `ctgan_study.optimize()`)
```python
create_optuna_visualizations(
    study=ctgan_study,
    model_name='CTGAN',
    results_path=results_path,
    verbose=True
)
```

#### 2. CTABGAN (after `ctabgan_study.optimize()`)
```python
create_optuna_visualizations(
    study=ctabgan_study,
    model_name='CTABGAN',
    results_path=results_path,
    verbose=True
)
```

#### 3. CTABGAN+ (after `ctabganplus_study.optimize()`)
```python
create_optuna_visualizations(
    study=ctabganplus_study,
    model_name='CTABGANPLUS',
    results_path=results_path,
    verbose=True
)
```

#### 4. GANerAid (after `ganeraid_study.optimize()`)
```python
create_optuna_visualizations(
    study=ganeraid_study,
    model_name='GANERAID',
    results_path=results_path,
    verbose=True
)
```

#### 5. CopulaGAN (after `copulagan_study.optimize()`)
```python
create_optuna_visualizations(
    study=copulagan_study,
    model_name='COPULAGAN',
    results_path=results_path,
    verbose=True
)
```

#### 6. TVAE (after `tvae_study.optimize()`)
```python
create_optuna_visualizations(
    study=tvae_study,
    model_name='TVAE',
    results_path=results_path,
    verbose=True
)
```

### Optional: Add Summary Visualization at End of Section 4

After all 6 models have been optimized, add this code to create a comparative summary:

```python
# Create summary visualization comparing all models
from src.visualization.section4 import create_all_models_optuna_summary

studies_dict = {
    'CTGAN': ctgan_study,
    'CTABGAN': ctabgan_study,
    'CTABGAN+': ctabganplus_study,
    'GANerAid': ganeraid_study,
    'CopulaGAN': copulagan_study,
    'TVAE': tvae_study
}

create_all_models_optuna_summary(
    studies_dict=studies_dict,
    results_path=results_path,
    verbose=True
)
```

## Expected Outputs Per Model

Each model will generate 3 visualizations in the Section-4 results folder:

1. **optim_history_{MODEL_NAME}.png**: Shows objective value progression over trials
2. **param_importance_{MODEL_NAME}.png**: Shows which hyperparameters matter most
3. **parallel_coord_{MODEL_NAME}.png**: Shows parameter interactions (top 5 params)

## Summary Output (Optional)

If the summary function is added:
- **optuna_summary_all_models.png**: Comparative view of best performance and optimization effort across all models

## Integration Steps

For each of the 7 notebooks:

1. Open notebook in Jupyter/VS Code
2. Find Section 4 (Hyperparameter Optimization)
3. Locate each of the 6 `study.optimize()` calls
4. Add the corresponding visualization code immediately after each optimize() call
5. (Optional) Add summary visualization at the end of Section 4
6. Save notebook
7. Test by running Section 4

## Dependencies

- **Required**: `optuna`, `matplotlib`, `numpy`
- **For image export**: `kaleido` package (install with `pip install kaleido`)
  - If kaleido not available, the function will gracefully skip visualization

## Error Handling

The `create_optuna_visualizations()` function includes robust error handling:
- Checks for Optuna visualization availability
- Catches individual plot failures (continues if one plot fails)
- Provides clear warning messages
- Returns dictionary of successfully generated files

## Testing

After adding to each notebook:

1. Run a single cell with the visualization code to test
2. Check that 3 PNG files are created in results folder
3. Open PNG files to verify they display correctly
4. Verify no errors in notebook output

## Success Criteria

- [ ] All 6 model optimizations in each notebook have visualization calls
- [ ] All 7 notebooks updated
- [ ] Visualizations generate without errors
- [ ] PNG files saved to Section-4 results folders
- [ ] Consistent implementation across all notebooks

## Notes

- The visualization code is wrapped in try-except, so it won't break optimization if visualization fails
- Image export requires `kaleido` package - if not installed, function will report warning but not crash
- File paths use the existing `results_path` variable from each notebook
- Model names should match the naming convention used elsewhere in the project (uppercase)

---
**Created**: 2025-12-06
**Status**: Infrastructure complete, notebook integration pending
**Related Tasks**: Task 3.8 (Optuna pruning), Task 4.2 (Documentation updates)
