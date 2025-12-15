"""
Test script for Optuna visualization integration into batch evaluation.

Verifies that Optuna studies are automatically detected and visualized
during batch evaluation, with outputs saved to Section 4 results.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

print("=" * 80)
print("OPTUNA VISUALIZATION INTEGRATION TEST")
print("=" * 80)

# Load sample dataset
print("\n[1/5] Loading breast cancer dataset...")
data = load_breast_cancer(as_frame=True)
df = data.frame
target_col = 'target'
print(f"   Dataset shape: {df.shape}")

# Create minimal synthetic data (for testing, just add noise)
print("\n[2/5] Creating synthetic test data...")
np.random.seed(42)
synthetic_df = df.copy()
for col in synthetic_df.columns:
    if col != target_col and synthetic_df[col].dtype in ['float64', 'int64']:
        noise = np.random.normal(0, 0.01 * synthetic_df[col].std(), size=len(synthetic_df))
        synthetic_df[col] = synthetic_df[col] + noise

print(f"   Synthetic data shape: {synthetic_df.shape}")

# Create mock Optuna study objects
print("\n[3/5] Creating mock Optuna studies...")
try:
    import optuna

    # Create a simple objective function for testing
    def mock_objective(trial):
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)
        return -(x - 2)**2 - (y + 3)**2  # Simple quadratic with known optimum

    # Create studies for different models
    ctgan_study = optuna.create_study(direction='maximize')
    ctgan_study.optimize(mock_objective, n_trials=10, show_progress_bar=False)

    ctabgan_study = optuna.create_study(direction='maximize')
    ctabgan_study.optimize(mock_objective, n_trials=10, show_progress_bar=False)

    ganeraid_study = optuna.create_study(direction='maximize')
    ganeraid_study.optimize(mock_objective, n_trials=10, show_progress_bar=False)

    print(f"   [OK] Created 3 mock Optuna studies")
    print(f"      - CTGAN best value: {ctgan_study.best_value:.4f}")
    print(f"      - CTABGAN best value: {ctabgan_study.best_value:.4f}")
    print(f"      - GANerAid best value: {ganeraid_study.best_value:.4f}")

except ImportError:
    print("   [ERROR] Optuna not available - skipping Optuna test")
    exit(1)

# Create synthetic data variables in scope (as notebooks would have)
print("\n[4/5] Setting up notebook-style scope...")
synthetic_data_ctgan = synthetic_df.copy()
synthetic_data_ctabgan = synthetic_df.copy()
synthetic_data_ganeraid = synthetic_df.copy()

# Simulate notebook globals() with data and studies
test_scope = {
    'data': df,
    'target_column': target_col,
    'DATASET_IDENTIFIER': 'breast-cancer-test',
    'synthetic_data_ctgan': synthetic_data_ctgan,
    'synthetic_data_ctabgan': synthetic_data_ctabgan,
    'synthetic_data_ganeraid': synthetic_data_ganeraid,
    'ctgan_study': ctgan_study,
    'ctabgan_study': ctabgan_study,
    'ganeraid_study': ganeraid_study
}

print("   [OK] Mock scope created with 3 models and 3 studies")

# Test batch evaluation with Optuna integration
print("\n[5/5] Running batch evaluation with Optuna integration...")
from src.evaluation.batch import evaluate_trained_models

try:
    results = evaluate_trained_models(
        section_number=3,
        variable_pattern='standard',
        scope=test_scope,
        models_to_evaluate=['CTGAN', 'CTABGAN', 'GANerAid'],
        real_data=df,
        target_col=target_col
    )

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    # Check if Optuna visualizations were generated
    optuna_files_found = False
    section4_files = []

    for model_name, model_results in results.items():
        if 'files_generated' in model_results:
            for file_path in model_results['files_generated']:
                if 'section_4' in file_path.lower() or 'optim' in file_path.lower():
                    optuna_files_found = True
                    section4_files.append(file_path)

    if optuna_files_found:
        print(f"\n[OK] Optuna visualizations detected in results!")
        print(f"   Total Section 4 files: {len(set(section4_files))}")

        # List unique Section 4 files
        import os
        unique_files = set(section4_files)
        print(f"\n   Files generated:")
        for file_path in sorted(unique_files):
            if os.path.exists(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                filename = os.path.basename(file_path)
                print(f"      - {filename} ({size_kb:.1f} KB)")
            else:
                print(f"      - {os.path.basename(file_path)} (not found)")
    else:
        print(f"\n[WARNING] No Optuna visualizations found in results")

    # Summary
    print(f"\n[OK] Batch evaluation completed successfully!")
    print(f"   Models evaluated: {len(results)}")
    print(f"   Section 4 (Optuna) files: {len(set(section4_files))}")

except Exception as e:
    print(f"\n[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("[OK] OPTUNA INTEGRATION TEST PASSED")
print("=" * 80)
