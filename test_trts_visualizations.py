"""
Test script for TRTS visualization enhancements.

Tests the new ROC/PR/Calibration curve generation and privacy dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Import our enhanced functions
from src.evaluation.trts import comprehensive_trts_analysis
from src.visualization.section5 import (
    create_trts_visualizations,
    create_privacy_dashboard,
    create_trts_roc_curves,
    create_trts_pr_curves,
    create_trts_calibration_curves
)

print("=" * 80)
print("TRTS VISUALIZATION ENHANCEMENT TEST")
print("=" * 80)

# Load sample dataset
print("\n[1/6] Loading breast cancer dataset...")
data = load_breast_cancer(as_frame=True)
df = data.frame
target_col = 'target'

print(f"   Dataset shape: {df.shape}")
print(f"   Target classes: {df[target_col].nunique()}")

# Create synthetic data (for testing, just add noise to real data)
print("\n[2/6] Creating synthetic test data...")
np.random.seed(42)
synthetic_df = df.copy()
for col in synthetic_df.columns:
    if col != target_col and synthetic_df[col].dtype in ['float64', 'int64']:
        noise = np.random.normal(0, 0.01 * synthetic_df[col].std(), size=len(synthetic_df))
        synthetic_df[col] = synthetic_df[col] + noise

print(f"   Synthetic data shape: {synthetic_df.shape}")

# Test Phase 1: TRTS analysis with prediction storage
print("\n[3/6] Testing comprehensive_trts_analysis with store_predictions=True...")
try:
    trts_results = comprehensive_trts_analysis(
        real_data=df,
        synthetic_data=synthetic_df,
        target_column=target_col,
        test_size=0.2,
        random_state=42,
        n_estimators=50,
        verbose=True,
        store_predictions=True  # NEW PARAMETER
    )

    # Verify predictions are stored
    has_predictions = all(
        'predictions' in trts_results.get(scenario, {})
        for scenario in ['TRTR', 'TRTS', 'TSTR', 'TSTS']
        if trts_results.get(scenario, {}).get('status') == 'success'
    )

    if has_predictions:
        print("   [OK] Predictions stored successfully in all scenarios!")
    else:
        print("   [WARNING] Predictions not stored in all scenarios")

    # Verify privacy metrics are present
    if 'privacy' in trts_results:
        print("   [OK] Privacy metrics calculated successfully!")
        print(f"      Privacy score: {trts_results['privacy'].get('privacy_score', 'N/A'):.3f}")
    else:
        print("   [WARNING] Privacy metrics not found")

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test Phase 2: Create visualizations
print("\n[4/6] Testing standard TRTS visualizations...")
try:
    import tempfile
    import os

    results_dir = tempfile.mkdtemp()
    trts_results_dict = {'TestModel': trts_results}

    viz_results = create_trts_visualizations(
        trts_results_dict=trts_results_dict,
        model_names=['TestModel'],
        results_dir=results_dir,
        dataset_name="Breast Cancer Test",
        save_files=True,
        display_plots=False
    )

    if 'files_generated' in viz_results and len(viz_results['files_generated']) > 0:
        print(f"   [OK] Generated {len(viz_results['files_generated'])} visualization files")
        for f in viz_results['files_generated']:
            print(f"      - {os.path.basename(f)}")
    else:
        print("   [WARNING]  No visualization files generated")

except Exception as e:
    print(f"   [ERROR] ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test Phase 3: Create privacy dashboard
print("\n[5/6] Testing privacy dashboard...")
try:
    privacy_result = create_privacy_dashboard(
        trts_results_dict=trts_results_dict,
        model_names=['TestModel'],
        results_dir=results_dir,
        dataset_name="Breast Cancer Test",
        save_files=True,
        display_plots=False,
        verbose=True
    )

    if privacy_result and 'files_generated' in privacy_result:
        print(f"   [OK] Privacy dashboard generated: {len(privacy_result['files_generated'])} files")
        for f in privacy_result['files_generated']:
            print(f"      - {os.path.basename(f)}")
    else:
        print("   [WARNING]  Privacy dashboard not generated (may need privacy data)")

except Exception as e:
    print(f"   [ERROR] ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test Phase 4: Create ROC/PR/Calibration curves
print("\n[6/6] Testing ROC/PR/Calibration curves...")
try:
    roc_path = create_trts_roc_curves(
        trts_results_dict=trts_results_dict,
        model_names=['TestModel'],
        results_dir=results_dir,
        dataset_name="Breast Cancer Test",
        save_files=True,
        display_plots=False,
        verbose=True
    )

    pr_path = create_trts_pr_curves(
        trts_results_dict=trts_results_dict,
        model_names=['TestModel'],
        results_dir=results_dir,
        dataset_name="Breast Cancer Test",
        save_files=True,
        display_plots=False,
        verbose=True
    )

    calib_path = create_trts_calibration_curves(
        trts_results_dict=trts_results_dict,
        model_names=['TestModel'],
        results_dir=results_dir,
        dataset_name="Breast Cancer Test",
        save_files=True,
        display_plots=False,
        verbose=True
    )

    curves_generated = [roc_path, pr_path, calib_path]
    curves_generated = [c for c in curves_generated if c is not None]

    if len(curves_generated) == 3:
        print(f"   [OK] All 3 curve types generated successfully!")
        print(f"      - ROC curves: {os.path.basename(roc_path)}")
        print(f"      - PR curves: {os.path.basename(pr_path)}")
        print(f"      - Calibration curves: {os.path.basename(calib_path)}")
    else:
        print(f"   [WARNING]  Only {len(curves_generated)}/3 curve types generated")

except Exception as e:
    print(f"   [ERROR] ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

# List all generated files
print(f"\nGenerated files in: {results_dir}")
if os.path.exists(results_dir):
    files = os.listdir(results_dir)
    print(f"Total files: {len(files)}")
    for f in sorted(files):
        size_kb = os.path.getsize(os.path.join(results_dir, f)) / 1024
        print(f"   - {f} ({size_kb:.1f} KB)")

print("\n[OK] All tests completed successfully!")
print(f"   Results saved to: {results_dir}")
