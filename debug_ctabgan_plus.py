#!/usr/bin/env python3
"""
Debug CTAB-GAN+ specific issues.
"""

import sys
import os
import pandas as pd
import numpy as np

print("=== CTAB-GAN+ Debug ===")

# Clear module cache completely
modules_to_clear = [mod for mod in sys.modules if 'model' in mod]
for mod in modules_to_clear:
    del sys.modules[mod]

# Add CTAB-GAN+ path
ctabganplus_path = os.path.join(os.path.dirname(__file__), "CTAB-GAN-Plus")
if ctabganplus_path not in sys.path:
    sys.path.insert(0, ctabganplus_path)

print(f"Added path: {ctabganplus_path}")
print(f"First few paths in sys.path: {sys.path[:3]}")

try:
    from model.ctabgan import CTABGAN
    from model.pipeline.data_preparation import DataPrep
    
    import inspect
    
    # Check CTABGAN signature
    ctabgan_sig = inspect.signature(CTABGAN.__init__)
    print(f"CTABGAN signature: {ctabgan_sig}")
    print(f"CTABGAN parameters: {list(ctabgan_sig.parameters.keys())}")
    
    # Check DataPrep signature
    dataprep_sig = inspect.signature(DataPrep.__init__)
    print(f"DataPrep signature: {dataprep_sig}")
    print(f"DataPrep parameters: {list(dataprep_sig.parameters.keys())}")
    
    # Test small instantiation
    print("\nTesting CTABGAN instantiation...")
    
    # Create test data
    data = pd.DataFrame({
        'age': np.random.randint(20, 80, 20),
        'gender': np.random.choice(['M', 'F'], 20),
        'score': np.random.normal(50, 15, 20),
    })
    temp_path = "temp_debug.csv"
    data.to_csv(temp_path, index=False)
    
    # Test with minimal parameters
    model = CTABGAN(
        raw_csv_path=temp_path,
        test_ratio=0.2,
        categorical_columns=['gender'],
        log_columns=[],
        mixed_columns={},
        general_columns=['score'],
        non_categorical_columns=['score'],
        integer_columns=['age'],
        problem_type={"Classification": "gender"}
    )
    
    print("CTABGAN+ instantiated successfully!")
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()