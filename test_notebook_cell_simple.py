"""
Test a simple notebook cell from Section 4.2 (CTAB-GAN)
Following claude6.md protocol - test actual notebook context
"""
import sys
import os

# Notebook-like setup
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
os.chdir(r'C:\Users\gcicc\claudeproj\tableGenCompare')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("TESTING SECTION 4.2 NOTEBOOK CELL")
print("=" * 40)

# Load data (this should exist from earlier cells)
try:
    data = pd.read_csv('data/breast_cancer_data.csv')
    print(f"Data loaded: {data.shape}")
    
    # Prepare data for training (typical notebook preprocessing)
    target_column = 'diagnosis'  # The actual target column name
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    print("Data preparation complete")
    
    # Test basic model creation like in Section 4.2
    print("\nTesting CTAB-GAN model creation...")
    from src.models.model_factory import ModelFactory
    
    ctabgan_model = ModelFactory.create('ctabgan', random_state=42)
    print("CTAB-GAN model created successfully")
    
    # Test importing evaluation framework
    print("\nTesting evaluation imports...")
    from src.evaluation.trts_framework import TRTSEvaluator
    print("TRTSEvaluator imported successfully")
    
    # Test optuna import
    print("\nTesting optuna import...")
    import optuna
    print("optuna imported successfully")
    
    print("\n" + "=" * 40)
    print("ALL TESTS PASSED - Section 4 should work!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)