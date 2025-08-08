"""
Test CTAB-GAN actual training - reproduce Section 4.2 functionality
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

print("TESTING CTAB-GAN TRAINING (Section 4.2)")
print("=" * 45)

try:
    # Load and prepare data
    data = pd.read_csv('data/breast_cancer_data.csv')
    print(f"Data loaded: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Create CTAB-GAN model
    from src.models.model_factory import ModelFactory
    ctabgan_model = ModelFactory.create('ctabgan', random_state=42)
    print("CTAB-GAN model created")
    
    # Test training with minimal parameters (fast test)
    print("\nStarting training with minimal epochs...")
    
    # Use very small epoch count for testing
    training_metadata = ctabgan_model.train(data, epochs=1)  
    print("Training completed!")
    print(f"Training metadata: {training_metadata}")
    
    # Test generation
    print("\nTesting sample generation...")
    synthetic_data = ctabgan_model.generate(10)
    print(f"Generated synthetic data: {synthetic_data.shape}")
    print("First few synthetic samples:")
    print(synthetic_data.head())
    
    print("\n" + "=" * 45)
    print("SUCCESS: CTAB-GAN training and generation working!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)