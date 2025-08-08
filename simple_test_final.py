#!/usr/bin/env python3
"""
Simple test without Unicode characters to verify the BayesianGaussianMixture fix.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("Testing BayesianGaussianMixture Fix")
    print("==================================")
    
    # Create test data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.uniform(0, 100, 50),
        'feature2': np.random.uniform(0, 100, 50), 
        'category': np.random.choice(['A', 'B'], 50),
        'target': np.random.choice([0, 1], 50)
    })
    
    print(f"Test data shape: {data.shape}")
    
    success_count = 0
    
    # Test CTAB-GAN
    try:
        from src.models.model_factory import ModelFactory
        model = ModelFactory.create("ctabgan", random_state=42)
        model.train(data, epochs=2)  # Very short training
        synthetic = model.generate(10)
        print(f"CTAB-GAN: Training and generation successful ({synthetic.shape})")
        success_count += 1
    except Exception as e:
        print(f"CTAB-GAN failed: {e}")
    
    # Test CTAB-GAN+
    try:
        model = ModelFactory.create("ctabganplus", random_state=42)
        model.train(data, epochs=2)  # Very short training
        synthetic = model.generate(10)
        print(f"CTAB-GAN+: Training and generation successful ({synthetic.shape})")
        success_count += 1
    except Exception as e:
        print(f"CTAB-GAN+ failed: {e}")
    
    print(f"\nResults: {success_count}/2 tests passed")
    
    if success_count == 2:
        print("SUCCESS: BayesianGaussianMixture compatibility issue is RESOLVED!")
        return True
    else:
        print("FAILURE: Some issues remain")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)