#!/usr/bin/env python3
"""
Test the fixes for CTAB-GAN and CTAB-GAN+ sections 4.2 and 4.3.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_ctabgan_fix():
    """Test CTAB-GAN fix for rint method error."""
    print("Testing CTAB-GAN fix...")
    
    try:
        from src.models.implementations.ctabgan_model import CTABGANModel
        
        # Create test data that might trigger the rint error
        # Mix of categorical and numeric data
        np.random.seed(42)
        data = pd.DataFrame({
            'age': np.random.randint(20, 80, 50),
            'gender': np.random.choice(['M', 'F'], 50),  # Categorical
            'diagnosis': np.random.choice([0, 1], 50),   # Categorical but numeric
            'score': np.random.normal(50, 15, 50),       # Continuous
            'category': np.random.randint(0, 3, 50)      # Could be categorical or integer
        })
        
        model = CTABGANModel(random_state=42)
        
        print(f"  Test data shape: {data.shape}")
        print(f"  Data types: {data.dtypes.to_dict()}")
        
        # Test training
        model.train(data, epochs=1)  # Very short training
        print("  Training completed")
        
        # Test generation (this is where the rint error occurred)
        synthetic_data = model.generate(10)
        print(f"  Generation completed: {synthetic_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"  CTAB-GAN test failed: {e}")
        return False

def test_ctabganplus_fix():
    """Test CTAB-GAN+ fix for general_columns parameter error."""
    print("Testing CTAB-GAN+ fix...")
    
    try:
        from src.models.implementations.ctabganplus_model import CTABGANPlusModel
        
        # Create test data
        np.random.seed(42)
        data = pd.DataFrame({
            'age': np.random.randint(20, 80, 50),
            'gender': np.random.choice(['M', 'F'], 50),
            'score': np.random.normal(50, 15, 50),
            'category': np.random.randint(0, 2, 50)
        })
        
        model = CTABGANPlusModel(random_state=42)
        
        print(f"  Test data shape: {data.shape}")
        
        # Test training (this is where the general_columns error occurred)
        model.train(data, epochs=1)  # Very short training
        print("  Training completed")
        
        # Test generation
        synthetic_data = model.generate(10)
        print(f"  Generation completed: {synthetic_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"  CTAB-GAN+ test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Section 4.2 and 4.3 Fixes")
    print("=" * 50)
    
    success_count = 0
    total_tests = 2
    
    if test_ctabgan_fix():
        success_count += 1
    
    if test_ctabganplus_fix():
        success_count += 1
    
    print("=" * 50)
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("All fixes are working correctly!")
    else:
        print("Some fixes may need additional work.")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)