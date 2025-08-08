#!/usr/bin/env python3
"""
Quick test to verify CTAB-GAN and CTAB-GAN+ fixes work properly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_ctabgan_imports():
    """Test that CTAB-GAN models can be imported and instantiated."""
    print("Testing CTAB-GAN model import and instantiation...")
    
    try:
        from src.models.implementations.ctabgan_model import CTABGANModel
        model = CTABGANModel(random_state=42)
        print("CTAB-GAN model imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"CTAB-GAN model failed: {e}")
        return False

def test_ctabganplus_imports():
    """Test that CTAB-GAN+ models can be imported and instantiated."""
    print("Testing CTAB-GAN+ model import and instantiation...")
    
    try:
        from src.models.implementations.ctabganplus_model import CTABGANPlusModel
        model = CTABGANPlusModel(random_state=42)
        print("CTAB-GAN+ model imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"CTAB-GAN+ model failed: {e}")
        return False

def test_small_training():
    """Test training on a small synthetic dataset."""
    print("Testing small scale training...")
    
    # Create a small test dataset similar to clinical data
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(20, 80, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'diagnosis': np.random.choice(['A', 'B', 'C'], 100),
        'score': np.random.normal(50, 15, 100),
        'category': np.random.randint(0, 5, 100)
    })
    
    success_count = 0
    
    # Test CTAB-GAN
    try:
        from src.models.implementations.ctabgan_model import CTABGANModel
        model = CTABGANModel(random_state=42)
        model.train(data, epochs=1)  # Very short training for quick test
        print("CTAB-GAN small training successful")
        success_count += 1
    except Exception as e:
        print(f"CTAB-GAN small training failed: {e}")
    
    # Test CTAB-GAN+
    try:
        from src.models.implementations.ctabganplus_model import CTABGANPlusModel
        model = CTABGANPlusModel(random_state=42)
        model.train(data, epochs=1)  # Very short training for quick test
        print("CTAB-GAN+ small training successful")
        success_count += 1
    except Exception as e:
        print(f"CTAB-GAN+ small training failed: {e}")
    
    return success_count == 2

def main():
    """Run all tests."""
    print("Testing CTAB-GAN and CTAB-GAN+ fixes...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_ctabgan_imports():
        tests_passed += 1
    
    if test_ctabganplus_imports():
        tests_passed += 1
    
    if test_small_training():
        tests_passed += 1
    
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("All tests passed! Fixes appear to be working.")
        return True
    else:
        print("Some tests failed. Issues may remain.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)