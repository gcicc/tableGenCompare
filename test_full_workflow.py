#!/usr/bin/env python3
"""
Test the complete CTAB-GAN and CTAB-GAN+ workflow to ensure the fixes work.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_full_workflow():
    """Test the complete workflow that would be used in sections 4.2 and 4.3."""
    print("Testing Full CTAB-GAN Workflow (Section 4.2 and 4.3 style)")
    print("=" * 60)
    
    # Create realistic test data similar to what might be in the notebook
    np.random.seed(42)
    n_samples = 100
    
    # Create a dataset similar to clinical data
    data = pd.DataFrame({
        'mean_radius': np.random.uniform(6, 28, n_samples),
        'mean_texture': np.random.uniform(9, 40, n_samples), 
        'mean_perimeter': np.random.uniform(43, 189, n_samples),
        'mean_area': np.random.uniform(143, 2501, n_samples),
        'mean_smoothness': np.random.uniform(0.05, 0.16, n_samples),
        'diagnosis': np.random.choice([0, 1], n_samples)  # Binary classification target
    })
    
    print(f"Test dataset shape: {data.shape}")
    print(f"Dataset info:\n{data.dtypes}")
    print(f"First 3 rows:\n{data.head(3)}")
    
    success_count = 0
    
    # Test 1: CTAB-GAN (Section 4.2)
    print("\n" + "="*30 + " CTAB-GAN Test " + "="*30)
    try:
        from src.models.model_factory import ModelFactory
        
        ctabgan_model = ModelFactory.create("ctabgan", random_state=42)
        print("✓ CTAB-GAN model created successfully")
        
        # Train with limited epochs for testing
        training_result = ctabgan_model.train(data, epochs=5)
        print(f"✓ CTAB-GAN training completed in {training_result['training_time']:.2f}s")
        
        # Generate samples
        synthetic_data = ctabgan_model.generate(50)
        print(f"✓ Generated {len(synthetic_data)} synthetic samples")
        print(f"  Synthetic data shape: {synthetic_data.shape}")
        print(f"  Synthetic data columns: {list(synthetic_data.columns)}")
        
        # Verify data quality
        assert synthetic_data.shape[1] == data.shape[1], "Column count mismatch"
        assert set(synthetic_data.columns) == set(data.columns), "Column names mismatch"
        print("✓ Data integrity checks passed")
        
        success_count += 1
        print("✅ CTAB-GAN test PASSED")
        
    except Exception as e:
        print(f"❌ CTAB-GAN test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: CTAB-GAN+ (Section 4.3)
    print("\n" + "="*30 + " CTAB-GAN+ Test " + "="*30)
    try:
        ctabganplus_model = ModelFactory.create("ctabganplus", random_state=42)
        print("✓ CTAB-GAN+ model created successfully")
        
        # Train with limited epochs for testing
        training_result = ctabganplus_model.train(data, epochs=5)
        print(f"✓ CTAB-GAN+ training completed in {training_result['training_time']:.2f}s")
        
        # Generate samples
        synthetic_data = ctabganplus_model.generate(50)
        print(f"✓ Generated {len(synthetic_data)} synthetic samples")
        print(f"  Synthetic data shape: {synthetic_data.shape}")
        print(f"  Synthetic data columns: {list(synthetic_data.columns)}")
        
        # Verify data quality
        assert synthetic_data.shape[1] == data.shape[1], "Column count mismatch"
        assert set(synthetic_data.columns) == set(data.columns), "Column names mismatch"
        print("✓ Data integrity checks passed")
        
        success_count += 1
        print("✅ CTAB-GAN+ test PASSED")
        
    except Exception as e:
        print(f"❌ CTAB-GAN+ test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print(f"FINAL RESULTS: {success_count}/2 tests passed")
    
    if success_count == 2:
        print("🎉 ALL TESTS PASSED!")
        print("✅ BayesianGaussianMixture scikit-learn compatibility issue RESOLVED")
        print("✅ Sections 4.2 and 4.3 should now work correctly")
        return True
    else:
        print("⚠️  Some tests failed - additional work may be needed")
        return False

if __name__ == "__main__":
    success = test_full_workflow()
    sys.exit(0 if success else 1)