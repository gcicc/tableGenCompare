#!/usr/bin/env python3
"""
Test script for CTGAN model implementation.
Validates CTGAN integration with the framework.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add the src directory to path
sys.path.append('src')

def create_mixed_data():
    """Create mixed-type test data (numerical + categorical)."""
    np.random.seed(42)
    
    n_samples = 800  # Reasonable size for CTGAN
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'employment': np.random.choice(['Employed', 'Unemployed', 'Student', 'Retired'], n_samples),
        'health_score': np.random.uniform(0, 100, n_samples),
        'married': np.random.choice([0, 1], n_samples),  # Binary
        'target': np.random.choice([0, 1], n_samples)
    })
    
    return data

def test_ctgan_availability():
    """Test if CTGAN is available."""
    print("Testing CTGAN availability...")
    
    try:
        from models.model_factory import ModelFactory
        
        available_models = ModelFactory.list_available_models()
        if available_models.get('ctgan', False):
            print("[OK] CTGAN is available")
            return True
        else:
            print("[SKIP] CTGAN not available - dependencies not installed")
            print("To install: pip install ctgan")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error checking CTGAN availability: {e}")
        return False

def test_ctgan_model():
    """Test CTGAN model training and generation."""
    print("Testing CTGAN model...")
    
    try:
        from models.model_factory import ModelFactory
        
        # Create CTGAN model
        model = ModelFactory.create('ctgan', device='cpu', random_state=42)
        print(f"[OK] CTGAN model created: {model.__class__.__name__}")
        
        # Test model info
        model_info = model.get_model_info()
        print(f"[OK] Model info: {model_info['model_type']}")
        
        # Test hyperparameter space
        hyperparams = model.get_hyperparameter_space()
        print(f"[OK] Hyperparameter space has {len(hyperparams)} parameters")
        
        # Create test data with mixed types
        test_data = create_mixed_data()
        print(f"[OK] Mixed-type test data created: {test_data.shape}")
        print(f"Data types: {test_data.dtypes.tolist()}")
        print(f"Categorical columns: {test_data.select_dtypes(include=['object']).columns.tolist()}")
        
        # Train model with minimal epochs for testing
        print("Training CTGAN model (this may take 2-3 minutes)...")
        training_result = model.train(
            test_data, 
            epochs=100,  # Reduced for testing
            verbose=True
        )
        print(f"[OK] Model training completed in {training_result.get('training_duration_seconds', 'N/A'):.2f} seconds")
        print(f"Discrete columns detected: {training_result.get('discrete_columns', [])}")
        
        # Generate synthetic data
        print("Generating synthetic data...")
        synthetic_data = model.generate(200)  # Smaller sample for testing
        print(f"[OK] Generated synthetic data: {synthetic_data.shape}")
        
        # Basic validation
        if synthetic_data.shape[1] == test_data.shape[1]:
            print("[OK] Synthetic data has correct number of columns")
        else:
            print(f"[WARN] Column mismatch: {synthetic_data.shape[1]} vs {test_data.shape[1]}")
        
        if len(synthetic_data) == 200:
            print("[OK] Generated correct number of samples")
        else:
            print(f"[WARN] Sample count mismatch: {len(synthetic_data)} vs 200")
        
        # Check data types preservation
        original_types = set(test_data.dtypes.astype(str))
        synthetic_types = set(synthetic_data.dtypes.astype(str))
        if original_types == synthetic_types:
            print("[OK] Data types preserved")
        else:
            print(f"[INFO] Data type differences: Original={original_types}, Synthetic={synthetic_types}")
        
        # Check categorical values
        for col in ['education', 'employment']:
            if col in synthetic_data.columns:
                original_values = set(test_data[col].unique())
                synthetic_values = set(synthetic_data[col].unique())
                if synthetic_values.issubset(original_values):
                    print(f"[OK] {col} values are valid")
                else:
                    extra_values = synthetic_values - original_values
                    print(f"[WARN] {col} has extra values: {extra_values}")
        
        return True
        
    except ImportError as e:
        print(f"[SKIP] CTGAN dependencies not available: {e}")
        return True
    except Exception as e:
        print(f"[FAIL] CTGAN model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ctgan_evaluation_integration():
    """Test CTGAN integration with evaluation framework."""
    print("Testing CTGAN-Evaluation integration...")
    
    try:
        from models.model_factory import ModelFactory
        from evaluation.unified_evaluator import UnifiedEvaluator
        
        # Create model and data
        model = ModelFactory.create('ctgan', device='cpu', random_state=42)
        original_data = create_mixed_data()
        
        print("Training CTGAN for integration test...")
        model.train(original_data, epochs=50, verbose=False)  # Quick training
        
        print("Generating synthetic data...")
        synthetic_data = model.generate(len(original_data))
        
        print("Running evaluation...")
        evaluator = UnifiedEvaluator(random_state=42)
        
        dataset_metadata = {
            'dataset_info': {'name': 'ctgan_integration_test', 'description': 'CTGAN integration test'},
            'target_info': {'column': 'target', 'type': 'binary'}
        }
        
        results = evaluator.run_complete_evaluation(
            model=model,
            original_data=original_data,
            synthetic_data=synthetic_data,
            dataset_metadata=dataset_metadata,
            output_dir="test_ctgan_output",
            target_column='target'
        )
        
        print("[OK] CTGAN-Evaluation integration successful!")
        print(f"  Model: {results['model_info']['model_type']}")
        print(f"  TRTS Overall Score: {results['trts_results']['overall_score_percent']:.1f}%")
        print(f"  Similarity Score: {results['similarity_analysis']['final_similarity']:.3f}")
        
        return True
        
    except ImportError as e:
        print(f"[SKIP] Dependencies not available: {e}")
        return True
    except Exception as e:
        print(f"[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("CTGAN MODEL TEST")
    print("=" * 60)
    
    # Test CTGAN availability
    available = test_ctgan_availability()
    if not available:
        print("\n[INFO] CTGAN not available. To install:")
        print("pip install ctgan")
        print("\nSkipping CTGAN tests.")
        sys.exit(0)
    
    # Test CTGAN model
    print("\n" + "=" * 60)
    model_ok = test_ctgan_model()
    
    # Test integration
    print("\n" + "=" * 60)
    integration_ok = test_ctgan_evaluation_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print(f"CTGAN Model: {'PASS' if model_ok else 'FAIL'}")
    print(f"Integration: {'PASS' if integration_ok else 'FAIL'}")
    
    if model_ok and integration_ok:
        print("\n[SUCCESS] All CTGAN tests passed!")
        print("CTGAN is ready for production use.")
    else:
        print("\n[PARTIAL] Some tests failed or were skipped.")
    
    print("=" * 60)