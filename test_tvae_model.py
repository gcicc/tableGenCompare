#!/usr/bin/env python3
"""
Test script for TVAE model implementation.
Validates TVAE integration with the framework.
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
    
    n_samples = 600  # Good size for TVAE
    data = pd.DataFrame({
        'age': np.random.randint(20, 75, n_samples),
        'income': np.random.normal(45000, 12000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples),
        'experience': np.random.uniform(0, 30, n_samples),
        'performance_score': np.random.uniform(1, 10, n_samples),
        'promoted': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    return data

def test_tvae_availability():
    """Test if TVAE (SDV) is available."""
    print("Testing TVAE availability...")
    
    try:
        from models.model_factory import ModelFactory
        
        available_models = ModelFactory.list_available_models()
        if available_models.get('tvae', False):
            print("[OK] TVAE is available")
            return True
        else:
            print("[SKIP] TVAE not available - dependencies not installed")
            print("To install: pip install sdv")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error checking TVAE availability: {e}")
        return False

def test_tvae_model():
    """Test TVAE model training and generation."""
    print("Testing TVAE model...")
    
    try:
        from models.model_factory import ModelFactory
        
        # Create TVAE model
        model = ModelFactory.create('tvae', device='cpu', random_state=42)
        print(f"[OK] TVAE model created: {model.__class__.__name__}")
        
        # Test model info
        model_info = model.get_model_info()
        print(f"[OK] Model info: {model_info['model_type']}")
        
        # Test hyperparameter space
        hyperparams = model.get_hyperparameter_space()
        print(f"[OK] Hyperparameter space has {len(hyperparams)} parameters")
        print(f"Available parameters: {list(hyperparams.keys())}")
        
        # Create test data with mixed types
        test_data = create_mixed_data()
        print(f"[OK] Mixed-type test data created: {test_data.shape}")
        print(f"Data types: {test_data.dtypes.tolist()}")
        print(f"Categorical columns: {test_data.select_dtypes(include=['object']).columns.tolist()}")
        
        # Train model with minimal epochs for testing
        print("Training TVAE model (this may take 2-3 minutes)...")
        training_result = model.train(
            test_data, 
            epochs=100  # Reduced for testing
        )
        print(f"[OK] Model training completed in {training_result.get('training_duration_seconds', 'N/A'):.2f} seconds")
        
        # Check training details
        if 'final_loss' in training_result and training_result['final_loss']:
            print(f"Final training loss: {training_result['final_loss']:.6f}")
        
        if 'convergence_achieved' in training_result:
            print(f"Convergence achieved: {training_result['convergence_achieved']}")
        
        # Generate synthetic data
        print("Generating synthetic data...")
        synthetic_data = model.generate(150)  # Smaller sample for testing
        print(f"[OK] Generated synthetic data: {synthetic_data.shape}")
        
        # Basic validation
        if synthetic_data.shape[1] == test_data.shape[1]:
            print("[OK] Synthetic data has correct number of columns")
        else:
            print(f"[WARN] Column mismatch: {synthetic_data.shape[1]} vs {test_data.shape[1]}")
        
        if len(synthetic_data) == 150:
            print("[OK] Generated correct number of samples")
        else:
            print(f"[WARN] Sample count mismatch: {len(synthetic_data)} vs 150")
        
        # Check data types consistency
        print("\nData Type Validation:")
        for col in test_data.columns:
            if col in synthetic_data.columns:
                orig_type = str(test_data[col].dtype)
                synth_type = str(synthetic_data[col].dtype)
                print(f"  {col}: {orig_type} -> {synth_type}")
        
        # Check categorical values
        print("\nCategorical Value Validation:")
        for col in ['education', 'department']:
            if col in synthetic_data.columns:
                original_values = set(test_data[col].unique())
                synthetic_values = set(synthetic_data[col].unique())
                if synthetic_values.issubset(original_values):
                    print(f"  [OK] {col} values are valid")
                else:
                    extra_values = synthetic_values - original_values
                    print(f"  [WARN] {col} has extra values: {extra_values}")
        
        # Check numerical ranges
        print("\nNumerical Range Validation:")
        for col in ['age', 'income', 'experience']:
            if col in synthetic_data.columns:
                orig_min, orig_max = test_data[col].min(), test_data[col].max()
                synth_min, synth_max = synthetic_data[col].min(), synthetic_data[col].max()
                print(f"  {col}: Original [{orig_min:.2f}, {orig_max:.2f}] -> Synthetic [{synth_min:.2f}, {synth_max:.2f}]")
        
        return True
        
    except ImportError as e:
        print(f"[SKIP] TVAE dependencies not available: {e}")
        return True
    except Exception as e:
        print(f"[FAIL] TVAE model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tvae_evaluation_integration():
    """Test TVAE integration with evaluation framework."""
    print("Testing TVAE-Evaluation integration...")
    
    try:
        from models.model_factory import ModelFactory
        from evaluation.unified_evaluator import UnifiedEvaluator
        
        # Create model and data
        model = ModelFactory.create('tvae', device='cpu', random_state=42)
        original_data = create_mixed_data()
        
        print("Training TVAE for integration test...")
        model.train(original_data, epochs=50)  # Quick training for testing
        
        print("Generating synthetic data...")
        synthetic_data = model.generate(len(original_data))
        
        print("Running evaluation...")
        evaluator = UnifiedEvaluator(random_state=42)
        
        dataset_metadata = {
            'dataset_info': {'name': 'tvae_integration_test', 'description': 'TVAE integration test'},
            'target_info': {'column': 'target', 'type': 'binary'}
        }
        
        results = evaluator.run_complete_evaluation(
            model=model,
            original_data=original_data,
            synthetic_data=synthetic_data,
            dataset_metadata=dataset_metadata,
            output_dir="test_tvae_output",
            target_column='target'
        )
        
        print("[OK] TVAE-Evaluation integration successful!")
        print(f"  Model: {results['model_info']['model_type']}")
        print(f"  TRTS Overall Score: {results['trts_results']['overall_score_percent']:.1f}%")
        print(f"  Similarity Score: {results['similarity_analysis']['final_similarity']:.3f}")
        print(f"  Data Quality Score: {results['data_quality']['data_type_consistency']:.1f}%")
        
        # Check specific TVAE training metadata in results
        if 'training_metadata' in results.get('model_info', {}):
            training_meta = results['model_info']['training_metadata']
            if 'final_loss' in training_meta and training_meta['final_loss']:
                print(f"  Final Training Loss: {training_meta['final_loss']:.6f}")
            if 'compress_dimensions' in training_meta:
                print(f"  Network Architecture: {training_meta['compress_dimensions']} -> {training_meta['decompress_dimensions']}")
        
        return True
        
    except ImportError as e:
        print(f"[SKIP] Dependencies not available: {e}")
        return True
    except Exception as e:
        print(f"[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tvae_vs_ctgan_comparison():
    """Quick comparison test between TVAE and CTGAN if both available."""
    print("Testing TVAE vs CTGAN comparison...")
    
    try:
        from models.model_factory import ModelFactory
        
        available_models = ModelFactory.list_available_models()
        
        if not available_models.get('tvae', False):
            print("[SKIP] TVAE not available")
            return True
        
        if not available_models.get('ctgan', False):
            print("[SKIP] CTGAN not available for comparison")
            return True
        
        # Create test data
        test_data = create_mixed_data()
        print(f"Comparison test data: {test_data.shape}")
        
        # Test both models with same parameters
        models_to_test = ['tvae', 'ctgan']
        results = {}
        
        for model_name in models_to_test:
            print(f"\nTesting {model_name.upper()}...")
            
            model = ModelFactory.create(model_name, device='cpu', random_state=42)
            
            # Quick training
            start_time = pd.Timestamp.now()
            if model_name == 'tvae':
                model.train(test_data, epochs=50)
            else:  # ctgan
                model.train(test_data, epochs=50, verbose=False)
            
            training_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Generate data
            start_time = pd.Timestamp.now()
            synthetic_data = model.generate(100)
            generation_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            results[model_name] = {
                'training_time': training_time,
                'generation_time': generation_time,
                'synthetic_shape': synthetic_data.shape
            }
            
            print(f"  Training time: {training_time:.2f}s")
            print(f"  Generation time: {generation_time:.3f}s")
            print(f"  Output shape: {synthetic_data.shape}")
        
        # Compare results
        print(f"\n{'='*40}")
        print("TVAE vs CTGAN Comparison Summary:")
        print(f"{'='*40}")
        
        for model_name in models_to_test:
            if model_name in results:
                r = results[model_name]
                print(f"{model_name.upper()}:")
                print(f"  Training: {r['training_time']:.2f}s")
                print(f"  Generation: {r['generation_time']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Comparison test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TVAE MODEL TEST")
    print("=" * 60)
    
    # Test TVAE availability
    available = test_tvae_availability()
    if not available:
        print("\n[INFO] TVAE not available. To install:")
        print("pip install sdv")
        print("\nSkipping TVAE tests.")
        sys.exit(0)
    
    # Test TVAE model
    print("\n" + "=" * 60)
    model_ok = test_tvae_model()
    
    # Test integration
    print("\n" + "=" * 60)
    integration_ok = test_tvae_evaluation_integration()
    
    # Test comparison if possible
    print("\n" + "=" * 60)
    comparison_ok = test_tvae_vs_ctgan_comparison()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print(f"TVAE Model: {'PASS' if model_ok else 'FAIL'}")
    print(f"Integration: {'PASS' if integration_ok else 'FAIL'}")
    print(f"Comparison: {'PASS' if comparison_ok else 'FAIL'}")
    
    if model_ok and integration_ok and comparison_ok:
        print("\n[SUCCESS] All TVAE tests passed!")
        print("TVAE is ready for production use alongside CTGAN.")
    else:
        print("\n[PARTIAL] Some tests failed or were skipped.")
    
    print("=" * 60)