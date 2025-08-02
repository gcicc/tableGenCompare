#!/usr/bin/env python3
"""
Test only the working components to validate our framework.
This focuses on GANerAid + evaluation pipeline which we know works.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.append('src')

def create_test_data():
    """Create numerical test data."""
    np.random.seed(42)
    
    n_samples = 200  # Small for quick testing
    data = pd.DataFrame({
        'age': np.random.randint(20, 70, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'score': np.random.uniform(0, 100, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    return data

def test_basic_framework():
    """Test the basic framework components we know work."""
    print("Testing basic framework components...")
    
    try:
        # Test evaluation framework (we know this works)
        from evaluation.unified_evaluator import UnifiedEvaluator
        print("[OK] UnifiedEvaluator import successful")
        
        # Test GANerAid model (we know this works)
        from models.model_factory import ModelFactory
        
        available_models = ModelFactory.list_available_models()
        print(f"[OK] Available models: {available_models}")
        
        if available_models.get('ganeraid', False):
            print("[OK] GANerAid is available")
        else:
            print("[WARN] GANerAid not available")
            return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic framework test failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete GANerAid + evaluation pipeline."""
    print("Testing full GANerAid pipeline...")
    
    try:
        from models.model_factory import ModelFactory
        from evaluation.unified_evaluator import UnifiedEvaluator
        
        # Create test data
        test_data = create_test_data()
        print(f"[OK] Test data created: {test_data.shape}")
        
        # Create and train GANerAid model
        model = ModelFactory.create('ganeraid', random_state=42)
        print("[OK] GANerAid model created")
        
        # Train with minimal epochs for testing
        training_result = model.train(test_data, epochs=30, verbose=False)
        print(f"[OK] Training completed in {training_result.get('training_duration_seconds', 'N/A'):.2f}s")
        
        # Generate synthetic data
        synthetic_data = model.generate(100)
        print(f"[OK] Generated synthetic data: {synthetic_data.shape}")
        
        # Run evaluation
        evaluator = UnifiedEvaluator(random_state=42)
        results = evaluator.run_complete_evaluation(
            model=model,
            original_data=test_data,
            synthetic_data=synthetic_data,
            dataset_metadata={
                'dataset_info': {'name': 'pipeline_test'},
                'target_info': {'column': 'target', 'type': 'binary'}
            },
            output_dir="test_pipeline_output",
            target_column='target'
        )
        
        print("[OK] Full pipeline evaluation completed!")
        print(f"  TRTS Overall Score: {results['trts_results']['overall_score_percent']:.1f}%")
        print(f"  Similarity Score: {results['similarity_analysis']['final_similarity']:.3f}")
        print(f"  Data Quality: {results['data_quality']['data_type_consistency']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_hyperparameter_comparison():
    """Test manual hyperparameter comparison to simulate optimization."""
    print("Testing manual hyperparameter comparison...")
    
    try:
        from models.model_factory import ModelFactory
        from evaluation.unified_evaluator import UnifiedEvaluator
        
        test_data = create_test_data()
        evaluator = UnifiedEvaluator(random_state=42)
        
        # Test 3 different hyperparameter sets
        hyperparameter_sets = [
            {'epochs': 25, 'lr_g': 0.0005, 'lr_d': 0.0005, 'hidden_feature_space': 150},
            {'epochs': 40, 'lr_g': 0.001, 'lr_d': 0.0008, 'hidden_feature_space': 200},
            {'epochs': 30, 'lr_g': 0.0002, 'lr_d': 0.0003, 'hidden_feature_space': 250}
        ]
        
        results = []
        
        for i, params in enumerate(hyperparameter_sets):
            print(f"\nTesting parameter set {i+1}: {params}")
            
            # Create and configure model
            model = ModelFactory.create('ganeraid', random_state=42)
            model.set_config(params)
            
            # Train and generate
            model.train(test_data, **params, verbose=False)
            synthetic_data = model.generate(len(test_data))
            
            # Evaluate
            eval_results = evaluator.run_complete_evaluation(
                model=model,
                original_data=test_data,
                synthetic_data=synthetic_data,
                dataset_metadata={
                    'dataset_info': {'name': f'param_test_{i}'},
                    'target_info': {'column': 'target', 'type': 'binary'}
                },
                output_dir=f"test_params_{i}",
                target_column='target'
            )
            
            # Store results
            result = {
                'params': params,
                'trts_overall': eval_results['trts_results']['overall_score_percent'],
                'similarity': eval_results['similarity_analysis']['final_similarity'],
                'data_quality': eval_results['data_quality']['data_type_consistency']
            }
            results.append(result)
            
            print(f"  TRTS Score: {result['trts_overall']:.1f}%")
            print(f"  Similarity: {result['similarity']:.3f}")
        
        # Find best configuration
        best_config = max(results, key=lambda x: x['trts_overall'])
        
        print(f"\n[OK] Manual hyperparameter comparison completed!")
        print(f"Best configuration: {best_config['params']}")
        print(f"Best TRTS score: {best_config['trts_overall']:.1f}%")
        print(f"Best similarity: {best_config['similarity']:.3f}")
        
        print(f"\nAll Results:")
        for i, result in enumerate(results):
            print(f"  Config {i+1}: TRTS={result['trts_overall']:.1f}%, Sim={result['similarity']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Manual hyperparameter comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_variations():
    """Test GANerAid with different configurations."""
    print("Testing GANerAid with different configurations...")
    
    try:
        from models.model_factory import ModelFactory
        
        test_data = create_test_data()
        
        # Test different model configurations
        configurations = [
            "Default Configuration",
            "High Learning Rate",
            "Large Hidden Space", 
            "Small Batch Size"
        ]
        
        configs = [
            {},  # Default
            {'lr_g': 0.002, 'lr_d': 0.002},
            {'hidden_feature_space': 300},
            {'batch_size': 50}
        ]
        
        for i, (name, config) in enumerate(zip(configurations, configs)):
            print(f"\nTesting: {name}")
            
            model = ModelFactory.create('ganeraid', random_state=42)
            if config:
                model.set_config(config)
            
            # Quick training
            model.train(test_data, epochs=20, verbose=False)
            synthetic_data = model.generate(50)
            
            print(f"  Generated: {synthetic_data.shape}")
            print(f"  Data types match: {list(synthetic_data.dtypes) == list(test_data.dtypes)}")
        
        print("[OK] All model configurations tested successfully!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Model variations test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("WORKING COMPONENTS TEST")
    print("=" * 60)
    
    # Test basic framework
    basic_ok = test_basic_framework()
    
    if basic_ok:
        # Test full pipeline
        print("\n" + "=" * 60)
        pipeline_ok = test_full_pipeline()
        
        # Test manual hyperparameter comparison
        print("\n" + "=" * 60)
        hyperparam_ok = test_manual_hyperparameter_comparison()
        
        # Test model variations
        print("\n" + "=" * 60)
        variations_ok = test_model_variations()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print(f"Basic Framework: {'PASS' if basic_ok else 'FAIL'}")
        print(f"Full Pipeline: {'PASS' if pipeline_ok else 'FAIL'}")
        print(f"Hyperparameter Testing: {'PASS' if hyperparam_ok else 'FAIL'}")
        print(f"Model Variations: {'PASS' if variations_ok else 'FAIL'}")
        
        if basic_ok and pipeline_ok and hyperparam_ok and variations_ok:
            print("\n[SUCCESS] All working components validated!")
            print("✅ GANerAid model: WORKING")
            print("✅ Evaluation pipeline: WORKING") 
            print("✅ Hyperparameter testing: WORKING")
            print("✅ Model configurations: WORKING")
            print("\nFramework is ready for:")
            print("  - Production use with GANerAid")
            print("  - Manual hyperparameter optimization") 
            print("  - Adding CTGAN/TVAE when dependencies installed")
            print("  - Full Optuna optimization when imports fixed")
        else:
            print("\n[PARTIAL] Some components failed.")
    else:
        print("\n[FAIL] Basic framework not working - cannot continue")
    
    print("=" * 60)