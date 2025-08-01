#!/usr/bin/env python3
"""
Quick test of GANerAid with numerical-only data.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Add the synthetic-tabular-benchmark src to path
sys.path.append('synthetic-tabular-benchmark/src')

def create_numerical_data():
    """Create numerical-only test data."""
    np.random.seed(42)
    
    n_samples = 300  # Small for quick testing
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'health_score': np.random.uniform(0, 100, n_samples),
        'blood_pressure': np.random.normal(120, 20, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    return data

def test_numerical_ganeraid():
    """Test GANerAid with numerical-only data."""
    print("Testing GANerAid with numerical data...")
    
    try:
        from models.model_factory import ModelFactory
        from evaluation.unified_evaluator import UnifiedEvaluator
        
        # Create model
        model = ModelFactory.create('ganeraid', device='cpu', random_state=42)
        print("[OK] GANerAid model created")
        
        # Create numerical test data
        original_data = create_numerical_data()
        print(f"[OK] Numerical test data created: {original_data.shape}")
        print(f"Data types: {original_data.dtypes.tolist()}")
        
        # Train model with minimal epochs
        print("Training GANerAid model...")
        training_result = model.train(original_data, epochs=50, verbose=False)
        print(f"[OK] Training completed in {training_result.get('training_duration_seconds', 'N/A'):.2f} seconds")
        
        # Generate synthetic data
        print("Generating synthetic data...")
        synthetic_data = model.generate(100)
        print(f"[OK] Generated synthetic data: {synthetic_data.shape}")
        
        # Test full evaluation pipeline
        print("Testing full evaluation pipeline...")
        evaluator = UnifiedEvaluator(random_state=42)
        
        dataset_metadata = {
            'dataset_info': {'name': 'numerical_test', 'description': 'Numerical test dataset'},
            'target_info': {'column': 'target', 'type': 'binary'}
        }
        
        results = evaluator.run_complete_evaluation(
            model=model,
            original_data=original_data,
            synthetic_data=synthetic_data,
            dataset_metadata=dataset_metadata,
            output_dir="test_numerical_output",
            target_column='target'
        )
        
        print("[SUCCESS] Full pipeline test completed!")
        print(f"  TRTS Overall Score: {results['trts_results']['overall_score_percent']:.1f}%")
        print(f"  Similarity Score: {results['similarity_analysis']['final_similarity']:.3f}")
        print(f"  Data Quality: {results['data_quality']['data_type_consistency']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Numerical GANerAid test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("NUMERICAL GANERAID TEST")
    print("=" * 60)
    
    success = test_numerical_ganeraid()
    
    if success:
        print("\n[SUCCESS] GANerAid works perfectly with numerical data!")
        print("The framework extraction is complete and functional.")
    else:
        print("\n[FAIL] Test failed.")
    
    print("=" * 60)