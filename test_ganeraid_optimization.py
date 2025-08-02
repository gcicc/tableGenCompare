#!/usr/bin/env python3
"""
Test GANerAid with optimization framework.
This tests the integration of our working GANerAid model with optimization.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.append('src')

def create_test_data():
    """Create numerical test data for GANerAid."""
    np.random.seed(42)
    
    n_samples = 300  # Small for quick testing
    data = pd.DataFrame({
        'feature1': np.random.normal(10, 2, n_samples),
        'feature2': np.random.normal(5, 1.5, n_samples),
        'feature3': np.random.uniform(0, 100, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    return data

def test_objective_functions_directly():
    """Test objective functions with mock data."""
    print("Testing objective functions directly...")
    
    try:
        from optimization.objective_functions import (
            TRTSObjective, SimilarityObjective, DataQualityObjective
        )
        
        # Create mock evaluation results (like what UnifiedEvaluator returns)
        mock_results = {
            'trts_results': {
                'overall_score_percent': 85.5,
                'utility_score_percent': 82.3,
                'quality_score_percent': 88.7
            },
            'similarity_analysis': {
                'final_similarity': 0.923,
                'univariate_similarity': 0.945,
                'bivariate_similarity': 0.901
            },
            'data_quality': {
                'data_type_consistency': 100.0,
                'range_validity_percentage': 95.2,
                'column_match': True,
                'shape_match': False
            }
        }
        
        # Test TRTS objective
        trts_obj = TRTSObjective(metric="overall")
        trts_score = trts_obj.evaluate(mock_results)
        print(f"[OK] TRTS objective: {trts_score:.3f}")
        
        # Test Similarity objective
        sim_obj = SimilarityObjective(metric="final")
        sim_score = sim_obj.evaluate(mock_results)
        print(f"[OK] Similarity objective: {sim_score:.3f}")
        
        # Test Data Quality objective
        quality_obj = DataQualityObjective(metric="overall")
        quality_score = quality_obj.evaluate(mock_results)
        print(f"[OK] Data Quality objective: {quality_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Objective functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_ganeraid_optimization():
    """Test manual hyperparameter optimization loop with GANerAid."""
    print("Testing manual GANerAid optimization...")
    
    try:
        from models.model_factory import ModelFactory
        from evaluation.unified_evaluator import UnifiedEvaluator
        from optimization.objective_functions import TRTSObjective
        
        # Create test data
        test_data = create_test_data()
        
        # Create evaluator and objective
        evaluator = UnifiedEvaluator(random_state=42)
        objective = TRTSObjective(metric="overall")
        
        dataset_metadata = {
            'dataset_info': {'name': 'manual_optimization_test'},
            'target_info': {'column': 'target', 'type': 'binary'}
        }
        
        # Test different hyperparameter configurations manually
        configs_to_test = [
            {'epochs': 50, 'lr_g': 0.0005, 'lr_d': 0.0005},
            {'epochs': 100, 'lr_g': 0.001, 'lr_d': 0.0005},
            {'epochs': 75, 'lr_g': 0.0002, 'lr_d': 0.001}
        ]
        
        results = []
        
        for i, config in enumerate(configs_to_test):
            print(f"Testing configuration {i+1}: {config}")
            
            # Create and configure model
            model = ModelFactory.create('ganeraid', random_state=42)
            model.set_config(config)
            
            # Train model
            model.train(test_data, **config)
            
            # Generate synthetic data
            synthetic_data = model.generate(len(test_data))
            
            # Evaluate
            eval_results = evaluator.run_complete_evaluation(
                model=model,
                original_data=test_data,
                synthetic_data=synthetic_data,
                dataset_metadata=dataset_metadata,
                output_dir=f"manual_opt_test_{i}",
                target_column='target'
            )
            
            # Calculate objective
            objective_score = objective.evaluate(eval_results)
            
            results.append({
                'config': config,
                'objective_score': objective_score,
                'trts_overall': eval_results['trts_results']['overall_score_percent'],
                'similarity': eval_results['similarity_analysis']['final_similarity']
            })
            
            print(f"  Objective Score: {objective_score:.3f}")
            print(f"  TRTS Overall: {eval_results['trts_results']['overall_score_percent']:.1f}%")
        
        # Find best configuration
        best_result = max(results, key=lambda x: x['objective_score'])
        
        print(f"\n[OK] Manual optimization completed!")
        print(f"Best configuration: {best_result['config']}")
        print(f"Best objective score: {best_result['objective_score']:.3f}")
        print(f"Best TRTS score: {best_result['trts_overall']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Manual optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ganeraid_with_different_objectives():
    """Test GANerAid with different optimization objectives."""
    print("Testing GANerAid with different objectives...")
    
    try:
        from models.model_factory import ModelFactory
        from evaluation.unified_evaluator import UnifiedEvaluator
        from optimization.objective_functions import (
            TRTSObjective, SimilarityObjective, DataQualityObjective,
            CompositeObjective
        )
        
        # Create test data and run one evaluation
        test_data = create_test_data()
        model = ModelFactory.create('ganeraid', random_state=42)
        model.train(test_data, epochs=50, verbose=False)
        synthetic_data = model.generate(len(test_data))
        
        evaluator = UnifiedEvaluator(random_state=42)
        eval_results = evaluator.run_complete_evaluation(
            model=model,
            original_data=test_data,
            synthetic_data=synthetic_data,
            dataset_metadata={'dataset_info': {'name': 'objective_test'}},
            output_dir="objective_test_output",
            target_column='target'
        )
        
        # Test different objectives
        objectives = [
            TRTSObjective(metric="overall"),
            TRTSObjective(metric="utility"),
            TRTSObjective(metric="quality"),
            SimilarityObjective(metric="final"),
            DataQualityObjective(metric="overall")
        ]
        
        print("Objective scores:")
        for obj in objectives:
            score = obj.evaluate(eval_results)
            print(f"  {obj.name}: {score:.3f}")
        
        # Test composite objective
        composite = CompositeObjective([
            TRTSObjective(metric="overall"),
            SimilarityObjective(metric="final")
        ], weights=[0.7, 0.3])
        
        composite_score = composite.evaluate(eval_results)
        print(f"  composite_objective: {composite_score:.3f}")
        
        print("[OK] All objectives evaluated successfully!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Objective evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("GANERAID OPTIMIZATION TEST")
    print("=" * 60)
    
    # Test objective functions
    obj_ok = test_objective_functions_directly()
    
    # Test manual optimization
    print("\n" + "=" * 60)
    manual_ok = test_manual_ganeraid_optimization()
    
    # Test different objectives
    print("\n" + "=" * 60)
    objectives_ok = test_ganeraid_with_different_objectives()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print(f"Objective Functions: {'PASS' if obj_ok else 'FAIL'}")
    print(f"Manual Optimization: {'PASS' if manual_ok else 'FAIL'}")
    print(f"Multiple Objectives: {'PASS' if objectives_ok else 'FAIL'}")
    
    if obj_ok and manual_ok and objectives_ok:
        print("\n[SUCCESS] GANerAid optimization framework working perfectly!")
        print("Ready for full Optuna-based optimization!")
    else:
        print("\n[PARTIAL] Some tests failed.")
    
    print("=" * 60)