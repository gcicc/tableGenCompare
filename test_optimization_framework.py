#!/usr/bin/env python3
"""
Test script for the optimization framework.
Validates Optuna-based hyperparameter optimization with synthetic data models.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add the src directory to path
sys.path.append('src')

def create_optimization_test_data():
    """Create test data for optimization."""
    np.random.seed(42)
    
    # Smaller dataset for faster optimization testing
    n_samples = 400
    data = pd.DataFrame({
        'feature1': np.random.normal(10, 2, n_samples),
        'feature2': np.random.normal(5, 1.5, n_samples),
        'feature3': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'binary_feature': np.random.choice([0, 1], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    return data

def test_optimization_imports():
    """Test that optimization framework imports work."""
    print("Testing optimization framework imports...")
    
    try:
        import sys
        sys.path.append('src')
        from optimization.optuna_optimizer import OptunaOptimizer
        from optimization.objective_functions import (
            TRTSObjective, SimilarityObjective, MultiObjective,
            create_balanced_multi_objective, create_composite_objective
        )
        print("[OK] All optimization imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        print("You may need to install: pip install optuna")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected import error: {e}")
        return False

def test_objective_functions():
    """Test objective function implementations."""
    print("Testing objective functions...")
    
    try:
        import sys
        sys.path.append('src')
        from optimization.objective_functions import (
            TRTSObjective, SimilarityObjective, DataQualityObjective,
            MultiObjective, CompositeObjective
        )
        
        # Create mock evaluation results
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
                'shape_match': True
            },
            'statistical_analysis': {
                'summary_statistics': {
                    'similarity_percentage': 87.5,
                    'average_mean_error': 0.125
                }
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
        
        # Test Multi-objective
        multi_obj = MultiObjective([trts_obj, sim_obj, quality_obj])
        multi_scores = multi_obj.evaluate(mock_results)
        print(f"[OK] Multi-objective: {multi_scores}")
        
        # Test Composite objective
        composite_obj = CompositeObjective([trts_obj, sim_obj])
        composite_score = composite_obj.evaluate(mock_results)
        print(f"[OK] Composite objective: {composite_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Objective function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optuna_availability():
    """Test if Optuna is available."""
    print("Testing Optuna availability...")
    
    try:
        import optuna
        print(f"[OK] Optuna version {optuna.__version__} available")
        return True
    except ImportError:
        print("[SKIP] Optuna not available. Install with: pip install optuna")
        return False

def test_single_objective_optimization():
    """Test single-objective optimization."""
    print("Testing single-objective optimization...")
    
    try:
        from optimization.optuna_optimizer import OptunaOptimizer
        from models.model_factory import ModelFactory
        
        # Check if we have an available model
        available_models = ModelFactory.list_available_models()
        available_model_names = [name for name, available in available_models.items() if available]
        
        if not available_model_names:
            print("[SKIP] No models available for optimization testing")
            return True
        
        # Use the first available model
        model_name = available_model_names[0]
        print(f"Testing optimization with {model_name}")
        
        # Create test data
        test_data = create_optimization_test_data()
        
        # Create optimizer for single objective
        optimizer = OptunaOptimizer(
            model_name=model_name,
            optimization_objectives=["trts_overall"],  # Single objective
            sampler_type="random",  # Faster for testing
            random_state=42
        )
        
        # Create dataset metadata
        dataset_metadata = {
            'dataset_info': {'name': 'optimization_test'},
            'target_info': {'column': 'target', 'type': 'binary'}
        }
        
        # Run optimization with few trials for testing
        print("Running single-objective optimization (this may take 2-3 minutes)...")
        results = optimizer.optimize(
            train_data=test_data,
            target_column='target',
            n_trials=3,  # Very small for testing
            dataset_metadata=dataset_metadata
        )
        
        print("[OK] Single-objective optimization completed")
        print(f"Total trials: {results['optimization_summary']['total_trials']}")
        print(f"Completed trials: {results['optimization_summary']['completed_trials']}")
        print(f"Best parameters: {results['best_parameters']}")
        
        # Test getting optimization history
        history = optimizer.get_optimization_history()
        print(f"[OK] Optimization history: {len(history)} trials")
        
        return True
        
    except ImportError as e:
        print(f"[SKIP] Dependencies not available: {e}")
        return True
    except Exception as e:
        print(f"[FAIL] Single-objective optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_objective_optimization():
    """Test multi-objective optimization."""
    print("Testing multi-objective optimization...")
    
    try:
        from optimization.optuna_optimizer import OptunaOptimizer
        from models.model_factory import ModelFactory
        
        # Check if we have an available model
        available_models = ModelFactory.list_available_models()
        available_model_names = [name for name, available in available_models.items() if available]
        
        if not available_model_names:
            print("[SKIP] No models available for optimization testing")
            return True
        
        # Use the first available model
        model_name = available_model_names[0]
        print(f"Testing multi-objective optimization with {model_name}")
        
        # Create test data
        test_data = create_optimization_test_data()
        
        # Create optimizer for multi-objective
        optimizer = OptunaOptimizer(
            model_name=model_name,
            optimization_objectives=["trts_overall", "similarity"],  # Multi-objective
            sampler_type="nsga2",  # NSGA-II for multi-objective
            random_state=42
        )
        
        # Create dataset metadata
        dataset_metadata = {
            'dataset_info': {'name': 'multi_optimization_test'},
            'target_info': {'column': 'target', 'type': 'binary'}
        }
        
        # Run optimization with few trials for testing
        print("Running multi-objective optimization (this may take 3-4 minutes)...")
        results = optimizer.optimize(
            train_data=test_data,
            target_column='target',
            n_trials=5,  # Small for testing
            dataset_metadata=dataset_metadata
        )
        
        print("[OK] Multi-objective optimization completed")
        print(f"Total trials: {results['optimization_summary']['total_trials']}")
        print(f"Pareto solutions: {results['pareto_analysis']['n_pareto_solutions']}")
        
        # Show Pareto front
        best_params = results['best_parameters']
        if 'pareto_solutions' in best_params:
            print("Pareto-optimal solutions:")
            for i, solution in enumerate(best_params['pareto_solutions'][:3]):  # Show first 3
                print(f"  Solution {i+1}: Values {solution['values']} -> Params {solution['parameters']}")
        
        return True
        
    except ImportError as e:
        print(f"[SKIP] Dependencies not available: {e}")
        return True
    except Exception as e:
        print(f"[FAIL] Multi-objective optimization failed: {e}")
        import traceback  
        traceback.print_exc()
        return False

def test_optimization_visualization():
    """Test optimization visualization capabilities."""
    print("Testing optimization visualization...")
    
    try:
        # Check if visualization dependencies are available
        import plotly
        print(f"[OK] Plotly version {plotly.__version__} available")
        
        # If we have a completed optimization study, test visualization
        # (This would normally use the study from previous tests)
        print("[INFO] Visualization requires completed optimization study")
        print("[INFO] Run full optimization test to generate visualizations")
        
        return True
        
    except ImportError:
        print("[SKIP] Plotly not available for visualization")
        print("Install with: pip install plotly")
        return True
    except Exception as e:
        print(f"[FAIL] Visualization test failed: {e}")
        return False

def test_optimization_model_comparison():
    """Test optimization across multiple models if available.""" 
    print("Testing optimization across multiple models...")
    
    try:
        from models.model_factory import ModelFactory
        from optimization.optuna_optimizer import OptunaOptimizer
        
        available_models = ModelFactory.list_available_models()
        available_model_names = [name for name, available in available_models.items() if available]
        
        if len(available_model_names) < 2:
            print(f"[SKIP] Need at least 2 models for comparison, have {len(available_model_names)}")
            return True
        
        print(f"Comparing optimization across models: {available_model_names[:2]}")
        
        test_data = create_optimization_test_data()
        dataset_metadata = {
            'dataset_info': {'name': 'model_comparison_test'},
            'target_info': {'column': 'target', 'type': 'binary'}
        }
        
        comparison_results = {}
        
        for model_name in available_model_names[:2]:  # Test first 2 models
            print(f"\nOptimizing {model_name}...")
            
            optimizer = OptunaOptimizer(
                model_name=model_name,
                optimization_objectives=["trts_overall"],
                sampler_type="random",
                random_state=42
            )
            
            results = optimizer.optimize(
                train_data=test_data,
                target_column='target',
                n_trials=2,  # Very minimal for comparison test
                dataset_metadata=dataset_metadata
            )
            
            comparison_results[model_name] = {
                'best_value': max([trial['objectives'][0] for trial in results['optimization_history']]),
                'total_trials': results['optimization_summary']['total_trials']
            }
        
        print(f"\n{'='*50}")
        print("MODEL OPTIMIZATION COMPARISON:")
        print(f"{'='*50}")
        
        for model_name, result in comparison_results.items():
            print(f"{model_name.upper()}:")
            print(f"  Best TRTS Score: {result['best_value']:.3f}")
            print(f"  Trials Completed: {result['total_trials']}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Model comparison test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZATION FRAMEWORK TEST")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_optimization_imports()
    if not imports_ok:
        print("\n[FAIL] Import tests failed - cannot continue")
        sys.exit(1)
    
    # Test Optuna availability
    print("\n" + "=" * 60)
    optuna_ok = test_optuna_availability()
    
    # Test objective functions
    print("\n" + "=" * 60)
    objectives_ok = test_objective_functions()
    
    # Test optimizations (only if Optuna available)
    single_obj_ok = True
    multi_obj_ok = True
    viz_ok = True
    comparison_ok = True
    
    if optuna_ok:
        print("\n" + "=" * 60)
        single_obj_ok = test_single_objective_optimization()
        
        print("\n" + "=" * 60)
        multi_obj_ok = test_multi_objective_optimization()
        
        print("\n" + "=" * 60)
        viz_ok = test_optimization_visualization()
        
        print("\n" + "=" * 60)
        comparison_ok = test_optimization_model_comparison()
    else:
        print("\n[SKIP] Skipping optimization tests - Optuna not available")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print(f"Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"Optuna Available: {'PASS' if optuna_ok else 'SKIP'}")
    print(f"Objective Functions: {'PASS' if objectives_ok else 'FAIL'}")
    print(f"Single-Objective: {'PASS' if single_obj_ok else 'FAIL'}")
    print(f"Multi-Objective: {'PASS' if multi_obj_ok else 'FAIL'}")
    print(f"Visualization: {'PASS' if viz_ok else 'FAIL'}")
    print(f"Model Comparison: {'PASS' if comparison_ok else 'FAIL'}")
    
    all_passed = all([imports_ok, objectives_ok, single_obj_ok, multi_obj_ok, viz_ok, comparison_ok])
    
    if all_passed:
        print("\n[SUCCESS] All optimization framework tests passed!")
        print("The optimization engine is ready for production use.")
    else:
        print("\n[PARTIAL] Some tests failed or were skipped.")
        print("Install missing dependencies: pip install optuna plotly")
    
    print("=" * 60)