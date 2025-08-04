#!/usr/bin/env python3
"""
Test script for advanced hyperparameter optimization.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add src to path
sys.path.append('src')

from optimization.advanced_optimizer import AdvancedOptunaOptimizer, optimize_model_hyperparameters

def create_test_data():
    """Create test dataset for optimization."""
    np.random.seed(42)
    
    n_samples = 300  # Moderate size for optimization testing
    data = pd.DataFrame({
        'age': np.random.randint(20, 70, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'experience': np.random.uniform(0, 30, n_samples),
        'score': np.random.uniform(0, 100, n_samples),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    return data

def test_optimizer_initialization():
    """Test optimizer initialization with different models."""
    print("Testing optimizer initialization...")
    
    try:
        from models.model_factory import ModelFactory
        available_models = ModelFactory.list_available_models()
        
        tested_models = []
        for model_name, is_available in available_models.items():
            if is_available:
                try:
                    optimizer = AdvancedOptunaOptimizer(
                        model_name=model_name,
                        sampler_name="tpe",
                        pruner_name="median"
                    )
                    print(f"✅ {model_name}: Optimizer initialized successfully")
                    
                    # Check hyperparameter space
                    hyperparam_count = len(optimizer.hyperparameter_space)
                    print(f"  Hyperparameter space: {hyperparam_count} parameters")
                    
                    tested_models.append(model_name)
                    
                except Exception as e:
                    print(f"❌ {model_name}: Failed to initialize - {e}")
            else:
                print(f"⏭️  {model_name}: Not available (dependencies missing)")
        
        if tested_models:
            print(f"✅ Successfully initialized optimizers for: {tested_models}")
            return True, tested_models
        else:
            print("❌ No models available for optimization testing")
            return False, []
            
    except Exception as e:
        print(f"❌ Optimizer initialization test failed: {e}")
        return False, []

def test_single_objective_optimization(model_name: str):
    """Test single-objective optimization."""
    print(f"Testing single-objective optimization for {model_name}...")
    
    try:
        # Create test data
        test_data = create_test_data()
        print(f"Test data created: {test_data.shape}")
        
        # Create optimizer
        optimizer = AdvancedOptunaOptimizer(
            model_name=model_name,
            sampler_name="tpe",
            pruner_name="median"
        )
        
        # Run short optimization
        study = optimizer.optimize(
            training_data=test_data,
            study_name=f"test_{model_name}_single",
            n_trials=5,  # Quick test
            target_column='target',
            objective_type="composite",
            validation_split=0.3,
            show_progress=True
        )
        
        # Check results
        completed_trials = [t for t in study.trials if t.state.name == 'COMPLETE']
        print(f"✅ Completed {len(completed_trials)}/5 trials")
        
        if completed_trials:
            best_params = optimizer.get_best_hyperparameters(1)
            if best_params:
                print(f"✅ Best parameters found: {best_params[0]}")
                print(f"✅ Best objective value: {study.best_value:.4f}")
            else:
                print("⚠️  No best parameters available")
        
        return True
        
    except Exception as e:
        print(f"❌ Single-objective optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_objective_optimization(model_name: str):
    """Test multi-objective optimization."""
    print(f"Testing multi-objective optimization for {model_name}...")
    
    try:
        # Create test data
        test_data = create_test_data()
        
        # Create optimizer
        optimizer = AdvancedOptunaOptimizer(
            model_name=model_name,
            sampler_name="nsgaii",  # Better for multi-objective
            pruner_name="none"  # Disable pruning for multi-objective
        )
        
        # Run multi-objective optimization
        study = optimizer.optimize(
            training_data=test_data,
            study_name=f"test_{model_name}_multi",
            n_trials=5,  # Quick test
            target_column='target',
            objective_type="multi",  # Multi-objective
            validation_split=0.3,
            show_progress=True
        )
        
        # Check results
        completed_trials = [t for t in study.trials if t.state.name == 'COMPLETE']
        print(f"✅ Completed {len(completed_trials)}/5 trials")
        
        if completed_trials and hasattr(study, 'best_trials'):
            pareto_solutions = study.best_trials
            print(f"✅ Found {len(pareto_solutions)} Pareto optimal solutions")
            
            if pareto_solutions:
                best_solution = pareto_solutions[0]
                print(f"✅ Best solution values: {best_solution.values}")
                print(f"✅ Best solution params: {best_solution.params}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-objective optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_persistence(model_name: str):
    """Test study persistence and loading."""
    print(f"Testing optimization persistence for {model_name}...")
    
    try:
        # Create test data
        test_data = create_test_data()
        
        # Create optimizer
        optimizer = AdvancedOptunaOptimizer(
            model_name=model_name,
            sampler_name="random",  # Fast for testing
            storage="sqlite:///test_optimization.db"  # Persistent storage
        )
        
        # Run initial optimization
        study1 = optimizer.optimize(
            training_data=test_data,
            study_name="persistence_test",
            n_trials=3,
            target_column='target',
            objective_type="composite",
            show_progress=False
        )
        
        initial_trials = len(study1.trials)
        print(f"✅ Initial optimization: {initial_trials} trials")
        
        # Save study
        study_file = "test_study.json"
        optimizer.save_study(study_file)
        print(f"✅ Study saved to {study_file}")
        
        # Create new optimizer and resume
        optimizer2 = AdvancedOptunaOptimizer(
            model_name=model_name,
            sampler_name="random",
            storage="sqlite:///test_optimization.db"
        )
        
        # Resume optimization
        study2 = optimizer2.optimize(
            training_data=test_data,
            study_name="persistence_test",  # Same name to resume
            n_trials=2,  # Additional trials
            target_column='target',
            objective_type="composite",
            show_progress=False
        )
        
        final_trials = len(study2.trials)
        print(f"✅ Resumed optimization: {final_trials} total trials")
        
        if final_trials > initial_trials:
            print("✅ Study successfully resumed and extended")
            return True
        else:
            print("⚠️  Study resumption may not have worked as expected")
            return False
        
    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
        return False

def test_convenience_function():
    """Test the convenience optimization function."""
    print("Testing convenience optimization function...")
    
    try:
        from models.model_factory import ModelFactory
        available_models = ModelFactory.list_available_models()
        
        # Find first available model
        test_model = None
        for model_name, is_available in available_models.items():
            if is_available:
                test_model = model_name
                break
        
        if not test_model:
            print("⏭️  No models available for convenience function test")
            return True
        
        # Create test data
        test_data = create_test_data()
        
        # Run optimization using convenience function
        results = optimize_model_hyperparameters(
            model_name=test_model,
            training_data=test_data,
            n_trials=3,  # Quick test
            target_column='target',
            objective_type="composite",
            output_dir="test_optimization_output"
        )
        
        # Check results
        required_keys = ['study', 'best_parameters', 'study_file', 'report_file', 'n_trials']
        for key in required_keys:
            if key not in results:
                print(f"❌ Missing key in results: {key}")
                return False
        
        print(f"✅ Optimization completed with {results['n_trials']} trials")
        print(f"✅ Best parameters: {results['best_parameters']}")
        print(f"✅ Study saved to: {results['study_file']}")
        print(f"✅ Report saved to: {results['report_file']}")
        
        if results['best_objective_value'] is not None:
            print(f"✅ Best objective value: {results['best_objective_value']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_samplers(model_name: str):
    """Test different sampling strategies."""
    print(f"Testing different samplers for {model_name}...")
    
    samplers_to_test = ["tpe", "random", "cmaes"]
    test_data = create_test_data()
    
    results = {}
    
    for sampler in samplers_to_test:
        print(f"  Testing {sampler} sampler...")
        
        try:
            optimizer = AdvancedOptunaOptimizer(
                model_name=model_name,
                sampler_name=sampler
            )
            
            study = optimizer.optimize(
                training_data=test_data,
                study_name=f"test_{sampler}",
                n_trials=3,  # Quick test
                objective_type="composite",
                show_progress=False
            )
            
            completed = len([t for t in study.trials if t.state.name == 'COMPLETE'])
            results[sampler] = completed
            print(f"    ✅ {sampler}: {completed}/3 trials completed")
            
        except Exception as e:
            print(f"    ❌ {sampler}: Failed - {e}")
            results[sampler] = 0
    
    successful_samplers = [s for s, count in results.items() if count > 0]
    print(f"✅ Working samplers: {successful_samplers}")
    
    return len(successful_samplers) > 0

def run_comprehensive_optimization_test():
    """Run comprehensive optimization test suite."""
    print("=" * 60)
    print("ADVANCED OPTIMIZATION TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Optimizer initialization
    init_success, available_models = test_optimizer_initialization()
    results['initialization'] = init_success
    print()
    
    if not available_models:
        print("❌ No models available - cannot continue with optimization tests")
        return False
    
    # Use first available model for remaining tests
    test_model = available_models[0]
    print(f"Using {test_model} for remaining tests\n")
    
    # Test 2: Single-objective optimization
    results['single_objective'] = test_single_objective_optimization(test_model)
    print()
    
    # Test 3: Multi-objective optimization
    results['multi_objective'] = test_multi_objective_optimization(test_model)
    print()
    
    # Test 4: Different samplers
    results['samplers'] = test_different_samplers(test_model)
    print()
    
    # Test 5: Persistence
    results['persistence'] = test_optimization_persistence(test_model)
    print()
    
    # Test 6: Convenience function
    results['convenience'] = test_convenience_function()
    print()
    
    # Summary
    print("=" * 60)
    print("OPTIMIZATION TEST RESULTS")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper().replace('_', ' '):.<25} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All optimization tests passed!")
        print("✅ Advanced optimization framework is working correctly")
        return True
    else:
        print("⚠️  Some optimization tests failed")
        print("   Check the output above for details")
        return False

if __name__ == "__main__":
    success = run_comprehensive_optimization_test()
    sys.exit(0 if success else 1)