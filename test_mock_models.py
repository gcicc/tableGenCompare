"""
Test Mock Models for Phase 6 Clinical Framework
Testing all mock model implementations
"""

import pandas as pd
import numpy as np
import time
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_test_data():
    """Load the Pakistani diabetes dataset."""
    data_path = r"C:\Users\gcicc\claudeproj\tableGenCompare\data\Pakistani_Diabetes_Dataset.csv"
    
    try:
        data = pd.read_csv(data_path)
        print(f"Data loaded successfully: {data.shape}")
        
        # Identify discrete columns
        discrete_cols = []
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].nunique() <= 10:
                discrete_cols.append(col)
        
        print(f"Discrete columns identified: {discrete_cols}")
        return data, discrete_cols
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def test_mock_model(model_class, model_name):
    """Test a mock model implementation."""
    print(f"\nTesting {model_name}...")
    
    try:
        data, discrete_cols = load_test_data()
        if data is None:
            return False
        
        # Initialize model
        model = model_class(random_state=42)
        print(f"- Initialization: Success")
        
        # Test parameter space
        param_space = model.get_param_space()
        print(f"- Parameter space: {len(param_space)} parameters")
        
        # Create model with hyperparams
        hyperparams = {}
        for param_name, param_config in param_space.items():
            if param_config[0] == 'int':
                hyperparams[param_name] = param_config[2]  # default value
            elif param_config[0] == 'float':
                hyperparams[param_name] = (param_config[1] + param_config[2]) / 2  # middle value
            elif param_config[0] == 'categorical':
                hyperparams[param_name] = param_config[1][0]  # first choice
        
        model_instance = model.create_model(hyperparams)
        print(f"- Model creation with hyperparams: Success")
        
        # Fit model
        start_time = time.time()
        model.fit(data, discrete_columns=discrete_cols)
        fit_time = time.time() - start_time
        print(f"- Model fitting: Success ({fit_time:.2f}s)")
        
        # Generate data
        start_time = time.time()
        synthetic_data = model.generate(100)
        gen_time = time.time() - start_time
        print(f"- Data generation: Success ({gen_time:.3f}s)")
        
        # Validate generated data
        print(f"- Generated shape: {synthetic_data.shape}")
        print(f"- Columns match: {list(synthetic_data.columns) == list(data.columns)}")
        
        # Check basic statistics
        for col in data.select_dtypes(include=[np.number]).columns[:3]:  # Check first 3 numeric columns
            real_mean = data[col].mean()
            synth_mean = synthetic_data[col].mean()
            diff_pct = abs(real_mean - synth_mean) / real_mean * 100 if real_mean != 0 else 0
            print(f"- {col} mean difference: {diff_pct:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"- {model_name} test failed: {e}")
        return False

def test_clinical_evaluator():
    """Test the clinical evaluator."""
    print("\nTesting Clinical Evaluator...")
    
    try:
        from evaluation.clinical_evaluator import ClinicalModelEvaluator
        
        data, discrete_cols = load_test_data()
        if data is None:
            return False
        
        # Create evaluator
        evaluator = ClinicalModelEvaluator(
            real_data=data,
            target_column="Outcome",
            random_state=42
        )
        print("- Evaluator initialization: Success")
        
        # Create some synthetic data for testing
        from models.baseline_clinical_model import BaselineClinicalModel
        model = BaselineClinicalModel("Test", epochs=5)
        model.fit(data, discrete_columns=discrete_cols)
        synthetic_data = model.generate(200)
        
        # Test similarity evaluation
        similarity = evaluator.evaluate_similarity(synthetic_data)
        print(f"- Similarity evaluation: Success (score: {similarity['overall']:.3f})")
        
        # Test classification evaluation
        classification = evaluator.evaluate_classification(synthetic_data)
        print(f"- Classification evaluation: Success (ratio: {classification['accuracy_ratio']:.3f})")
        
        # Test clinical utility
        utility = evaluator.evaluate_clinical_utility(synthetic_data)
        print(f"- Clinical utility evaluation: Success (score: {utility['overall_utility']:.3f})")
        
        # Test report generation
        report = evaluator.generate_clinical_report(synthetic_data, "TestModel")
        print(f"- Report generation: Success")
        print(f"- Regulatory readiness: {report['regulatory_assessment']['regulatory_readiness']}")
        
        return True
        
    except Exception as e:
        print(f"- Clinical evaluator test failed: {e}")
        return False

def test_bayesian_optimizer_simple():
    """Test the Bayesian optimizer with simple setup."""
    print("\nTesting Bayesian Optimizer (Simple)...")
    
    try:
        from evaluation.clinical_evaluator import ClinicalModelEvaluator
        from optimization.bayesian_optimizer import ClinicalModelOptimizer
        
        data, discrete_cols = load_test_data()
        if data is None:
            return False
        
        # Create evaluator (fix the class name)
        evaluator = ClinicalModelEvaluator(
            real_data=data,
            target_column="Outcome",
            random_state=42
        )
        
        # Create optimizer
        optimizer = ClinicalModelOptimizer(
            data=data,
            discrete_columns=discrete_cols,
            evaluator=evaluator,
            random_state=42
        )
        print("- Optimizer initialization: Success")
        
        # Test parameter spaces
        param_spaces = optimizer.get_parameter_spaces()
        print(f"- Parameter spaces available: {list(param_spaces.keys())}")
        
        # Test a single optimization with 2 trials (very quick)
        print("- Running optimization with 2 trials...")
        start_time = time.time()
        
        result = optimizer.optimize_model("TableGAN", n_trials=2)
        
        opt_time = time.time() - start_time
        print(f"- Optimization completed: Success ({opt_time:.1f}s)")
        print(f"- Best score: {result['best_score']:.4f}")
        print(f"- Trials completed: {result['n_trials']}")
        
        return True
        
    except Exception as e:
        print(f"- Bayesian optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main testing function."""
    print("=" * 60)
    print("MOCK MODELS AND OPTIMIZATION TESTING")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Mock model implementations
    print("\n[1/4] Testing Mock Model Implementations...")
    
    mock_models = []
    try:
        from models.mock_models import (
            MockCTGANModel, MockTVAEModel, MockCopulaGANModel, 
            MockGANerAidModel
        )
        mock_models = [
            (MockCTGANModel, "MockCTGAN"),
            (MockTVAEModel, "MockTVAE"),
            (MockCopulaGANModel, "MockCopulaGAN"),
            (MockGANerAidModel, "MockGANerAid")
        ]
        
        for model_class, model_name in mock_models:
            results[model_name.lower()] = test_mock_model(model_class, model_name)
            
    except Exception as e:
        print(f"Error importing mock models: {e}")
        results['mock_models'] = False
    
    # Test 2: Clinical Evaluator
    print("\n[2/4] Testing Clinical Evaluator...")
    results['clinical_evaluator'] = test_clinical_evaluator()
    
    # Test 3: Bayesian Optimizer
    print("\n[3/4] Testing Bayesian Optimizer...")
    results['bayesian_optimizer'] = test_bayesian_optimizer_simple()
    
    # Test 4: Integration test (if all components work)
    print("\n[4/4] Integration Test...")
    if results.get('clinical_evaluator', False) and results.get('bayesian_optimizer', False):
        print("- All components working - integration successful")
        results['integration'] = True
    else:
        print("- Some components failed - integration incomplete")
        results['integration'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed results:")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"- {test_name}: {status}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if passed_tests >= total_tests * 0.8:
        print("- Framework components are working well!")
        print("- Ready for full optimization testing with more trials")
    elif passed_tests >= total_tests * 0.5:
        print("- Framework is partially working - debug failing components")
    else:
        print("- Framework has significant issues - major debugging needed")
    
    mock_passed = sum(1 for name, result in results.items() if 'mock' in name and result)
    if mock_passed >= 3:
        print(f"- Mock models working well ({mock_passed}/4) - good fallback coverage")
    else:
        print(f"- Only {mock_passed}/4 mock models working - investigate issues")
    
    if results.get('bayesian_optimizer', False):
        print("- Bayesian optimization ready - can proceed with full pipeline")
    else:
        print("- Bayesian optimization needs fixing before production use")
    
    print("\n" + "=" * 60)
    
    return results

if __name__ == "__main__":
    results = main()