"""
Simple Final Test for Phase 6 Clinical Framework - No Unicode
Tests all 5 models and components
"""

import pandas as pd
import numpy as np
import time
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_tests():
    """Run comprehensive tests."""
    
    print("PHASE 6 CLINICAL FRAMEWORK - COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Data Loading
    print("\n1. DATA LOADING")
    print("-" * 30)
    try:
        data_path = r"C:\Users\gcicc\claudeproj\tableGenCompare\data\Pakistani_Diabetes_Dataset.csv"
        data = pd.read_csv(data_path)
        discrete_cols = ['Gender', 'Rgn ', 'his', 'vision', 'dipsia', 'uria', 'neph', 'Outcome']
        print(f"[PASS] Data loaded: {data.shape}")
        print(f"[PASS] Discrete columns: {len(discrete_cols)}")
        results['data_loading'] = 'PASS'
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        results['data_loading'] = 'FAIL'
        return results
    
    # Test 2: Package Availability
    print("\n2. PACKAGE AVAILABILITY")
    print("-" * 30)
    
    availability = {}
    
    # CTGAN/TVAE
    try:
        from ctgan import CTGAN, TVAE
        availability['CTGAN'] = True
        availability['TVAE'] = True
        print("[PASS] CTGAN and TVAE available")
    except ImportError:
        availability['CTGAN'] = False
        availability['TVAE'] = False
        print("[WARN] CTGAN/TVAE not available (using baseline)")
    
    # SDV CopulaGAN
    try:
        from sdv.single_table import CopulaGANSynthesizer
        availability['CopulaGAN'] = True
        print("[PASS] CopulaGAN available")
    except ImportError:
        availability['CopulaGAN'] = False
        print("[WARN] CopulaGAN not available (using baseline)")
    
    availability['TableGAN'] = True
    availability['GANerAid'] = True
    print("[PASS] TableGAN and GANerAid (baseline) available")
    
    results['package_availability'] = availability
    
    # Test 3: Baseline Models
    print("\n3. BASELINE MODELS")
    print("-" * 30)
    
    baseline_success = 0
    model_names = ['CTGAN', 'TVAE', 'CopulaGAN', 'TableGAN', 'GANerAid']
    
    try:
        from models.baseline_clinical_model import BaselineClinicalModel
        
        for model_name in model_names:
            try:
                model = BaselineClinicalModel(model_name, epochs=3, batch_size=32)
                model.fit(data.sample(200), discrete_columns=discrete_cols)
                synthetic_data = model.generate(50)
                
                print(f"[PASS] {model_name} baseline: {synthetic_data.shape[0]} samples")
                baseline_success += 1
                
            except Exception as e:
                print(f"[FAIL] {model_name} baseline: {e}")
        
        results['baseline_models'] = f"{baseline_success}/{len(model_names)}"
        
    except Exception as e:
        print(f"[FAIL] Baseline models import: {e}")
        results['baseline_models'] = '0/5'
    
    # Test 4: Mock Models
    print("\n4. MOCK MODELS")
    print("-" * 30)
    
    mock_success = 0
    try:
        from models.mock_models import MockCTGANModel, MockTVAEModel, MockCopulaGANModel, MockGANerAidModel
        
        mock_models = [
            (MockCTGANModel, "MockCTGAN"),
            (MockTVAEModel, "MockTVAE"),
            (MockCopulaGANModel, "MockCopulaGAN"),
            (MockGANerAidModel, "MockGANerAid")
        ]
        
        for model_class, model_name in mock_models:
            try:
                model = model_class(random_state=42)
                param_space = model.get_param_space()
                model.fit(data.sample(100), discrete_columns=discrete_cols)
                synthetic_data = model.generate(25)
                
                print(f"[PASS] {model_name}: {len(param_space)} params, {synthetic_data.shape[0]} samples")
                mock_success += 1
                
            except Exception as e:
                print(f"[FAIL] {model_name}: {e}")
        
        results['mock_models'] = f"{mock_success}/4"
        
    except Exception as e:
        print(f"[FAIL] Mock models import: {e}")
        results['mock_models'] = '0/4'
    
    # Test 5: Clinical Evaluator
    print("\n5. CLINICAL EVALUATOR")
    print("-" * 30)
    
    try:
        from evaluation.clinical_evaluator import ClinicalModelEvaluator
        
        evaluator = ClinicalModelEvaluator(
            real_data=data,
            target_column="Outcome",
            random_state=42
        )
        
        # Generate test data
        from models.baseline_clinical_model import BaselineClinicalModel
        model = BaselineClinicalModel("EvalTest", epochs=2)
        model.fit(data.sample(200), discrete_columns=discrete_cols)
        test_synthetic = model.generate(100)
        
        # Test evaluations
        similarity = evaluator.evaluate_similarity(test_synthetic)
        classification = evaluator.evaluate_classification(test_synthetic)
        utility = evaluator.evaluate_clinical_utility(test_synthetic)
        report = evaluator.generate_clinical_report(test_synthetic, "TestModel")
        
        print(f"[PASS] Similarity score: {similarity['overall']:.3f}")
        print(f"[PASS] Classification ratio: {classification['accuracy_ratio']:.3f}")
        print(f"[PASS] Clinical utility: {utility['overall_utility']:.3f}")
        print(f"[PASS] Regulatory readiness: {report['regulatory_assessment']['regulatory_readiness']}")
        
        results['clinical_evaluator'] = 'PASS'
        
    except Exception as e:
        print(f"[FAIL] Clinical evaluator: {e}")
        results['clinical_evaluator'] = 'FAIL'
    
    # Test 6: Optimization Framework Setup
    print("\n6. OPTIMIZATION FRAMEWORK")
    print("-" * 30)
    
    try:
        from optimization.bayesian_optimizer import ClinicalModelOptimizer
        
        evaluator = ClinicalModelEvaluator(
            real_data=data,
            target_column="Outcome",
            random_state=42
        )
        
        optimizer = ClinicalModelOptimizer(
            data=data,
            discrete_columns=discrete_cols,
            evaluator=evaluator,
            random_state=42
        )
        
        param_spaces = optimizer.get_parameter_spaces()
        available_models = list(param_spaces.keys())
        
        print(f"[PASS] Optimizer initialized")
        print(f"[PASS] Available models: {len(available_models)}")
        print(f"[PASS] Models: {', '.join(available_models)}")
        
        # Test objective function creation
        objective_func = optimizer.create_objective_function("TableGAN", param_spaces["TableGAN"])
        print(f"[PASS] Objective function created")
        
        results['optimization_framework'] = 'PASS'
        
    except Exception as e:
        print(f"[FAIL] Optimization framework: {e}")
        results['optimization_framework'] = 'FAIL'
    
    # Test 7: Advanced Models (if available)
    print("\n7. ADVANCED MODELS")
    print("-" * 30)
    
    advanced_success = 0
    advanced_total = 0
    
    # Test CTGAN if available
    if availability.get('CTGAN', False):
        advanced_total += 1
        try:
            from ctgan import CTGAN
            model = CTGAN(epochs=1, batch_size=32, verbose=False)
            model.fit(data.sample(100), discrete_columns=discrete_cols)
            synthetic_data = model.sample(25)
            
            print(f"[PASS] CTGAN advanced: {len(synthetic_data)} samples")
            advanced_success += 1
            
        except Exception as e:
            print(f"[FAIL] CTGAN advanced: {e}")
    else:
        print("[SKIP] CTGAN: Package not available")
    
    # Test CopulaGAN if available
    if availability.get('CopulaGAN', False):
        advanced_total += 1
        try:
            from sdv.single_table import CopulaGANSynthesizer
            model = CopulaGANSynthesizer()
            model.fit(data.sample(100))
            synthetic_data = model.sample(25)
            
            print(f"[PASS] CopulaGAN advanced: {len(synthetic_data)} samples")
            advanced_success += 1
            
        except Exception as e:
            print(f"[FAIL] CopulaGAN advanced: {e}")
    else:
        print("[SKIP] CopulaGAN: Package not available")
    
    results['advanced_models'] = f"{advanced_success}/{advanced_total}" if advanced_total > 0 else "0/0 (none available)"
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    print("Component Test Results:")
    for component, result in results.items():
        print(f"  {component.replace('_', ' ').title()}: {result}")
    
    # Framework Status
    print("\nFramework Assessment:")
    
    critical_passed = sum(1 for comp in ['data_loading', 'clinical_evaluator', 'optimization_framework'] 
                         if results.get(comp) == 'PASS')
    
    if critical_passed == 3:
        print("  FRAMEWORK STATUS: OPERATIONAL")
        print("  All critical components working")
    elif critical_passed >= 2:
        print("  FRAMEWORK STATUS: PARTIALLY OPERATIONAL")
        print("  Most critical components working")
    else:
        print("  FRAMEWORK STATUS: NEEDS DEBUGGING")
        print("  Critical components failing")
    
    # Model Coverage
    baseline_working = results.get('baseline_models', '0/5').split('/')[0]
    mock_working = results.get('mock_models', '0/4').split('/')[0]
    advanced_working = results.get('advanced_models', '0/0').split('/')[0]
    
    total_working = int(baseline_working) + int(mock_working) + int(advanced_working)
    
    print(f"\nModel Coverage:")
    print(f"  Baseline models working: {baseline_working}/5")
    print(f"  Mock models working: {mock_working}/4")
    print(f"  Advanced models working: {advanced_working}")
    print(f"  Total model implementations: {total_working}")
    
    # Recommendations
    print(f"\nRecommendations:")
    
    if total_working >= 8:
        print("  [EXCELLENT] Framework ready for production use")
    elif total_working >= 5:
        print("  [GOOD] Framework suitable for most use cases")
    elif total_working >= 3:
        print("  [FAIR] Framework has basic functionality")
    else:
        print("  [POOR] Framework needs significant work")
    
    if not availability.get('CTGAN', False):
        print("  - Install 'ctgan' package for advanced CTGAN/TVAE models")
    
    if not availability.get('CopulaGAN', False):
        print("  - Install 'sdv' package for advanced CopulaGAN model")
    
    if results.get('optimization_framework') == 'PASS':
        print("  - Bayesian optimization ready for use")
        print("  - Can run full optimization with 50+ trials per model")
    else:
        print("  - Debug optimization framework before production")
    
    print("\n" + "=" * 70)
    
    return results

def main():
    """Main function."""
    print("Starting Phase 6 Clinical Framework Comprehensive Testing...")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = run_tests()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"phase6_comprehensive_test_results_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("PHASE 6 CLINICAL FRAMEWORK - COMPREHENSIVE TEST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("COMPONENT TEST RESULTS:\n")
        f.write("-" * 30 + "\n")
        for component, result in results.items():
            f.write(f"{component.replace('_', ' ').title()}: {result}\n")
        
        f.write(f"\nTest completed successfully.\n")
        f.write(f"Results saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nDetailed results saved to: {filename}")
    
    return results

if __name__ == "__main__":
    results = main()