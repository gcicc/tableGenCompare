"""
Quick Optimization Test for Phase 6 Clinical Framework
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("Quick Optimization Test")
    print("=" * 40)
    
    # Load data
    data_path = r"C:\Users\gcicc\claudeproj\tableGenCompare\data\Pakistani_Diabetes_Dataset.csv"
    data = pd.read_csv(data_path)
    print(f"Data loaded: {data.shape}")
    
    # Test 1: Import check
    print("\n1. Testing imports...")
    try:
        from evaluation.clinical_evaluator import ClinicalModelEvaluator
        print("   - ClinicalModelEvaluator: OK")
    except Exception as e:
        print(f"   - ClinicalModelEvaluator: FAIL - {e}")
        return
    
    try:
        from optimization.bayesian_optimizer import ClinicalModelOptimizer
        print("   - ClinicalModelOptimizer: OK")
    except Exception as e:
        print(f"   - ClinicalModelOptimizer: FAIL - {e}")
        return
    
    # Test 2: Component initialization
    print("\n2. Testing initialization...")
    try:
        discrete_cols = ['Gender', 'Outcome']  # Simplified
        
        evaluator = ClinicalModelEvaluator(
            real_data=data,
            target_column="Outcome",
            random_state=42
        )
        print("   - Evaluator: OK")
        
        optimizer = ClinicalModelOptimizer(
            data=data,
            discrete_columns=discrete_cols,
            evaluator=evaluator,
            random_state=42
        )
        print("   - Optimizer: OK")
        
    except Exception as e:
        print(f"   - Initialization: FAIL - {e}")
        return
    
    # Test 3: Quick optimization (1 trial)
    print("\n3. Testing optimization (1 trial)...")
    try:
        result = optimizer.optimize_model("TableGAN", n_trials=1, timeout=30)
        print(f"   - Optimization: OK (score: {result['best_score']:.4f})")
        
    except Exception as e:
        print(f"   - Optimization: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n4. Testing baseline model directly...")
    try:
        from models.baseline_clinical_model import BaselineClinicalModel
        
        model = BaselineClinicalModel("QuickTest", epochs=5, batch_size=32)
        model.fit(data, discrete_columns=discrete_cols)
        synthetic_data = model.generate(50)
        
        print(f"   - Baseline model: OK (generated {len(synthetic_data)} samples)")
        
        # Quick evaluation
        similarity = evaluator.evaluate_similarity(synthetic_data)
        print(f"   - Similarity score: {similarity['overall']:.3f}")
        
    except Exception as e:
        print(f"   - Baseline model: FAIL - {e}")
        return
    
    print("\nALL TESTS PASSED!")
    print("Framework is working correctly.")

if __name__ == "__main__":
    main()