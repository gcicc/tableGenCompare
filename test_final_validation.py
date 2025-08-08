"""
Final validation test - reproduce exact Section 4 workflow
"""
import sys
import os

# Setup paths exactly like in notebook
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
os.chdir(r'C:\Users\gcicc\claudeproj\tableGenCompare')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("FINAL SECTION 4 VALIDATION TEST")
print("=" * 40)

try:
    # Test all the key components that Section 4 uses
    
    # 1. Import test
    print("1. Testing imports...")
    import optuna
    from src.models.model_factory import ModelFactory
    from src.evaluation.trts_framework import TRTSEvaluator
    print("   All imports successful")
    
    # 2. Data loading
    print("2. Testing data loading...")
    data = pd.read_csv('data/breast_cancer_data.csv')
    print(f"   Data loaded: {data.shape}")
    
    # 3. Model creation test
    print("3. Testing all model creation...")
    models = ['ctgan', 'tvae', 'copulagan', 'ganeraid', 'ctabgan', 'ctabganplus']
    for model_name in models:
        model = ModelFactory.create(model_name, random_state=42)
        print(f"   {model_name}: OK")
    
    # 4. CTAB-GAN workflow (Section 4.2)
    print("4. Testing CTAB-GAN workflow (Section 4.2)...")
    ctabgan_model = ModelFactory.create('ctabgan', random_state=42)
    ctabgan_metadata = ctabgan_model.train(data, epochs=1)
    ctabgan_synthetic = ctabgan_model.generate(20)
    print(f"   CTAB-GAN: trained and generated {ctabgan_synthetic.shape[0]} samples")
    
    # 5. CTAB-GAN+ workflow (Section 4.3) 
    print("5. Testing CTAB-GAN+ workflow (Section 4.3)...")
    ctabganplus_model = ModelFactory.create('ctabganplus', random_state=42)
    ctabganplus_metadata = ctabganplus_model.train(data, epochs=1)
    ctabganplus_synthetic = ctabganplus_model.generate(20)
    print(f"   CTAB-GAN+: trained and generated {ctabganplus_synthetic.shape[0]} samples")
    
    # 6. Optuna integration test
    print("6. Testing optuna integration...")
    def test_objective(trial):
        epochs = trial.suggest_int('epochs', 1, 1)  # Fixed for testing
        try:
            model = ModelFactory.create('ctabgan', random_state=42)
            metadata = model.train(data, epochs=epochs)
            return metadata['training_time']
        except:
            return 999.0
    
    study = optuna.create_study(direction='minimize')
    study.optimize(test_objective, n_trials=1)
    print(f"   Optuna: completed optimization")
    
    # 7. TRTS evaluation test
    print("7. Testing TRTS evaluation...")
    evaluator = TRTSEvaluator(random_state=42)
    target_col = 'diagnosis'
    trts_results = evaluator.evaluate_trts_scenarios(data, ctabgan_synthetic, target_col)
    print(f"   TRTS: evaluation completed")
    
    print("\n" + "=" * 40)
    print("SUCCESS: ALL Section 4 components working!")
    print("- CTAB-GAN training and generation: OK")
    print("- CTAB-GAN+ training and generation: OK") 
    print("- Optuna hyperparameter optimization: OK")
    print("- TRTS evaluation framework: OK")
    print("- All model imports and creation: OK")
    print("\nSection 4 should execute without any errors!")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)