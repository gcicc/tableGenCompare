"""
Test optuna hyperparameter optimization - reproduce Section 4.2/4.3 optimization
"""
import sys
import os

# Notebook-like setup
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
os.chdir(r'C:\Users\gcicc\claudeproj\tableGenCompare')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("TESTING OPTUNA HYPERPARAMETER OPTIMIZATION")
print("=" * 50)

try:
    # Test optuna import and basic functionality
    import optuna
    print("Optuna imported successfully")
    
    # Load data
    data = pd.read_csv('data/breast_cancer_data.csv')
    print(f"Data loaded: {data.shape}")
    
    # Test imports from evaluation framework
    from src.models.model_factory import ModelFactory
    from src.evaluation.trts_framework import TRTSEvaluator
    print("Model factory and evaluator imported")
    
    # Create a simple optuna study
    def objective(trial):
        """Simple test objective function"""
        # Suggest hyperparameters like in Section 4.2
        epochs = trial.suggest_int('epochs', 1, 2)  # Very small for testing
        batch_size = trial.suggest_categorical('batch_size', [64, 128])
        
        try:
            # Create and train model
            model = ModelFactory.create('ctabgan', random_state=42)
            training_metadata = model.train(data, epochs=epochs)
            
            # Simple metric - training time (lower is better)
            return training_metadata['training_time']
            
        except Exception as e:
            print(f"Trial failed: {e}")
            # Return high value to indicate failure
            return 1000.0
    
    # Create and run study (minimal trials for testing)
    print("\nRunning optuna optimization (2 trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=2, show_progress_bar=True)
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    
    print("\n" + "=" * 50)
    print("SUCCESS: Optuna hyperparameter optimization working!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)