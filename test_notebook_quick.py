# Quick test version of the notebook with reduced parameters for validation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
from pathlib import Path

# Import model classes
try:
    from src.models.implementations.ctgan_model import CTGANModel
    from src.models.implementations.tvae_model import TVAEModel  
    from src.models.implementations.copulagan_model import CopulaGANModel
    from src.models.implementations.ganeraid_model import GANerAidModel
    from src.models.implementations.tablegan_model import TableGANModel
    print("SUCCESS: All model imports successful")
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    exit(1)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
np.random.seed(42)

print("QUICK VALIDATION TEST")
print("="*40)

# Load data
DATA_FILE = "data/Breast_cancer_data.csv"
TARGET_COLUMN = "diagnosis"

try:
    data = pd.read_csv(DATA_FILE)
    print(f"SUCCESS: Data loaded: {data.shape}")
except Exception as e:
    print(f"ERROR: Data loading failed: {e}")
    exit(1)

# Quick test parameters (reduced for speed)
TEST_EPOCHS = 10  # Very reduced for quick test
TEST_SAMPLES = 100  # Reduced sample size
models_to_test = ['CTGAN', 'TVAE']  # Test only 2 models

results = {}

print(f"\nTesting {len(models_to_test)} models with {TEST_EPOCHS} epochs...")

for model_name in models_to_test:
    print(f"\nTesting {model_name}...")
    
    try:
        # Initialize model
        if model_name == 'CTGAN':
            model = CTGANModel()
            params = {'epochs': TEST_EPOCHS, 'batch_size': 100}
        elif model_name == 'TVAE':
            model = TVAEModel()
            params = {'epochs': TEST_EPOCHS, 'batch_size': 100}
        
        # Train model
        start_time = time.time()
        model.train(data, **params)
        train_time = time.time() - start_time
        
        # Generate synthetic data
        start_gen = time.time()
        synthetic_data = model.generate(TEST_SAMPLES)
        gen_time = time.time() - start_gen
        
        # Basic evaluation
        X_real = data.drop(columns=[TARGET_COLUMN])
        y_real = data[TARGET_COLUMN]
        X_synth = synthetic_data.drop(columns=[TARGET_COLUMN])
        y_synth = synthetic_data[TARGET_COLUMN]
        
        # Train-Synthetic, Test-Real evaluation
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real, test_size=0.3, random_state=42
        )
        X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
            X_synth, y_synth, test_size=0.3, random_state=42
        )
        
        # TSTR evaluation
        clf = DecisionTreeClassifier(random_state=42, max_depth=5)
        clf.fit(X_synth_train, y_synth_train)
        tstr_score = clf.score(X_real_test, y_real_test)
        
        # TRTR baseline
        clf.fit(X_real_train, y_real_train)
        trtr_score = clf.score(X_real_test, y_real_test)
        
        utility_score = tstr_score / trtr_score if trtr_score > 0 else 0
        
        results[model_name] = {
            'status': 'success',
            'train_time': train_time,
            'gen_time': gen_time,
            'utility_score': utility_score,
            'tstr_score': tstr_score,
            'trtr_score': trtr_score,
            'generated_samples': len(synthetic_data)
        }
        
        print(f"   SUCCESS - Utility: {utility_score:.4f}, Time: {train_time:.1f}s")
        
    except Exception as e:
        results[model_name] = {
            'status': 'failed',
            'error': str(e)
        }
        print(f"   ERROR - Failed: {str(e)[:60]}...")

print(f"\nQUICK TEST RESULTS")
print("="*30)

successful = []
failed = []

for model_name, result in results.items():
    if result['status'] == 'success':
        successful.append(model_name)
        print(f"SUCCESS {model_name}: {result['utility_score']:.4f} utility, {result['train_time']:.1f}s train")
    else:
        failed.append(model_name)
        print(f"ERROR {model_name}: {result['error'][:50]}...")

print(f"\nSUMMARY:")
print(f"   - Successful: {len(successful)}/{len(models_to_test)} models")
print(f"   - Framework validation: {'PASSED' if len(successful) > 0 else 'FAILED'}")

if successful:
    best_model = max(successful, key=lambda x: results[x]['utility_score'])
    print(f"   - Best performing: {best_model} ({results[best_model]['utility_score']:.4f})")