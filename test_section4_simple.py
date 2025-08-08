"""
Simple test of Section 4 key functionality
Following claude6.md protocol - test notebook context
"""
import sys
import os

# Set up paths
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

print("TESTING SECTION 4 KEY COMPONENTS")
print("=" * 50)

# Test 1: Basic imports that Section 4 needs
print("\n1. Testing basic imports...")
try:
    import optuna
    print("OK optuna imported successfully")
except ImportError as e:
    print(f"FAIL optuna import failed: {e}")

try:
    from src.models.model_factory import ModelFactory
    print("OK ModelFactory imported successfully")
except ImportError as e:
    print(f"FAIL ModelFactory import failed: {e}")

try:
    from src.evaluation.trts import TRTSEvaluator
    print("OK TRTSEvaluator imported successfully")
except ImportError as e:
    print(f"FAIL TRTSEvaluator import failed: {e}")

# Test 2: Model creation
print("\n2. Testing model creation...")
models_to_test = ['ctgan', 'tvae', 'copulagan', 'ganeraid', 'ctabgan', 'ctabganplus']

working_models = []
broken_models = []

for model_name in models_to_test:
    try:
        from src.models.model_factory import ModelFactory
        model = ModelFactory.create(model_name, random_state=42)
        working_models.append(model_name)
        print(f"OK {model_name}: WORKING")
    except Exception as e:
        broken_models.append((model_name, str(e)))
        print(f"FAIL {model_name}: BROKEN - {e}")

# Test 3: Basic data loading
print("\n3. Testing data loading...")
try:
    import pandas as pd
    data = pd.read_csv('data/breast_cancer_data.csv')
    print(f"OK Data loaded: {data.shape}")
except Exception as e:
    print(f"FAIL Data loading failed: {e}")

# Summary
print("\n" + "=" * 50)
print("SUMMARY:")
print(f"Working models: {len(working_models)} - {working_models}")
print(f"Broken models: {len(broken_models)} - {[name for name, _ in broken_models]}")

if broken_models:
    print("\nBROKEN MODEL DETAILS:")
    for name, error in broken_models:
        print(f"  {name}: {error}")

# Return exit code based on results
exit_code = 0 if len(broken_models) == 0 else 1
print(f"\nEXIT CODE: {exit_code}")
sys.exit(exit_code)