#!/usr/bin/env python3
"""
Test script to verify all model imports work correctly.
This helps diagnose import issues before running the full notebook.
"""

print("TESTING MODEL IMPORTS")
print("="*40)

MODEL_STATUS = {}

# Test CTGAN
try:
    from src.models.implementations.ctgan_model import CTGANModel
    MODEL_STATUS['CTGAN'] = True
    print("OK CTGAN available")
except ImportError as e:
    MODEL_STATUS['CTGAN'] = False
    print(f"WARN CTGAN not available: {e}")

# Test TVAE
try:
    from src.models.implementations.tvae_model import TVAEModel
    MODEL_STATUS['TVAE'] = True
    print("OK TVAE available")
except ImportError as e:
    MODEL_STATUS['TVAE'] = False
    print(f"WARN TVAE not available: {e}")

# Test CopulaGAN
try:
    from src.models.implementations.copulagan_model import CopulaGANModel
    MODEL_STATUS['CopulaGAN'] = True
    print("OK CopulaGAN available")
except ImportError as e:
    MODEL_STATUS['CopulaGAN'] = False
    print(f"WARN CopulaGAN not available: {e}")

# Test GANerAid
try:
    from src.models.implementations.ganeraid_model import GANerAidModel
    MODEL_STATUS['GANerAid'] = True
    print("OK GANerAid available")
except ImportError as e:
    MODEL_STATUS['GANerAid'] = False
    print(f"WARN GANerAid not available: {e}")

# Test TableGAN
try:
    from src.models.implementations.tablegan_model import TableGANModel
    MODEL_STATUS['TableGAN'] = True
    print("OK TableGAN available")
except ImportError as e:
    MODEL_STATUS['TableGAN'] = False
    print(f"WARN TableGAN not available: {e}")

# Test framework components
print(f"\nTESTING FRAMEWORK COMPONENTS")
print("="*40)

try:
    from src.models.model_factory import ModelFactory
    print("OK ModelFactory available")
except ImportError as e:
    print(f"WARN ModelFactory not available: {e}")

try:
    from src.evaluation.unified_evaluator import UnifiedEvaluator
    print("OK UnifiedEvaluator available")
except ImportError as e:
    print(f"WARN UnifiedEvaluator not available: {e}")

try:
    from src.optimization.optuna_optimizer import OptunaOptimizer
    print("OK OptunaOptimizer available")
except ImportError as e:
    print(f"WARN OptunaOptimizer not available: {e}")

try:
    import optuna
    print("OK Optuna available")
except ImportError as e:
    print(f"WARN Optuna not available: {e}")

# Summary
available_models = [model for model, status in MODEL_STATUS.items() if status]
unavailable_models = [model for model, status in MODEL_STATUS.items() if not status]

print(f"\nSUMMARY")
print("="*20)
print(f"Available models ({len(available_models)}): {', '.join(available_models)}")
if unavailable_models:
    print(f"Unavailable models ({len(unavailable_models)}): {', '.join(unavailable_models)}")

print(f"\nAll imports tested successfully!")
if len(available_models) >= 3:
    print(f"Ready to run multi-model notebook with {len(available_models)} models")
else:
    print(f"Limited models available - some notebook features may not work")