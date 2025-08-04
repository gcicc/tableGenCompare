#!/usr/bin/env python3
"""
Simple TableGAN test without Unicode characters.
"""

import sys
import pandas as pd
import numpy as np

# Add the src directory to path
sys.path.append('src')

def test_tablegan_basic():
    """Basic TableGAN functionality test."""
    print("Testing TableGAN basic functionality...")
    
    try:
        from models.model_factory import ModelFactory
        
        # Check if TableGAN is available
        available_models = ModelFactory.list_available_models()
        print(f"Available models: {available_models}")
        
        if not available_models.get('tablegan', False):
            print("TableGAN not available - PyTorch not installed")
            print("To install: pip install torch")
            return False
        
        # Create simple test data
        np.random.seed(42)
        test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        print(f"Test data shape: {test_data.shape}")
        
        # Create TableGAN model
        model = ModelFactory.create('tablegan', device='cpu', random_state=42)
        print(f"TableGAN model created: {model.__class__.__name__}")
        
        # Get model info
        info = model.get_model_info()
        print(f"Model type: {info['model_type']}")
        print(f"Supports categorical: {info['supports_categorical']}")
        print(f"Supports mixed types: {info['supports_mixed_types']}")
        
        # Get hyperparameters
        hyperparams = model.get_hyperparameter_space()
        print(f"Hyperparameters: {len(hyperparams)} parameters")
        print(f"Parameters: {list(hyperparams.keys())}")
        
        # Train model (very small for testing)
        print("Training TableGAN (reduced epochs for testing)...")
        training_result = model.train(
            test_data,
            epochs=10,  # Very small for quick test
            batch_size=32,
            learning_rate=1e-3,
            verbose=False
        )
        
        print(f"Training completed in {training_result['training_duration_seconds']:.2f} seconds")
        print(f"Epochs completed: {training_result['epochs_completed']}")
        print(f"Model size: {training_result['model_size_mb']:.2f} MB")
        
        # Generate synthetic data
        print("Generating synthetic data...")
        synthetic_data = model.generate(50)
        print(f"Generated data shape: {synthetic_data.shape}")
        
        # Basic validation
        print("Validation results:")
        print(f"  Original columns: {test_data.shape[1]}")
        print(f"  Synthetic columns: {synthetic_data.shape[1]}")
        print(f"  Generated samples: {len(synthetic_data)}")
        
        # Check data ranges
        for col in test_data.columns:
            if col in synthetic_data.columns:
                orig_min, orig_max = test_data[col].min(), test_data[col].max()
                synth_min, synth_max = synthetic_data[col].min(), synthetic_data[col].max()
                print(f"  {col}: Original [{orig_min:.2f}, {orig_max:.2f}] -> Synthetic [{synth_min:.2f}, {synth_max:.2f}]")
        
        print("TableGAN basic test PASSED")
        return True
        
    except ImportError as e:
        print(f"Import error (dependencies not available): {e}")
        return False
    except Exception as e:
        print(f"TableGAN test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TABLEGAN SIMPLE TEST")
    print("=" * 50)
    
    success = test_tablegan_basic()
    
    print("=" * 50)
    if success:
        print("TableGAN implementation test PASSED!")
        print("TableGAN is ready for use in the framework.")
    else:
        print("TableGAN implementation test FAILED or SKIPPED.")
    print("=" * 50)
    
    sys.exit(0 if success else 1)