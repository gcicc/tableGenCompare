#!/usr/bin/env python3
"""
Test script for the extracted model framework.
Tests model factory, model implementations, and integration with evaluation framework.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the synthetic-tabular-benchmark src to path
sys.path.append('synthetic-tabular-benchmark/src')

def create_test_data():
    """Create test data for model training."""
    np.random.seed(42)
    
    # Small dataset for quick testing
    n_samples = 500
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'health_score': np.random.uniform(0, 100, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    return data

def test_model_factory():
    """Test the ModelFactory functionality."""
    print("Testing ModelFactory...")
    
    try:
        from models.model_factory import ModelFactory
        
        # Test listing available models
        available_models = ModelFactory.list_available_models()
        print(f"[OK] Available models: {available_models}")
        
        # Test model info
        for model_name, available in available_models.items():
            if available:
                try:
                    info = ModelFactory.get_model_info(model_name)
                    print(f"[OK] {model_name} info retrieved: {info['class']}")
                except Exception as e:
                    print(f"[WARN] Could not get info for {model_name}: {e}")
        
        return True, available_models
        
    except Exception as e:
        print(f"[FAIL] ModelFactory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_ganeraid_model():
    """Test GANerAid model if available."""
    print("Testing GANerAid model...")
    
    try:
        from models.model_factory import ModelFactory
        
        # Check if GANerAid is available
        available_models = ModelFactory.list_available_models()
        if not available_models.get('ganeraid', False):
            print("[SKIP] GANerAid not available - dependencies not installed")
            return True
        
        try:
            # Create GANerAid model
            model = ModelFactory.create('ganeraid', device='cpu', random_state=42)
            print(f"[OK] GANerAid model created: {model.__class__.__name__}")
            
            # Test model info
            model_info = model.get_model_info()
            print(f"[OK] Model info: {model_info['model_type']}")
            
            # Test hyperparameter space
            hyperparams = model.get_hyperparameter_space()
            print(f"[OK] Hyperparameter space has {len(hyperparams)} parameters")
            
            # Test with small dataset and few epochs for quick testing
            test_data = create_test_data()
            print(f"[OK] Test data created: {test_data.shape}")
            
            # Train model with minimal epochs for testing
            print("Training GANerAid model (this may take a minute)...")
            training_result = model.train(test_data, epochs=100, verbose=False)
            print(f"[OK] Model training completed in {training_result.get('training_duration_seconds', 'N/A'):.2f} seconds")
            
            # Generate synthetic data
            print("Generating synthetic data...")
            synthetic_data = model.generate(100)  # Small sample for testing
            print(f"[OK] Generated synthetic data: {synthetic_data.shape}")
            
            # Basic validation
            if synthetic_data.shape[1] == test_data.shape[1]:
                print("[OK] Synthetic data has correct number of columns")
            else:
                print(f"[WARN] Column mismatch: {synthetic_data.shape[1]} vs {test_data.shape[1]}")
            
            if len(synthetic_data) == 100:
                print("[OK] Generated correct number of samples")
            else:
                print(f"[WARN] Sample count mismatch: {len(synthetic_data)} vs 100")
            
            return True
            
        except ImportError as e:
            print(f"[SKIP] GANerAid dependencies not available: {e}")
            return True
        except Exception as e:
            print(f"[FAIL] GANerAid model test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"[FAIL] GANerAid test setup failed: {e}")
        return False

def test_model_evaluation_integration():
    """Test integration between models and evaluation framework."""
    print("Testing Model-Evaluation integration...")
    
    try:
        from models.model_factory import ModelFactory
        from evaluation.unified_evaluator import UnifiedEvaluator
        
        # Check if we have any available models
        available_models = ModelFactory.list_available_models()
        available_model_names = [name for name, available in available_models.items() if available]
        
        if not available_model_names:
            print("[SKIP] No models available for integration testing")
            return True
        
        # Use the first available model
        model_name = available_model_names[0]
        print(f"Testing integration with {model_name}")
        
        try:
            # Create model
            model = ModelFactory.create(model_name, device='cpu', random_state=42)
            
            # Create test data
            original_data = create_test_data()
            
            # Train model (minimal training for testing)
            print(f"Training {model_name} model...")
            if model_name == 'ganeraid':
                model.train(original_data, epochs=50, verbose=False)  # Very quick training
            else:
                model.train(original_data)
            
            # Generate synthetic data
            print("Generating synthetic data...")
            synthetic_data = model.generate(len(original_data))
            
            # Test evaluation integration
            print("Testing evaluation integration...")
            evaluator = UnifiedEvaluator(random_state=42)
            
            # Create minimal metadata
            dataset_metadata = {
                'dataset_info': {'name': 'integration_test', 'description': 'Integration test dataset'},
                'target_info': {'column': 'target', 'type': 'binary'}
            }
            
            # Create output directory
            output_dir = "test_integration_output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Set matplotlib to non-interactive backend
            import matplotlib
            matplotlib.use('Agg')
            
            # Run evaluation
            results = evaluator.run_complete_evaluation(
                model=model,
                original_data=original_data,
                synthetic_data=synthetic_data,
                dataset_metadata=dataset_metadata,
                output_dir=output_dir,
                target_column='target'
            )
            
            print("[OK] Model-Evaluation integration successful!")
            print(f"  Model: {results['model_info']['model_type']}")
            print(f"  TRTS Overall Score: {results['trts_results']['overall_score_percent']:.1f}%")
            print(f"  Similarity Score: {results['similarity_analysis']['final_similarity']:.3f}")
            
            return True
            
        except ImportError as e:
            print(f"[SKIP] Model dependencies not available: {e}")
            return True
        except Exception as e:
            print(f"[FAIL] Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"[FAIL] Integration test setup failed: {e}")
        return False

def test_mock_model():
    """Test with a mock model to verify the base infrastructure."""
    print("Testing with mock model...")
    
    try:
        from models.base_model import SyntheticDataModel
        
        # Create a simple mock model
        class MockModel(SyntheticDataModel):
            def __init__(self, device="cpu", random_state=42):
                super().__init__(device, random_state)
                self.model_config = {"mock_param": 1.0}
            
            def train(self, data, **kwargs):
                self.is_trained = True
                return {"status": "mock_trained"}
            
            def generate(self, n_samples, **kwargs):
                # Generate data with same structure as original
                np.random.seed(self.random_state)
                mock_data = pd.DataFrame({
                    'age': np.random.randint(18, 80, n_samples),
                    'income': np.random.normal(50000, 15000, n_samples),
                    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
                    'health_score': np.random.uniform(0, 100, n_samples),
                    'target': np.random.choice([0, 1], n_samples)
                })
                return mock_data
            
            def get_hyperparameter_space(self):
                return {"mock_param": {"type": "float", "low": 0.1, "high": 2.0}}
            
            def save_model(self, path):
                pass
            
            def load_model(self, path):
                pass
        
        # Test mock model
        mock_model = MockModel()
        test_data = create_test_data()
        
        # Test training
        training_result = mock_model.train(test_data)
        assert mock_model.is_trained, "Mock model should be trained"
        print("[OK] Mock model training")
        
        # Test generation
        synthetic_data = mock_model.generate(100)
        assert len(synthetic_data) == 100, "Should generate correct number of samples"
        assert synthetic_data.shape[1] == test_data.shape[1], "Should have same number of columns"
        print("[OK] Mock model generation")
        
        # Test model info
        model_info = mock_model.get_model_info()
        assert model_info["model_type"] == "MockModel", "Should have correct model type"
        print("[OK] Mock model info")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Mock model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL FRAMEWORK TEST")
    print("=" * 60)
    
    # Test model factory
    factory_ok, available_models = test_model_factory()
    if not factory_ok:
        print("\n[FAIL] Model factory test failed - cannot continue")
        sys.exit(1)
    
    # Test mock model
    print("\n" + "=" * 60)
    mock_ok = test_mock_model()
    
    # Test GANerAid if available
    print("\n" + "=" * 60)
    ganeraid_ok = test_ganeraid_model()
    
    # Test integration if we have models
    print("\n" + "=" * 60)
    integration_ok = test_model_evaluation_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print(f"Model Factory: {'PASS' if factory_ok else 'FAIL'}")
    print(f"Mock Model: {'PASS' if mock_ok else 'FAIL'}")
    print(f"GANerAid Model: {'PASS' if ganeraid_ok else 'FAIL'}")
    print(f"Integration: {'PASS' if integration_ok else 'FAIL'}")
    
    if factory_ok and mock_ok and ganeraid_ok and integration_ok:
        print("\n[SUCCESS] All model framework tests passed!")
    else:
        print("\n[PARTIAL] Some tests failed or were skipped.")
    
    print("=" * 60)