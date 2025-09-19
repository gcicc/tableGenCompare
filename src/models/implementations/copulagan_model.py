#!/usr/bin/env python3
"""
CopulaGAN Model Implementation.

This module provides a wrapper for the CopulaGAN model from the SDV library,
integrating it with our unified synthetic data generation framework.

CopulaGAN combines copula modeling with adversarial training to preserve
marginal distributions and dependencies in tabular data.
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
import importlib.util

# Try to import SDV CopulaGAN
try:
    from sdv.single_table import CopulaGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    COPULAGAN_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("SDV CopulaGAN successfully imported")
except ImportError as e:
    COPULAGAN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"SDV not available: {e}")
    
    # Create dummy classes for type hints
    class CopulaGANSynthesizer:
        pass
    
    class SingleTableMetadata:
        pass

from ..base_model import SyntheticDataModel

# Import data preprocessing functions from setup.py
try:
    import sys
    import os

    # Multiple attempts to find and import setup.py functions
    PREPROCESSING_AVAILABLE = False

    # Method 1: Try direct import from project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from setup import clean_and_preprocess_data, get_categorical_columns_for_models
        PREPROCESSING_AVAILABLE = True
        logger.info("Data preprocessing functions imported from setup.py (method 1)")
    except ImportError:
        # Method 2: Try importing from current working directory
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            from setup import clean_and_preprocess_data, get_categorical_columns_for_models
            PREPROCESSING_AVAILABLE = True
            logger.info("Data preprocessing functions imported from setup.py (method 2)")
        except ImportError:
            # Method 3: Try relative import patterns
            possible_paths = [
                os.path.join(project_root, 'setup.py'),
                os.path.join(cwd, 'setup.py'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'setup.py')
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"Found setup.py at: {path}")
                    spec = importlib.util.spec_from_file_location("setup", path)
                    setup_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(setup_module)

                    clean_and_preprocess_data = setup_module.clean_and_preprocess_data
                    get_categorical_columns_for_models = setup_module.get_categorical_columns_for_models
                    PREPROCESSING_AVAILABLE = True
                    logger.info(f"Data preprocessing functions imported from setup.py (method 3): {path}")
                    break

    if not PREPROCESSING_AVAILABLE:
        raise ImportError("Could not import setup.py functions after all attempts")

except Exception as e:
    import importlib.util
    PREPROCESSING_AVAILABLE = False
    logger.warning(f"Could not import preprocessing functions: {e}")
    logger.warning(f"Current working directory: {os.getcwd()}")
    logger.warning(f"File location: {__file__}")
    logger.warning(f"Project root attempted: {project_root}")

    # Create enhanced fallback functions that provide basic preprocessing
    def clean_and_preprocess_data(data, categorical_columns=None):
        """Fallback preprocessing function"""
        try:
            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import LabelEncoder

            cleaned_data = data.copy()
            encoders_dict = {}

            # Handle missing values
            for col in cleaned_data.columns:
                if cleaned_data[col].isnull().any():
                    if cleaned_data[col].dtype == 'object':
                        cleaned_data[col].fillna('Unknown', inplace=True)
                    else:
                        cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)

            # Handle categorical columns
            categorical_cols = categorical_columns or []
            if not categorical_cols:
                categorical_cols = cleaned_data.select_dtypes(include=['object']).columns.tolist()

            for col in categorical_cols:
                if col in cleaned_data.columns and cleaned_data[col].dtype == 'object':
                    le = LabelEncoder()
                    cleaned_data[col] = le.fit_transform(cleaned_data[col].astype(str))
                    encoders_dict[col] = le

            # Handle infinity values
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if np.isinf(cleaned_data[col]).any():
                    cleaned_data[col] = cleaned_data[col].replace([np.inf, -np.inf], 0)

            return cleaned_data, categorical_cols, encoders_dict

        except Exception as fallback_error:
            logger.error(f"Fallback preprocessing failed: {fallback_error}")
            return data, categorical_columns or [], {}

    def get_categorical_columns_for_models():
        """Fallback categorical detection function"""
        return []

    logger.info("Using fallback preprocessing functions")

class CopulaGANModel(SyntheticDataModel):
    """
    CopulaGAN model wrapper for synthetic tabular data generation.
    
    CopulaGAN is a GAN-based model that:
    1. Uses copula modeling to preserve marginal distributions
    2. Applies adversarial training on the copula space
    3. Provides excellent preservation of statistical properties
    4. Handles mixed data types effectively
    
    Key advantages:
    - Strong statistical fidelity
    - Robust handling of complex dependencies
    - Good performance on diverse datasets
    - Preserves marginal distributions well
    """
    
    def __init__(self, device: str = "cpu", random_state: int = 42):
        """
        Initialize CopulaGAN model.
        
        Args:
            device: Computing device (CopulaGAN uses CPU by default)
            random_state: Random seed for reproducibility
        """
        if not COPULAGAN_AVAILABLE:
            raise ImportError("CopulaGAN is not available. Please install it with: pip install sdv")
        
        super().__init__(device, random_state)
        self._copulagan_model = None
        self._metadata = None
        self._training_history = None
        
        logger.info("CopulaGAN model initialized")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        return {
            "model_type": "CopulaGAN",
            "description": "Copula-based GAN for tabular data synthesis with strong statistical fidelity",
            "paper": "Modeling Tabular Data using Conditional GAN (Xu et al., 2019)",
            "supports_categorical": True,
            "supports_mixed_types": True,
            "supports_conditional": False,  # Not directly supported in this implementation
            "handles_missing": True,
            "preserves_distributions": True,
            "training_stability": "high",
            "generation_speed": "medium",
            "memory_usage": "medium",
            "best_for": ["mixed_data", "statistical_fidelity", "copula_modeling"],
            "dependencies": ["sdv>=1.0.0"]
        }
    
    def get_hyperparameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the enhanced hyperparameter search space for CopulaGAN optimization.
        Production-ready hyperparameter space designed for diverse tabular datasets.
        
        Returns:
            Dictionary defining comprehensive hyperparameter space for CopulaGAN
        """
        return {
            "epochs": {
                "type": "int",
                "low": 100,
                "high": 800,
                "step": 50,
                "default": 300,
                "description": "Training epochs - 300 optimal for copula modeling convergence"
            },
            "batch_size": {
                "type": "categorical",
                "choices": [32, 64, 128, 256, 500, 1000, 2000],
                "default": 500,
                "description": "Batch size - larger batches (500+) improve GAN stability and copula learning"
            },
            "generator_lr": {
                "type": "float",
                "low": 5e-6,
                "high": 5e-3,
                "log": True,
                "default": 2e-4,
                "description": "Generator learning rate - 2e-4 optimal for copula-based adversarial training"
            },
            "discriminator_lr": {
                "type": "float",
                "low": 5e-6,
                "high": 5e-3,
                "log": True,
                "default": 2e-4,
                "description": "Discriminator learning rate - balanced with generator for stable training"
            },
            "generator_dim": {
                "type": "categorical",
                "choices": [
                    (128, 128),          # Small datasets (<1K samples, <20 features)
                    (256, 256),          # Medium datasets (1K-10K samples, 20-50 features)
                    (512, 512),          # Large datasets (10K-100K samples, 50+ features)
                    (256, 512),          # Asymmetric for complex copula modeling
                    (512, 256),          # Bottleneck for regularization in copula space
                    (128, 256, 128),     # Deep generator for small-medium datasets
                    (256, 512, 256),     # Deep generator for medium-large datasets
                    (512, 1024, 512),    # Deep generator for large/complex datasets
                    (128, 256, 512, 256), # Very deep for complex dependencies
                    (256, 512, 1024, 512) # Very deep for very complex copula structures
                ],
                "default": (256, 256),
                "description": "Generator architecture - adaptive to dataset complexity and copula modeling needs"
            },
            "discriminator_dim": {
                "type": "categorical",
                "choices": [
                    (128, 128),          # Matches small generator
                    (256, 256),          # Matches medium generator - most stable
                    (512, 512),          # Matches large generator
                    (256, 512),          # Stronger discriminator for challenging copula structures
                    (512, 256),          # Funnel discriminator for feature selection in copula space
                    (128, 256, 128),     # Deep discriminator for small datasets
                    (256, 512, 256),     # Deep discriminator for medium datasets
                    (512, 1024, 512),    # Deep discriminator for complex datasets
                    (128, 256, 512, 256), # Very deep for complex dependency detection
                    (512, 1024, 512, 256) # Very deep for challenging copula modeling
                ],
                "default": (256, 256),
                "description": "Discriminator architecture - balanced complexity for copula space discrimination"
            },
            "pac": {
                "type": "int",
                "low": 1,
                "high": 20,
                "step": 1,
                "default": 10,
                "description": "PackedGAN group size - 10 optimal for most copula structures"
            },
            "generator_decay": {
                "type": "float",
                "low": 1e-8,
                "high": 1e-3,
                "log": True,
                "default": 1e-6,
                "description": "Generator L2 regularization - prevents overfitting in copula modeling"
            },
            "discriminator_decay": {
                "type": "float",
                "low": 1e-8,
                "high": 1e-3,
                "log": True,
                "default": 1e-6,
                "description": "Discriminator L2 regularization - maintains training balance"
            },
            "discriminator_steps": {
                "type": "int",
                "low": 1,
                "high": 5,
                "step": 1,
                "default": 1,
                "description": "Discriminator training steps per generator step - 1 for balanced copula learning"
            },
            "beta1": {
                "type": "float",
                "low": 0.1,
                "high": 0.9,
                "default": 0.5,
                "description": "Adam optimizer beta1 parameter - 0.5 optimal for GAN training stability"
            },
            "beta2": {
                "type": "float",
                "low": 0.9,
                "high": 0.999,
                "default": 0.999,
                "description": "Adam optimizer beta2 parameter - 0.999 for smooth convergence"
            },
            "copula_regularization": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
                "default": 0.1,
                "description": "Copula structure regularization strength - 0.1 preserves marginal distributions"
            },
            "gradient_penalty": {
                "type": "float",
                "low": 0.0,
                "high": 10.0,
                "default": 0.0,
                "description": "Gradient penalty coefficient - 0 for standard training, >0 for WGAN-GP style"
            }
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set model configuration parameters.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.model_config.update(config)
        logger.info(f"CopulaGAN configuration updated: {config}")
    
    def train(
        self,
        data: pd.DataFrame,
        epochs: int = 300,
        batch_size: int = 500,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the CopulaGAN model on the provided data.
        
        Args:
            data: Training dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to show training progress
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results and metadata
        """
        logger.info(f"Starting CopulaGAN training on data shape: {data.shape}")

        if data.empty:
            raise ValueError("Training data cannot be empty")

        start_time = time.time()

        try:
            # MINIMAL PREPROCESSING - just handle missing values that cause NoneType errors
            training_data = data.copy()

            # Enhanced preprocessing for dataset-specific issues
            missing_count = training_data.isnull().sum().sum()
            logger.info(f"[COPULAGAN] Data analysis: {missing_count} missing values, shape: {training_data.shape}")

            # Step 1: Handle missing values
            if missing_count > 0:
                logger.info(f"[COPULAGAN] Found {missing_count} missing values - applying preprocessing")
                for col in training_data.columns:
                    if training_data[col].isnull().any():
                        if training_data[col].dtype == 'object':
                            training_data[col].fillna('Unknown', inplace=True)
                        else:
                            # Use median for numerical columns (more robust than 0 for extreme distributions)
                            median_val = training_data[col].median()
                            if pd.isna(median_val):
                                median_val = 0.0
                            training_data[col].fillna(median_val, inplace=True)
                logger.info(f"[COPULAGAN] Missing values filled: {training_data.isnull().sum().sum()} remaining")

            # Step 2: Handle extreme outliers that can break beta distribution fitting
            numeric_cols = training_data.select_dtypes(include=[np.number]).columns
            outlier_count = 0
            for col in numeric_cols:
                # Calculate IQR and outlier bounds
                q1 = training_data[col].quantile(0.01)  # Use 1st and 99th percentile for extreme datasets
                q3 = training_data[col].quantile(0.99)

                # Cap extreme outliers to reduce distribution fitting issues
                original_min, original_max = training_data[col].min(), training_data[col].max()
                training_data[col] = training_data[col].clip(lower=q1, upper=q3)

                clipped_values = ((training_data[col] == q1) | (training_data[col] == q3)).sum()
                if clipped_values > 0:
                    outlier_count += clipped_values
                    logger.info(f"[COPULAGAN] Clipped {clipped_values} extreme values in '{col}' (was {original_min:.2f}-{original_max:.2f}, now {q1:.2f}-{q3:.2f})")

            if outlier_count > 0:
                logger.info(f"[COPULAGAN] Total extreme outliers clipped: {outlier_count}")
            else:
                logger.info("[COPULAGAN] No extreme outliers detected")

            # Step 3: Handle "near-zero range" beta distribution issues
            # This occurs when columns have very tight value ranges
            zero_range_fixes = 0
            for col in numeric_cols:
                col_range = training_data[col].max() - training_data[col].min()
                col_std = training_data[col].std()

                # If range or std is very small, add small amount of noise to prevent fitting issues
                if col_range < 1e-6 or col_std < 1e-6:
                    logger.warning(f"[COPULAGAN] Column '{col}' has near-zero range (range={col_range:.8f}, std={col_std:.8f})")

                    # Add minimal gaussian noise (0.1% of mean) to break ties and create variance
                    noise_scale = max(abs(training_data[col].mean()) * 0.001, 1e-6)
                    noise = np.random.normal(0, noise_scale, len(training_data))
                    training_data[col] = training_data[col] + noise

                    new_range = training_data[col].max() - training_data[col].min()
                    new_std = training_data[col].std()
                    logger.info(f"[COPULAGAN] Added noise to '{col}': new range={new_range:.8f}, new std={new_std:.8f}")
                    zero_range_fixes += 1

            if zero_range_fixes > 0:
                logger.info(f"[COPULAGAN] Fixed {zero_range_fixes} columns with near-zero ranges")
            else:
                logger.info("[COPULAGAN] No near-zero range issues detected")

            # Create metadata
            self._metadata = SingleTableMetadata()
            self._metadata.detect_from_dataframe(training_data)
            
            # Update metadata with any additional information
            logger.info(f"Detected {len(self._metadata.columns)} columns in metadata")
            
            # Extract hyperparameters from config and kwargs (simplified approach like main branch)
            model_params = {}

            # Core training parameters
            model_params['epochs'] = kwargs.get('epochs', self.model_config.get('epochs', epochs))
            model_params['batch_size'] = kwargs.get('batch_size', self.model_config.get('batch_size', batch_size))

            # Learning rates
            if 'generator_lr' in self.model_config:
                model_params['generator_lr'] = self.model_config['generator_lr']
            if 'discriminator_lr' in self.model_config:
                model_params['discriminator_lr'] = self.model_config['discriminator_lr']

            # Network dimensions
            if 'generator_dim' in self.model_config:
                model_params['generator_dim'] = self.model_config['generator_dim']
            if 'discriminator_dim' in self.model_config:
                model_params['discriminator_dim'] = self.model_config['discriminator_dim']

            # Additional parameters
            for param in ['pac', 'generator_decay', 'discriminator_decay']:
                if param in self.model_config:
                    model_params[param] = self.model_config[param]
            
            # Create CopulaGAN model (like main branch)
            self._copulagan_model = CopulaGANSynthesizer(
                metadata=self._metadata,
                enforce_min_max_values=True,
                enforce_rounding=True,
                **model_params
            )

            if verbose:
                logger.info(f"Training CopulaGAN with parameters: {model_params}")

            # Train the model with enhanced error handling for beta distribution issues
            try:
                logger.info("[COPULAGAN] Starting model training...")
                self._copulagan_model.fit(training_data)
                logger.info("[COPULAGAN] Model training completed successfully")

            except Exception as fit_error:
                error_str = str(fit_error)
                logger.error(f"[COPULAGAN] Model fit failed: {error_str}")

                # Specific handling for beta distribution errors
                if "beta distribution" in error_str and "near-zero range" in error_str:
                    logger.warning("[COPULAGAN] Beta distribution near-zero range error - applying emergency preprocessing")

                    # Emergency preprocessing: Add more aggressive noise to all numeric columns
                    emergency_data = training_data.copy()
                    for col in numeric_cols:
                        # Add more substantial noise (1% of range or 0.01, whichever is larger)
                        col_range = emergency_data[col].max() - emergency_data[col].min()
                        noise_scale = max(col_range * 0.01, 0.01)
                        noise = np.random.normal(0, noise_scale, len(emergency_data))
                        emergency_data[col] = emergency_data[col] + noise

                        logger.info(f"[COPULAGAN] Emergency: Added {noise_scale:.6f} noise to '{col}'")

                    # Try training again with heavily preprocessed data
                    try:
                        logger.info("[COPULAGAN] Retrying training with emergency preprocessing...")
                        self._copulagan_model.fit(emergency_data)
                        logger.info("[COPULAGAN] Emergency preprocessing successful!")
                        training_data = emergency_data  # Update for consistency

                    except Exception as emergency_error:
                        logger.error(f"[COPULAGAN] Emergency preprocessing also failed: {emergency_error}")
                        # Try one more time with even simpler preprocessing
                        logger.warning("[COPULAGAN] Attempting final fallback: data standardization")

                        try:
                            from sklearn.preprocessing import StandardScaler
                            fallback_data = training_data.copy()

                            # Standardize all numeric columns
                            scaler = StandardScaler()
                            fallback_data[numeric_cols] = scaler.fit_transform(fallback_data[numeric_cols])

                            logger.info("[COPULAGAN] Final attempt with standardized data...")
                            self._copulagan_model.fit(fallback_data)
                            logger.info("[COPULAGAN] Standardization fallback successful!")
                            training_data = fallback_data

                        except Exception as final_error:
                            logger.error(f"[COPULAGAN] All preprocessing attempts failed: {final_error}")
                            raise RuntimeError(f"CopulaGAN training failed after all preprocessing attempts: {error_str}")

                else:
                    # For non-beta distribution errors, raise immediately
                    raise RuntimeError(f"CopulaGAN training error: {error_str}")
            
            training_duration = time.time() - start_time
            
            # Try to extract training history if available
            training_losses = []
            final_loss = None
            convergence_achieved = False
            
            # Note: SDV CopulaGAN doesn't expose detailed training history by default
            # We'll record what we can
            
            training_result = {
                "training_duration_seconds": training_duration,
                "epochs_completed": model_params.get('epochs', epochs),
                "batch_size": model_params.get('batch_size', batch_size),
                "final_loss": final_loss,
                "training_losses": training_losses,
                "convergence_achieved": convergence_achieved,
                "model_parameters": model_params,
                "data_shape": data.shape,
                "memory_usage_mb": self._estimate_memory_usage(),
                "model_size_mb": self._estimate_model_size()
            }
            
            self._training_history = training_result
            
            if verbose:
                logger.info(f"CopulaGAN training completed in {training_duration:.2f} seconds")
            
            return training_result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"CopulaGAN training failed: {error_msg}")
            
            # Provide more specific error handling
            if "metadata" in error_msg.lower():
                detailed_error = f"CopulaGAN metadata error: {error_msg}. Check data types and column names."
            elif "memory" in error_msg.lower():
                detailed_error = f"CopulaGAN memory error: {error_msg}. Try reducing batch_size or epochs."
            elif "dimension" in error_msg.lower() or "shape" in error_msg.lower():
                detailed_error = f"CopulaGAN dimension error: {error_msg}. Check generator_dim and discriminator_dim parameters."
            else:
                detailed_error = f"CopulaGAN training error: {error_msg}"
            
            raise RuntimeError(detailed_error)
    
    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic samples using the trained CopulaGAN model.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional generation parameters
            
        Returns:
            DataFrame containing synthetic samples
        """
        if self._copulagan_model is None:
            raise ValueError("Model must be trained before generating samples")
        
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        logger.info(f"Generating {n_samples} synthetic samples with CopulaGAN")
        
        try:
            start_time = time.time()
            
            # Generate synthetic data
            synthetic_data = self._copulagan_model.sample(num_rows=n_samples)

            generation_time = time.time() - start_time

            logger.info(f"Generated {len(synthetic_data)} samples in {generation_time:.2f} seconds")

            # Validate generated data
            if synthetic_data.empty:
                raise RuntimeError("Generated data is empty")

            if len(synthetic_data) != n_samples:
                logger.warning(f"Generated {len(synthetic_data)} samples instead of requested {n_samples}")

            # No complex reverse transformation - keep it simple like main branch

            return synthetic_data
            
        except Exception as e:
            logger.error(f"CopulaGAN generation failed: {e}")
            raise RuntimeError(f"CopulaGAN generation failed: {str(e)}")
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if self._copulagan_model is None:
            raise ValueError("No trained model to save")
        
        try:
            self._copulagan_model.save(filepath)
            logger.info(f"CopulaGAN model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save CopulaGAN model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            self._copulagan_model = CopulaGANSynthesizer.load(filepath)
            logger.info(f"CopulaGAN model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load CopulaGAN model: {e}")
            raise
    
    def get_training_history(self) -> Optional[Dict[str, Any]]:
        """
        Get training history and metrics.
        
        Returns:
            Dictionary containing training history, or None if not available
        """
        return self._training_history
    
    def _estimate_memory_usage(self) -> float:
        """
        Estimate current memory usage in MB.
        
        Returns:
            Estimated memory usage in MB
        """
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except ImportError:
            return 0.0
    
    def _estimate_model_size(self) -> float:
        """
        Estimate model size in MB.
        
        Returns:
            Estimated model size in MB
        """
        # This is a rough estimate for CopulaGAN
        if self._copulagan_model is None:
            return 0.0
        
        # Estimate based on typical CopulaGAN model sizes
        base_size = 10.0  # Base model size in MB
        
        # Adjust based on network dimensions if available
        try:
            if hasattr(self._copulagan_model, '_generator_dim'):
                gen_dim = self._copulagan_model._generator_dim
                if isinstance(gen_dim, (list, tuple)):
                    gen_params = sum(gen_dim)
                    base_size += gen_params * 0.001  # Rough parameter size estimation
            
            if hasattr(self._copulagan_model, '_discriminator_dim'):
                disc_dim = self._copulagan_model._discriminator_dim
                if isinstance(disc_dim, (list, tuple)):
                    disc_params = sum(disc_dim)
                    base_size += disc_params * 0.001
                    
        except Exception:
            pass
        
        return base_size
    
    def evaluate_on_holdout(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate model on holdout test data.
        
        Args:
            train_data: Training dataset
            test_data: Test dataset for evaluation
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating CopulaGAN on holdout data")
        
        try:
            # Train on training data
            self.train(train_data, verbose=False, **kwargs)
            
            # Generate synthetic data
            synthetic_data = self.generate(len(test_data))
            
            # Basic statistical comparisons
            metrics = {}
            
            # Column-wise statistical similarity
            numerical_cols = test_data.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) > 0:
                from scipy import stats
                
                ks_statistics = []
                wasserstein_distances = []
                
                for col in numerical_cols:
                    if col in synthetic_data.columns:
                        # Kolmogorov-Smirnov test
                        ks_stat, _ = stats.ks_2samp(test_data[col], synthetic_data[col])
                        ks_statistics.append(ks_stat)
                        
                        # Wasserstein distance
                        ws_dist = stats.wasserstein_distance(test_data[col], synthetic_data[col])
                        wasserstein_distances.append(ws_dist)
                
                metrics['mean_ks_statistic'] = np.mean(ks_statistics) if ks_statistics else 0.0
                metrics['mean_wasserstein_distance'] = np.mean(wasserstein_distances) if wasserstein_distances else 0.0
                metrics['statistical_similarity'] = 1.0 - metrics['mean_ks_statistic']  # Higher is better
            
            # Correlation preservation
            if len(numerical_cols) > 1:
                try:
                    orig_corr = test_data[numerical_cols].corr()
                    synth_corr = synthetic_data[numerical_cols].corr()
                    
                    # Correlation matrix similarity
                    corr_diff = np.abs(orig_corr - synth_corr).mean().mean()
                    metrics['correlation_preservation'] = 1.0 / (1.0 + corr_diff)
                except Exception:
                    metrics['correlation_preservation'] = 0.0
            
            logger.info(f"CopulaGAN evaluation completed with {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"CopulaGAN evaluation failed: {e}")
            return {"error": str(e)}

# Export for easy importing
__all__ = ['CopulaGANModel', 'COPULAGAN_AVAILABLE']