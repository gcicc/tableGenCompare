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

from ..base_model import BaseSyntheticModel

class CopulaGANModel(BaseSyntheticModel):
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
        Get the hyperparameter search space for optimization.
        
        Returns:
            Dictionary defining the hyperparameter space for CopulaGAN
        """
        return {
            "epochs": {
                "type": "int",
                "low": 50,
                "high": 500,
                "default": 300,
                "description": "Number of training epochs"
            },
            "batch_size": {
                "type": "int", 
                "low": 32,
                "high": 1000,
                "default": 500,
                "description": "Training batch size"
            },
            "generator_lr": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-2,
                "default": 2e-4,
                "log": True,
                "description": "Generator learning rate"
            },
            "discriminator_lr": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-2,
                "default": 2e-4,
                "log": True,
                "description": "Discriminator learning rate"
            },
            "generator_dim": {
                "type": "categorical",
                "choices": [(128, 128), (256, 256), (512, 512), (128, 256), (256, 512)],
                "default": (256, 256),
                "description": "Generator network dimensions"
            },
            "discriminator_dim": {
                "type": "categorical",
                "choices": [(128, 128), (256, 256), (512, 512), (128, 256), (256, 512)],
                "default": (256, 256),
                "description": "Discriminator network dimensions"
            },
            "pac": {
                "type": "int",
                "low": 1,
                "high": 10,
                "default": 10,
                "description": "Number of samples to group together when applying discriminator"
            },
            "generator_decay": {
                "type": "float",
                "low": 1e-8,
                "high": 1e-2,
                "default": 1e-6,
                "log": True,
                "description": "Generator weight decay"
            },
            "discriminator_decay": {
                "type": "float",
                "low": 1e-8,
                "high": 1e-2,
                "default": 1e-6,
                "log": True,
                "description": "Discriminator weight decay"
            }
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set model configuration parameters.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.config.update(config)
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
            # Create metadata
            self._metadata = SingleTableMetadata()
            self._metadata.detect_from_dataframe(data)
            
            # Update metadata with any additional information
            logger.info(f"Detected {len(self._metadata.columns)} columns in metadata")
            
            # Extract hyperparameters from config and kwargs
            model_params = {}
            
            # Core training parameters
            model_params['epochs'] = kwargs.get('epochs', self.config.get('epochs', epochs))
            model_params['batch_size'] = kwargs.get('batch_size', self.config.get('batch_size', batch_size))
            
            # Learning rates
            if 'generator_lr' in self.config:
                model_params['generator_lr'] = self.config['generator_lr']
            if 'discriminator_lr' in self.config:
                model_params['discriminator_lr'] = self.config['discriminator_lr']
            
            # Network dimensions
            if 'generator_dim' in self.config:
                model_params['generator_dim'] = self.config['generator_dim']
            if 'discriminator_dim' in self.config:
                model_params['discriminator_dim'] = self.config['discriminator_dim']
            
            # Additional parameters
            for param in ['pac', 'generator_decay', 'discriminator_decay']:
                if param in self.config:
                    model_params[param] = self.config[param]
            
            # Create CopulaGAN model
            self._copulagan_model = CopulaGANSynthesizer(
                metadata=self._metadata,
                enforce_min_max_values=True,
                enforce_rounding=True,
                **model_params
            )
            
            if verbose:
                logger.info(f"Training CopulaGAN with parameters: {model_params}")
            
            # Train the model
            self._copulagan_model.fit(data)
            
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
            logger.error(f"CopulaGAN training failed: {e}")
            raise RuntimeError(f"CopulaGAN training failed: {str(e)}")
    
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