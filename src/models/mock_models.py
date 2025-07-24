"""
Mock Model Implementations for Initial Testing

Creates mock synthetic data generators that produce realistic-looking results
for initial visualization and framework testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import time
import logging
from .base_model import BaseSyntheticModel

class MockCTGANModel(BaseSyntheticModel):
    """Mock CTGAN model that generates synthetic data with realistic patterns."""
    
    def __init__(self, random_state: Optional[int] = None):
        super().__init__("CTGAN (Mock)", random_state)
        self.original_data = None
        self.column_stats = {}
        
    def get_param_space(self) -> Dict[str, Any]:
        return {
            'epochs': ('int', 100, 500, 50),
            'batch_size': ('categorical', [32, 64, 128]),
            'learning_rate': ('float', 1e-4, 1e-2, True),
        }
    
    def create_model(self, hyperparams: Dict[str, Any]) -> Any:
        return {"hyperparams": hyperparams, "model_type": "mock_ctgan"}
    
    def fit(self, data: pd.DataFrame, discrete_columns: list = None) -> None:
        self.validate_data(data)
        start_time = time.time()
        
        self.original_data = data.copy()
        
        # Store column statistics for synthetic generation
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                self.column_stats[col] = {
                    'type': 'numeric',
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max()
                }
            else:
                self.column_stats[col] = {
                    'type': 'categorical',
                    'categories': data[col].unique(),
                    'probabilities': data[col].value_counts(normalize=True).to_dict()
                }
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        logging.info(f"Mock CTGAN training completed in {self.training_time:.2f} seconds")
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating samples")
        
        start_time = time.time()
        np.random.seed(self.random_state)
        
        synthetic_data = {}
        
        for col, stats in self.column_stats.items():
            if stats['type'] == 'numeric':
                # Add slight noise to make it look more realistic
                noise_factor = 0.1
                synthetic_data[col] = np.random.normal(
                    stats['mean'] * (1 + np.random.normal(0, noise_factor, n_samples)),
                    stats['std'] * (1 + np.random.normal(0, noise_factor, n_samples)),
                    n_samples
                )
                # Clip to original bounds
                synthetic_data[col] = np.clip(synthetic_data[col], stats['min'], stats['max'])
            else:
                # Categorical data
                categories = list(stats['probabilities'].keys())
                probabilities = list(stats['probabilities'].values())
                synthetic_data[col] = np.random.choice(categories, n_samples, p=probabilities)
        
        result = pd.DataFrame(synthetic_data)
        self.generation_time = time.time() - start_time
        
        logging.info(f"Generated {n_samples} samples in {self.generation_time:.2f} seconds")
        return result

class MockTVAEModel(BaseSyntheticModel):
    """Mock TVAE model with different characteristics than CTGAN."""
    
    def __init__(self, random_state: Optional[int] = None):
        super().__init__("TVAE (Mock)", random_state)
        self.original_data = None
        self.column_stats = {}
        
    def get_param_space(self) -> Dict[str, Any]:
        return {
            'epochs': ('int', 50, 300, 25),
            'batch_size': ('categorical', [64, 128, 256]),
            'latent_dim': ('int', 16, 128, 16),
        }
    
    def create_model(self, hyperparams: Dict[str, Any]) -> Any:
        return {"hyperparams": hyperparams, "model_type": "mock_tvae"}
    
    def fit(self, data: pd.DataFrame, discrete_columns: list = None) -> None:
        self.validate_data(data)
        start_time = time.time()
        
        self.original_data = data.copy()
        
        # Store column statistics (similar to CTGAN but with different noise patterns)
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                self.column_stats[col] = {
                    'type': 'numeric',
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median()
                }
            else:
                self.column_stats[col] = {
                    'type': 'categorical',
                    'categories': data[col].unique(),
                    'probabilities': data[col].value_counts(normalize=True).to_dict()
                }
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        logging.info(f"Mock TVAE training completed in {self.training_time:.2f} seconds")
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating samples")
        
        start_time = time.time()
        np.random.seed(self.random_state)
        
        synthetic_data = {}
        
        for col, stats in self.column_stats.items():
            if stats['type'] == 'numeric':
                # TVAE tends to preserve median better
                base_values = np.random.normal(stats['median'], stats['std'] * 0.8, n_samples)
                noise = np.random.normal(0, stats['std'] * 0.2, n_samples)
                synthetic_data[col] = base_values + noise
                synthetic_data[col] = np.clip(synthetic_data[col], stats['min'], stats['max'])
            else:
                # Slightly different categorical sampling
                categories = list(stats['probabilities'].keys())
                probabilities = list(stats['probabilities'].values())
                # Add small perturbation to probabilities
                probabilities = np.array(probabilities)
                probabilities += np.random.normal(0, 0.01, len(probabilities))
                probabilities = np.abs(probabilities)
                probabilities /= probabilities.sum()
                synthetic_data[col] = np.random.choice(categories, n_samples, p=probabilities)
        
        result = pd.DataFrame(synthetic_data)
        self.generation_time = time.time() - start_time
        
        logging.info(f"Generated {n_samples} samples in {self.generation_time:.2f} seconds")
        return result

class MockCopulaGANModel(BaseSyntheticModel):
    """Mock CopulaGAN model with copula-based characteristics."""
    
    def __init__(self, random_state: Optional[int] = None):
        super().__init__("CopulaGAN (Mock)", random_state)
        self.original_data = None
        self.correlations = None
        self.column_stats = {}
        
    def get_param_space(self) -> Dict[str, Any]:
        return {
            'epochs': ('int', 100, 400, 50),
            'batch_size': ('categorical', [32, 64, 128]),
            'discriminator_lr': ('float', 1e-4, 1e-2, True),
        }
    
    def create_model(self, hyperparams: Dict[str, Any]) -> Any:
        return {"hyperparams": hyperparams, "model_type": "mock_copulagan"}
    
    def fit(self, data: pd.DataFrame, discrete_columns: list = None) -> None:
        self.validate_data(data)
        start_time = time.time()
        
        self.original_data = data.copy()
        
        # Store correlations for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            self.correlations = data[numeric_cols].corr()
        
        # Store column statistics
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                self.column_stats[col] = {
                    'type': 'numeric',
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'skew': data[col].skew()
                }
            else:
                self.column_stats[col] = {
                    'type': 'categorical',
                    'categories': data[col].unique(),
                    'probabilities': data[col].value_counts(normalize=True).to_dict()
                }
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        logging.info(f"Mock CopulaGAN training completed in {self.training_time:.2f} seconds")
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating samples")
        
        start_time = time.time()
        np.random.seed(self.random_state)
        
        synthetic_data = {}
        
        # Generate correlated numeric data if correlations exist
        numeric_cols = [col for col, stats in self.column_stats.items() if stats['type'] == 'numeric']
        
        if len(numeric_cols) > 1 and self.correlations is not None:
            # Generate multivariate normal data to preserve correlations
            means = [self.column_stats[col]['mean'] for col in numeric_cols]
            stds = [self.column_stats[col]['std'] for col in numeric_cols]
            
            # Create covariance matrix
            cov_matrix = np.outer(stds, stds) * self.correlations.loc[numeric_cols, numeric_cols].values
            
            # Generate correlated data
            correlated_data = np.random.multivariate_normal(means, cov_matrix, n_samples)
            
            for i, col in enumerate(numeric_cols):
                synthetic_data[col] = np.clip(
                    correlated_data[:, i], 
                    self.column_stats[col]['min'], 
                    self.column_stats[col]['max']
                )
        else:
            # Generate numeric data independently
            for col in numeric_cols:
                stats = self.column_stats[col]
                synthetic_data[col] = np.random.normal(stats['mean'], stats['std'], n_samples)
                synthetic_data[col] = np.clip(synthetic_data[col], stats['min'], stats['max'])
        
        # Generate categorical data
        for col, stats in self.column_stats.items():
            if stats['type'] == 'categorical':
                categories = list(stats['probabilities'].keys())
                probabilities = list(stats['probabilities'].values())
                synthetic_data[col] = np.random.choice(categories, n_samples, p=probabilities)
        
        result = pd.DataFrame(synthetic_data)
        self.generation_time = time.time() - start_time
        
        logging.info(f"Generated {n_samples} samples in {self.generation_time:.2f} seconds")
        return result

class MockGANerAidModel(BaseSyntheticModel):
    """Mock GANerAid model with LSTM-based characteristics."""
    
    def __init__(self, random_state: Optional[int] = None):
        super().__init__("GANerAid (Mock)", random_state)
        self.original_data = None
        self.column_stats = {}
        
    def get_param_space(self) -> Dict[str, Any]:
        return {
            'epochs': ('int', 200, 600, 100),
            'batch_size': ('categorical', [64, 128, 256]),
            'lstm_units': ('int', 32, 256, 32),
        }
    
    def create_model(self, hyperparams: Dict[str, Any]) -> Any:
        return {"hyperparams": hyperparams, "model_type": "mock_ganeraid"}
    
    def fit(self, data: pd.DataFrame, discrete_columns: list = None) -> None:
        self.validate_data(data)
        start_time = time.time()
        
        self.original_data = data.copy()
        
        # Store more detailed statistics for LSTM-based generation
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                self.column_stats[col] = {
                    'type': 'numeric',
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'q25': data[col].quantile(0.25),
                    'q75': data[col].quantile(0.75)
                }
            else:
                self.column_stats[col] = {
                    'type': 'categorical',
                    'categories': data[col].unique(),
                    'probabilities': data[col].value_counts(normalize=True).to_dict()
                }
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        logging.info(f"Mock GANerAid training completed in {self.training_time:.2f} seconds")
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating samples")
        
        start_time = time.time()
        np.random.seed(self.random_state)
        
        synthetic_data = {}
        
        for col, stats in self.column_stats.items():
            if stats['type'] == 'numeric':
                # GANerAid tends to produce data with better boundary preservation
                q25, q75 = stats['q25'], stats['q75']
                iqr = q75 - q25
                
                # Generate data using a mixture of distributions
                normal_samples = np.random.normal(stats['mean'], stats['std'] * 0.9, n_samples)
                uniform_samples = np.random.uniform(q25 - iqr*0.1, q75 + iqr*0.1, n_samples)
                
                # Mix the two distributions
                mix_weights = np.random.beta(2, 2, n_samples)
                synthetic_data[col] = mix_weights * normal_samples + (1 - mix_weights) * uniform_samples
                synthetic_data[col] = np.clip(synthetic_data[col], stats['min'], stats['max'])
            else:
                # More stable categorical generation
                categories = list(stats['probabilities'].keys())
                probabilities = list(stats['probabilities'].values())
                synthetic_data[col] = np.random.choice(categories, n_samples, p=probabilities)
        
        result = pd.DataFrame(synthetic_data)
        self.generation_time = time.time() - start_time
        
        logging.info(f"Generated {n_samples} samples in {self.generation_time:.2f} seconds")
        return result