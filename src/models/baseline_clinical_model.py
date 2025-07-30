"""
Baseline Clinical Model for Synthetic Data Generation

Simple baseline model that can be used as fallback when advanced models are not available.
"""

import pandas as pd
import numpy as np
import time


class BaselineClinicalModel:
    """Baseline synthetic data model for clinical applications."""
    
    def __init__(self, name, **params):
        self.name = name
        self.params = params
        self.is_fitted = False
        self.training_time = 0
        
    def fit(self, data, discrete_columns=None):
        """Fit model to training data."""
        start_time = time.time()
        
        # Store data statistics for generation
        self.data_stats = {}
        self.discrete_columns = discrete_columns or []
        
        for col in data.columns:
            if col in self.discrete_columns:
                self.data_stats[col] = {
                    'type': 'discrete',
                    'values': list(data[col].unique()),
                    'probabilities': data[col].value_counts(normalize=True).to_dict()
                }
            else:
                self.data_stats[col] = {
                    'type': 'continuous',
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max()
                }
        
        # Simulate training time based on complexity
        complexity_factor = self.params.get('epochs', 100) * self.params.get('batch_size', 64) / 10000
        time.sleep(min(complexity_factor, 0.5))  # Cap at 0.5 seconds for demo
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
    def generate(self, n_samples):
        """Generate synthetic samples."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating data")
        
        synthetic_data = pd.DataFrame()
        
        for col, stats in self.data_stats.items():
            if stats['type'] == 'discrete':
                # Sample from observed distribution
                values = list(stats['probabilities'].keys())
                probs = list(stats['probabilities'].values())
                synthetic_data[col] = np.random.choice(values, size=n_samples, p=probs)
            else:
                # Generate from normal distribution with noise
                noise_factor = self.params.get('noise_level', 0.1)
                mean = stats['mean']
                std = stats['std'] * (1 + noise_factor)
                samples = np.random.normal(mean, std, n_samples)
                # Clip to observed range
                samples = np.clip(samples, stats['min'], stats['max'])
                synthetic_data[col] = samples
        
        return synthetic_data