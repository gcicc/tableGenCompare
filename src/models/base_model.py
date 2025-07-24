from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

class BaseSyntheticModel(ABC):
    """Abstract base class for synthetic data generation models."""
    
    def __init__(self, name: str, random_state: Optional[int] = None):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.training_time = 0.0
        self.generation_time = 0.0
        
    @abstractmethod
    def get_param_space(self) -> Dict[str, Any]:
        """Return Optuna parameter space for hyperparameter optimization."""
        pass
        
    @abstractmethod
    def create_model(self, hyperparams: Dict[str, Any]) -> Any:
        """Create model instance with given hyperparameters."""
        pass
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, discrete_columns: list = None) -> None:
        """Fit the model to training data."""
        pass
        
    @abstractmethod
    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic samples."""
        pass
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format."""
        if data.empty:
            raise ValueError("Input data is empty")
        if data.isnull().all().any():
            raise ValueError("Input data contains columns with all NaN values")
        return True
        
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information and statistics."""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'generation_time': self.generation_time,
            'random_state': self.random_state
        }