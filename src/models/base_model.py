"""
Abstract base class for synthetic tabular data generation models.

This module defines the common interface that all synthetic data models must implement,
ensuring consistent behavior across different model types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path


class SyntheticDataModel(ABC):
    """
    Abstract base class for synthetic tabular data generation models.
    
    This class defines the standard interface that all synthetic data models
    must implement to work with the benchmarking framework.
    """
    
    def __init__(self, device: str = "cpu", random_state: int = 42):
        """
        Initialize the synthetic data model.
        
        Args:
            device: Computing device ("cpu", "cuda", "mps")
            random_state: Random seed for reproducibility
        """
        self.device = device
        self.random_state = random_state
        self.is_trained = False
        self.training_metadata = {}
        self.model_config = {}
        
    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train the synthetic data model on the provided dataset.
        
        Args:
            data: Training dataset as pandas DataFrame
            **kwargs: Model-specific training parameters
            
        Returns:
            Dictionary containing training metadata and metrics
        """
        pass
    
    @abstractmethod
    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data samples.
        
        Args:
            n_samples: Number of synthetic samples to generate
            **kwargs: Model-specific generation parameters
            
        Returns:
            Generated synthetic data as pandas DataFrame
        """
        pass
    
    @abstractmethod
    def get_hyperparameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the hyperparameter search space for optimization.
        
        Returns:
            Dictionary defining hyperparameter search space with structure:
            {
                'parameter_name': {
                    'type': 'int' | 'float' | 'categorical',
                    'low': min_value,  # for int/float
                    'high': max_value,  # for int/float 
                    'choices': [list],  # for categorical
                    'log': bool,  # for float (log scale)
                    'step': int,  # for int (step size)
                    'description': str
                }
            }
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Directory path where model should be saved
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Directory path where model is saved
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_type": self.__class__.__name__,
            "device": self.device,
            "random_state": self.random_state,
            "is_trained": self.is_trained,
            "config": self.model_config,
            "training_metadata": self.training_metadata
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set model configuration parameters.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.model_config.update(config)
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate input data for training or generation.
        
        Args:
            data: Input dataset to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if data.empty:
            return False, "Dataset is empty"
        
        if data.isnull().all().any():
            return False, "Dataset contains columns with all missing values"
        
        if len(data.columns) == 0:
            return False, "Dataset has no columns"
        
        return True, ""
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training process and results.
        
        Returns:
            Dictionary containing training summary
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "metadata": self.training_metadata,
            "config": self.model_config
        }


class ModelNotTrainedError(Exception):
    """Exception raised when attempting to use an untrained model."""
    pass


class ModelConfigurationError(Exception):
    """Exception raised when model configuration is invalid."""
    pass


class DataValidationError(Exception):
    """Exception raised when input data validation fails."""
    pass