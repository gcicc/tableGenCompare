"""
CTAB-GAN model implementation for the synthetic tabular data framework.

This module wraps the CTAB-GAN model to work with the unified framework interface.
CTAB-GAN is specifically designed for mixed-type tabular data with advanced
conditional generation capabilities.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import logging
import sys
import os

from ..base_model import SyntheticDataModel, ModelNotTrainedError, DataValidationError

logger = logging.getLogger(__name__)

# Add CTAB-GAN path to Python path
try:
    ctabgan_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "CTAB-GAN")
    if ctabgan_path not in sys.path:
        sys.path.insert(0, ctabgan_path)  # Insert at beginning to prioritize this version
    
    # Clear module cache to ensure fresh import
    if 'model.ctabgan' in sys.modules:
        del sys.modules['model.ctabgan']
    if 'model' in sys.modules:
        del sys.modules['model']
    
    from model.ctabgan import CTABGAN
    CTABGAN_AVAILABLE = True
except ImportError as e:
    CTABGAN_AVAILABLE = False  
    logger.warning(f"CTAB-GAN not available: {e}")


class CTABGANModel(SyntheticDataModel):
    """
    CTAB-GAN model implementation for synthetic tabular data generation.
    
    CTAB-GAN (Conditional Tabular GAN) is a specialized GAN for tabular data
    with advanced conditional generation and mixed-type data handling.
    """
    
    def __init__(self, device: str = "cpu", random_state: int = 42):
        """
        Initialize CTAB-GAN model wrapper.
        
        Args:
            device: Computing device (CTAB-GAN uses CPU/GPU internally)
            random_state: Random seed for reproducibility
        """
        if not CTABGAN_AVAILABLE:
            raise ImportError(
                "CTAB-GAN is not available. Please ensure CTAB-GAN directory exists."
            )
        
        super().__init__(device, random_state)
        
        # CTAB-GAN specific initialization
        self._ctabgan_model = None
        self._categorical_columns = []
        self._integer_columns = []
        self._mixed_columns = {}
        self._log_columns = []
        self._training_history = None
        
        # Default CTAB-GAN parameters
        self.default_config = {
            "epochs": 300,
            "batch_size": 256,
            "class_dim": 256,
            "random_dim": 100,
            "num_channels": 64,
            "test_ratio": 0.2
        }
        
        self.set_config(self.default_config)
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train the CTAB-GAN model on the provided dataset.
        
        Args:
            data: Training dataset as pandas DataFrame
            **kwargs: Training parameters
            
        Returns:
            Dictionary containing training metadata and metrics
        """
        # Validate input data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise DataValidationError(f"Data validation failed: {error_msg}")
        
        start_time = datetime.now()
        
        # Extract training parameters
        epochs = kwargs.get("epochs", self.model_config.get("epochs", 300))
        categorical_columns = kwargs.get("categorical_columns", None)
        integer_columns = kwargs.get("integer_columns", None)
        mixed_columns = kwargs.get("mixed_columns", {})
        log_columns = kwargs.get("log_columns", [])
        
        # Auto-detect column types if not provided
        if categorical_columns is None:
            categorical_columns = self._auto_detect_categorical_columns(data)
        if integer_columns is None:
            integer_columns = self._auto_detect_integer_columns(data)
        
        self._categorical_columns = categorical_columns
        self._integer_columns = integer_columns
        self._mixed_columns = mixed_columns
        self._log_columns = log_columns
        
        try:
            # Save data to temporary CSV for CTAB-GAN
            temp_csv_path = "temp_ctabgan_data.csv"
            data.to_csv(temp_csv_path, index=False)
            
            # Determine a reasonable target column for CTAB-GAN
            # Use the last column as target (common ML convention), ensure it's categorical for stratification
            target_column = data.columns[-1]
            
            # Ensure target column is in categorical columns for proper handling
            if target_column not in categorical_columns:
                categorical_columns.append(target_column)
            
            # Remove target column from integer columns if it's there to avoid conflicts
            if target_column in integer_columns:
                integer_columns.remove(target_column)
            
            problem_type = {"Classification": target_column}
            
            # Initialize CTAB-GAN model
            self._ctabgan_model = CTABGAN(
                raw_csv_path=temp_csv_path,
                test_ratio=self.model_config.get("test_ratio", 0.2),
                categorical_columns=categorical_columns,
                log_columns=log_columns,
                mixed_columns=mixed_columns,
                integer_columns=integer_columns,
                problem_type=problem_type,
                epochs=epochs
            )
            
            # Train the model
            self._ctabgan_model.fit()
            
            # Clean up temporary file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
            
            self.is_trained = True
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Store training metadata
            self.training_metadata = {
                "training_time": training_time,
                "epochs": epochs,
                "data_shape": data.shape,
                "categorical_columns": categorical_columns,
                "integer_columns": integer_columns,
                "mixed_columns": mixed_columns,
                "log_columns": log_columns,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            logger.info(f"CTAB-GAN training completed in {training_time:.2f} seconds")
            
            return self.training_metadata
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists("temp_ctabgan_data.csv"):
                os.remove("temp_ctabgan_data.csv")
            logger.error(f"CTAB-GAN training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")
    
    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data samples.
        
        Args:
            n_samples: Number of synthetic samples to generate
            **kwargs: Generation parameters
            
        Returns:
            Generated synthetic data as pandas DataFrame
        """
        if not self.is_trained or self._ctabgan_model is None:
            raise ModelNotTrainedError("Model must be trained before generating samples")
        
        try:
            # Generate samples using CTAB-GAN
            synthetic_data = self._ctabgan_model.generate_samples()
            
            # CTAB-GAN FIX: Ensure categorical columns have consistent data types
            # This prevents TRTS evaluation errors with string/int comparison
            for col in self._categorical_columns:
                if col in synthetic_data.columns:
                    try:
                        # Convert string categorical values back to numeric if they represent numbers
                        if synthetic_data[col].dtype == 'object':
                            # Try to convert to numeric, handling potential string categories
                            numeric_values = pd.to_numeric(synthetic_data[col], errors='coerce')
                            if not numeric_values.isna().all():
                                # If conversion was successful for most values, use it
                                if numeric_values.notna().sum() / len(numeric_values) > 0.8:
                                    synthetic_data[col] = numeric_values.fillna(0).astype(int)
                    except Exception:
                        pass  # If conversion fails, leave as is
            
            # Sample the requested number of rows
            if len(synthetic_data) > n_samples:
                synthetic_data = synthetic_data.sample(n=n_samples, random_state=self.random_state)
            elif len(synthetic_data) < n_samples:
                # If we need more samples, repeat the generation
                additional_needed = n_samples - len(synthetic_data)
                additional_data = synthetic_data.sample(n=additional_needed, replace=True, random_state=self.random_state)
                synthetic_data = pd.concat([synthetic_data, additional_data], ignore_index=True)
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"CTAB-GAN generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")
    
    def get_hyperparameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the hyperparameter search space for CTAB-GAN optimization.
        
        Returns:
            Dictionary defining hyperparameter search space
        """
        return {
            "epochs": {
                "type": "int",
                "low": 100,
                "high": 1000,
                "step": 50,
                "description": "Number of training epochs"
            },
            "batch_size": {
                "type": "categorical",
                "choices": [64, 128, 256, 500],
                "description": "Training batch size"
            },
            "class_dim": {
                "type": "categorical",
                "choices": [128, 256, 512],
                "description": "Class embedding dimension"
            },
            "random_dim": {
                "type": "int",
                "low": 50,
                "high": 200,
                "step": 25,
                "description": "Random noise dimension"
            },
            "num_channels": {
                "type": "int",
                "low": 32,
                "high": 128,
                "step": 16,
                "description": "Number of channels in generator/discriminator"
            }
        }
    
    def save_model(self, path: str) -> None:
        """
        Save the trained CTAB-GAN model to disk.
        
        Args:
            path: Directory path where model should be saved
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Cannot save untrained model")
        
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "model_type": "CTABGAN",
            "training_metadata": self.training_metadata,
            "model_config": self.model_config,
            "categorical_columns": self._categorical_columns,
            "integer_columns": self._integer_columns,
            "mixed_columns": self._mixed_columns,
            "log_columns": self._log_columns
        }
        
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Note: CTAB-GAN doesn't have built-in model saving
        # This implementation saves metadata for reconstruction
        logger.info(f"CTAB-GAN model metadata saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained CTAB-GAN model from disk.
        
        Args:
            path: Directory path where model is saved
        """
        model_dir = Path(path)
        metadata_file = model_dir / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Model metadata not found at {path}")
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Restore model state
        self.training_metadata = metadata["training_metadata"]
        self.model_config = metadata["model_config"]
        self._categorical_columns = metadata["categorical_columns"]
        self._integer_columns = metadata["integer_columns"]
        self._mixed_columns = metadata["mixed_columns"]
        self._log_columns = metadata["log_columns"]
        
        # Note: CTAB-GAN would need to be retrained as it doesn't support model saving
        # This is a limitation of the original CTAB-GAN implementation
        logger.warning("CTAB-GAN model loaded (metadata only). Model needs retraining for generation.")
    
    def _auto_detect_categorical_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect categorical columns based on data types and cardinality."""
        categorical_columns = []
        
        for column in data.columns:
            if data[column].dtype == 'object':
                categorical_columns.append(column)
            elif data[column].dtype in ['int64', 'float64']:
                # Check if it's a low-cardinality numeric column
                unique_values = data[column].nunique()
                if unique_values <= 10:  # Threshold for categorical
                    categorical_columns.append(column)
        
        return categorical_columns
    
    def _auto_detect_integer_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect integer columns, excluding those already identified as categorical."""
        categorical_columns = self._auto_detect_categorical_columns(data)
        integer_columns = []
        
        for column in data.columns:
            # Skip columns that are already identified as categorical
            if column in categorical_columns:
                continue
                
            if data[column].dtype in ['int64', 'int32']:
                integer_columns.append(column)
            elif data[column].dtype in ['float64', 'float32']:
                # Check if float column contains only whole numbers
                non_null_data = data[column].dropna()
                if len(non_null_data) > 0 and non_null_data.apply(lambda x: float(x).is_integer()).all():
                    integer_columns.append(column)
        
        return integer_columns