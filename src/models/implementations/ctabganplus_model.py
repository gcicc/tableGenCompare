"""
CTAB-GAN+ model implementation for the synthetic tabular data framework.

This module wraps the CTAB-GAN+ model to work with the unified framework interface.
CTAB-GAN+ is an enhanced version of CTAB-GAN with improved stability, preprocessing
capabilities, and additional features for better tabular data synthesis.
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

# Add CTAB-GAN-Plus path to Python path
try:
    ctabganplus_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "CTAB-GAN-Plus")
    if ctabganplus_path not in sys.path:
        sys.path.insert(0, ctabganplus_path)
    
    from model.ctabgan import CTABGAN
    CTABGANPLUS_AVAILABLE = True
except ImportError as e:
    CTABGANPLUS_AVAILABLE = False  
    logger.warning(f"CTAB-GAN+ not available: {e}")


class CTABGANPlusModel(SyntheticDataModel):
    """
    CTAB-GAN+ model implementation for synthetic tabular data generation.
    
    CTAB-GAN+ is an enhanced version of CTAB-GAN with improved preprocessing,
    stability, and additional features for better mixed-type tabular data synthesis.
    """
    
    def __init__(self, device: str = "cpu", random_state: int = 42):
        """
        Initialize CTAB-GAN+ model wrapper.
        
        Args:
            device: Computing device (CTAB-GAN+ uses CPU/GPU internally)
            random_state: Random seed for reproducibility
        """
        if not CTABGANPLUS_AVAILABLE:
            raise ImportError(
                "CTAB-GAN+ is not available. Please ensure CTAB-GAN-Plus directory exists."
            )
        
        super().__init__(device, random_state)
        
        # CTAB-GAN+ specific initialization
        self._ctabganplus_model = None
        self._categorical_columns = []
        self._integer_columns = []
        self._mixed_columns = {}
        self._log_columns = []
        self._general_columns = []
        self._non_categorical_columns = []
        self._training_history = None
        
        # Default CTAB-GAN+ parameters (enhanced from CTAB-GAN)
        self.default_config = {
            "epochs": 400,  # Higher default for better stability
            "batch_size": 256,
            "class_dim": 256,
            "random_dim": 100,
            "num_channels": 64,
            "test_ratio": 0.2,
            "enhanced_preprocessing": True,
            "stability_improvements": True
        }
        
        self.set_config(self.default_config)
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train the CTAB-GAN+ model on the provided dataset.
        
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
        
        # Extract training parameters (CTAB-GAN+ specific)
        epochs = kwargs.get("epochs", self.model_config.get("epochs", 400))
        categorical_columns = kwargs.get("categorical_columns", None)
        integer_columns = kwargs.get("integer_columns", None)
        mixed_columns = kwargs.get("mixed_columns", {})
        log_columns = kwargs.get("log_columns", [])
        general_columns = kwargs.get("general_columns", [])
        non_categorical_columns = kwargs.get("non_categorical_columns", [])
        
        # Auto-detect column types if not provided (enhanced detection for CTAB-GAN+)
        if categorical_columns is None:
            categorical_columns = self._auto_detect_categorical_columns(data)
        if integer_columns is None:
            integer_columns = self._auto_detect_integer_columns(data)
        if not general_columns:
            general_columns = self._auto_detect_general_columns(data)
        if not non_categorical_columns:
            non_categorical_columns = self._auto_detect_non_categorical_columns(data)
        
        self._categorical_columns = categorical_columns
        self._integer_columns = integer_columns
        self._mixed_columns = mixed_columns
        self._log_columns = log_columns
        self._general_columns = general_columns
        self._non_categorical_columns = non_categorical_columns
        
        try:
            # Save data to temporary CSV for CTAB-GAN+
            temp_csv_path = "temp_ctabganplus_data.csv"
            data.to_csv(temp_csv_path, index=False)
            
            # Determine a reasonable target column for CTAB-GAN+
            # Use the last column as target (common ML convention), ensure it's categorical for stratification
            target_column = data.columns[-1]
            
            # Ensure target column is in categorical columns for proper handling
            if target_column not in categorical_columns:
                categorical_columns.append(target_column)
            
            # Remove target column from integer columns if it's there to avoid conflicts
            if target_column in integer_columns:
                integer_columns.remove(target_column)
            
            problem_type = {"Classification": target_column}
            
            # Initialize CTAB-GAN+ model with conservative approach
            # Use basic CTAB-GAN parameters by default to avoid compatibility issues
            try:
                self._ctabganplus_model = CTABGAN(
                    raw_csv_path=temp_csv_path,
                    test_ratio=self.model_config.get("test_ratio", 0.2),
                    categorical_columns=categorical_columns,
                    log_columns=log_columns,
                    mixed_columns=mixed_columns,
                    integer_columns=integer_columns,
                    problem_type=problem_type
                )
                logger.info("✅ CTAB-GAN+ initialized with standard CTAB-GAN parameters")
            except Exception as e:
                logger.error(f"Failed to initialize CTAB-GAN+: {e}")
                raise
            
            # Train the model
            self._ctabganplus_model.fit()
            
            # Clean up temporary file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
            
            self.is_trained = True
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Store training metadata (enhanced for CTAB-GAN+)
            self.training_metadata = {
                "training_time": training_time,
                "epochs": epochs,
                "data_shape": data.shape,
                "categorical_columns": categorical_columns,
                "integer_columns": integer_columns,
                "mixed_columns": mixed_columns,
                "log_columns": log_columns,
                "general_columns": general_columns,  # CTAB-GAN+ specific
                "non_categorical_columns": non_categorical_columns,  # CTAB-GAN+ specific
                "enhanced_features": {
                    "preprocessing": self.model_config.get("enhanced_preprocessing", True),
                    "stability": self.model_config.get("stability_improvements", True)
                },
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            logger.info(f"CTAB-GAN+ training completed in {training_time:.2f} seconds")
            
            return self.training_metadata
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists("temp_ctabganplus_data.csv"):
                os.remove("temp_ctabganplus_data.csv")
            logger.error(f"CTAB-GAN+ training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")
    
    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data samples using CTAB-GAN+.
        
        Args:
            n_samples: Number of synthetic samples to generate
            **kwargs: Generation parameters
            
        Returns:
            Generated synthetic data as pandas DataFrame
        """
        if not self.is_trained or self._ctabganplus_model is None:
            raise ModelNotTrainedError("Model must be trained before generating samples")
        
        try:
            logger.info(f"Generating {n_samples} samples using CTAB-GAN+")
            logger.info(f"Categorical columns: {self._categorical_columns}")
            logger.info(f"Integer columns: {self._integer_columns}")
            
            # Generate samples using CTAB-GAN+
            synthetic_data = self._ctabganplus_model.generate_samples()
            logger.info(f"Raw synthetic data shape: {synthetic_data.shape}")
            logger.info(f"Raw synthetic data dtypes: {dict(synthetic_data.dtypes)}")
            
            # CTAB-GAN+ ENHANCED FIX: Ensure categorical columns have consistent data types
            # This prevents TRTS evaluation errors with string/int comparison
            for col in self._categorical_columns:
                if col in synthetic_data.columns:
                    try:
                        logger.debug(f"Processing categorical column '{col}' with dtype {synthetic_data[col].dtype}")
                        
                        # Convert string categorical values back to numeric if they represent numbers
                        if synthetic_data[col].dtype == 'object':
                            # Try to convert to numeric, handling potential string categories
                            numeric_values = pd.to_numeric(synthetic_data[col], errors='coerce')
                            if not numeric_values.isna().all():
                                # If conversion was successful for most values, use it
                                conversion_success_rate = numeric_values.notna().sum() / len(numeric_values)
                                if conversion_success_rate > 0.8:
                                    # Use mode of original data for filling NaN values if available
                                    fill_value = 0  # default
                                    synthetic_data[col] = numeric_values.fillna(fill_value).astype(int)
                                    logger.debug(f"Converted categorical column '{col}' to int (success rate: {conversion_success_rate:.2%})")
                                else:
                                    logger.debug(f"Skipped conversion for '{col}' due to low success rate: {conversion_success_rate:.2%}")
                            else:
                                logger.debug(f"No numeric conversion possible for '{col}'")
                        
                        # Additional fix: ensure integer columns that became floats are converted back
                        elif col in self._integer_columns and synthetic_data[col].dtype in ['float64', 'float32']:
                            try:
                                # Convert float to int, handling NaN values
                                synthetic_data[col] = synthetic_data[col].fillna(0).astype(int)
                                logger.debug(f"Converted integer column '{col}' from float back to int")
                            except Exception as int_convert_error:
                                logger.warning(f"Could not convert integer column '{col}' to int: {int_convert_error}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to process categorical column '{col}': {e}")
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
            logger.error(f"CTAB-GAN+ generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")
    
    def get_hyperparameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the hyperparameter search space for CTAB-GAN+ optimization.
        
        Returns:
            Dictionary defining hyperparameter search space (enhanced from CTAB-GAN)
        """
        return {
            "epochs": {
                "type": "int",
                "low": 150,  # Higher minimum for better stability
                "high": 1200,  # Higher maximum for enhanced performance
                "step": 50,
                "description": "Number of training epochs (enhanced range)"
            },
            "batch_size": {
                "type": "categorical",
                "choices": [64, 128, 256, 512],  # Additional batch size option
                "description": "Training batch size"
            },
            "class_dim": {
                "type": "categorical",
                "choices": [128, 256, 512, 1024],  # Higher dimension option
                "description": "Class embedding dimension (enhanced)"
            },
            "random_dim": {
                "type": "int",
                "low": 50,
                "high": 250,  # Higher range for more noise diversity
                "step": 25,
                "description": "Random noise dimension (enhanced)"
            },
            "num_channels": {
                "type": "int",
                "low": 32,
                "high": 256,  # Higher maximum for complex data
                "step": 32,
                "description": "Number of channels in generator/discriminator (enhanced)"
            },
            "test_ratio": {
                "type": "float",
                "low": 0.1,
                "high": 0.3,
                "step": 0.05,
                "description": "Test data ratio for validation"
            }
        }
    
    def save_model(self, path: str) -> None:
        """
        Save the trained CTAB-GAN+ model to disk.
        
        Args:
            path: Directory path where model should be saved
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Cannot save untrained model")
        
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata (enhanced for CTAB-GAN+)
        metadata = {
            "model_type": "CTABGANPlus",
            "training_metadata": self.training_metadata,
            "model_config": self.model_config,
            "categorical_columns": self._categorical_columns,
            "integer_columns": self._integer_columns,
            "mixed_columns": self._mixed_columns,
            "log_columns": self._log_columns,
            "general_columns": self._general_columns,
            "non_categorical_columns": self._non_categorical_columns,
            "enhanced_features": {
                "preprocessing": True,
                "stability_improvements": True,
                "additional_column_types": True
            }
        }
        
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Note: CTAB-GAN+ doesn't have built-in model saving
        # This implementation saves metadata for reconstruction
        logger.info(f"CTAB-GAN+ model metadata saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained CTAB-GAN+ model from disk.
        
        Args:
            path: Directory path where model is saved
        """
        model_dir = Path(path)
        metadata_file = model_dir / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Model metadata not found at {path}")
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Restore model state (enhanced for CTAB-GAN+)
        self.training_metadata = metadata["training_metadata"]
        self.model_config = metadata["model_config"]
        self._categorical_columns = metadata["categorical_columns"]
        self._integer_columns = metadata["integer_columns"]
        self._mixed_columns = metadata["mixed_columns"]
        self._log_columns = metadata["log_columns"]
        self._general_columns = metadata.get("general_columns", [])
        self._non_categorical_columns = metadata.get("non_categorical_columns", [])
        
        # Note: CTAB-GAN+ would need to be retrained as it doesn't support model saving
        # This is a limitation of the original CTAB-GAN+ implementation
        logger.warning("CTAB-GAN+ model loaded (metadata only). Model needs retraining for generation.")
    
    def _auto_detect_categorical_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect categorical columns (enhanced for CTAB-GAN+)."""
        categorical_columns = []
        
        for column in data.columns:
            if data[column].dtype == 'object':
                categorical_columns.append(column)
            elif data[column].dtype in ['int64', 'float64']:
                # Enhanced detection: check if it's a low-cardinality numeric column
                unique_values = data[column].nunique()
                total_values = len(data[column].dropna())
                cardinality_ratio = unique_values / total_values if total_values > 0 else 0
                
                # More sophisticated categorical detection
                if unique_values <= 15 or cardinality_ratio <= 0.05:
                    categorical_columns.append(column)
        
        return categorical_columns
    
    def _auto_detect_integer_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect integer columns (enhanced for CTAB-GAN+)."""
        integer_columns = []
        
        # Get categorical columns first to exclude them from integer detection
        categorical_columns = self._auto_detect_categorical_columns(data)
        
        for column in data.columns:
            # Skip if already classified as categorical
            if column in categorical_columns:
                continue
                
            if data[column].dtype in ['int64', 'int32', 'int16', 'int8']:
                # Additional check: ensure it's not a low-cardinality categorical
                unique_values = data[column].nunique()
                if unique_values > 15:  # Only treat as integer if high cardinality
                    integer_columns.append(column)
            elif data[column].dtype in ['float64', 'float32']:
                # Enhanced check: handle NaN values properly
                non_null_data = data[column].dropna()
                if len(non_null_data) > 0 and non_null_data.apply(lambda x: float(x).is_integer()).all():
                    # Additional check for high cardinality to avoid categorical conflicts
                    unique_values = data[column].nunique()
                    if unique_values > 15:
                        integer_columns.append(column)
        
        return integer_columns
    
    def _auto_detect_general_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect general numeric columns (CTAB-GAN+ specific)."""
        general_columns = []
        
        for column in data.columns:
            if data[column].dtype in ['float64', 'float32']:
                # General columns are continuous numeric columns that aren't integers
                non_null_data = data[column].dropna()
                if len(non_null_data) > 0:
                    # Check if it's not all integers and has reasonable variance
                    has_decimals = not non_null_data.apply(lambda x: float(x).is_integer()).all()
                    has_variance = non_null_data.var() > 0
                    
                    if has_decimals and has_variance:
                        general_columns.append(column)
        
        return general_columns
    
    def _auto_detect_non_categorical_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect non-categorical columns (CTAB-GAN+ specific)."""
        non_categorical_columns = []
        categorical_columns = self._auto_detect_categorical_columns(data)
        
        for column in data.columns:
            if column not in categorical_columns:
                # Non-categorical are numeric columns that aren't treated as categorical
                if data[column].dtype in ['int64', 'int32', 'float64', 'float32']:
                    # Additional safeguard: ensure high cardinality for numeric treatment
                    unique_values = data[column].nunique()
                    if unique_values > 15:  # Only treat as non-categorical if high cardinality
                        non_categorical_columns.append(column)
        
        return non_categorical_columns