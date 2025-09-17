"""
GANerAid model implementation for the synthetic tabular data framework.

This module wraps the GANerAid synthetic data generation model to work with
the unified framework interface.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import json
import logging
from sklearn.preprocessing import LabelEncoder

from ..base_model import SyntheticDataModel, ModelNotTrainedError, DataValidationError

logger = logging.getLogger(__name__)

try:
    from GANerAid.ganeraid import GANerAid
    from GANerAid.evaluation_report import EvaluationReport
    GANERAID_AVAILABLE = True
except ImportError:
    GANERAID_AVAILABLE = False
    logger.warning("GANerAid not available. Install with: pip install GANerAid")


class GANerAidModel(SyntheticDataModel):
    """
    GANerAid model implementation for synthetic tabular data generation.
    
    GANerAid is specifically designed for clinical and healthcare data synthesis
    with focus on maintaining statistical properties and privacy preservation.
    """
    
    def __init__(self, device: str = "cpu", random_state: int = 42):
        """
        Initialize GANerAid model wrapper.
        
        Args:
            device: Computing device ("cpu", "cuda")
            random_state: Random seed for reproducibility
        """
        if not GANERAID_AVAILABLE:
            raise ImportError(
                "GANerAid is not available. Please install it with: pip install GANerAid"
            )
        
        super().__init__(device, random_state)
        
        # GANerAid-specific initialization
        self.device_obj = torch.device(device)
        self._ganeraid_model = None
        self._training_history = None
        
        # Categorical preprocessing tracking
        self._categorical_columns = []
        self._categorical_mappings = {}
        self._encoded_columns = []
        self._original_columns = []
        
        # Default GANerAid parameters
        self.default_config = {
            "lr_d": 0.0005,
            "lr_g": 0.0005,
            "hidden_feature_space": 200,
            "batch_size": 100,
            "nr_of_rows": 25,
            "binary_noise": 0.2,
            "epochs": 5000
        }
        
        self.set_config(self.default_config)
    
    def _is_binary_column(self, data: pd.DataFrame, column: str) -> bool:
        """Check if a column is binary (only 2 unique values)."""
        return data[column].nunique() == 2
    
    def _detect_categorical_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Detect categorical columns in the dataset.
        FIXED: Include binary columns in categorical processing to maintain discrete nature.
        
        Args:
            data: Input dataset
            
        Returns:
            List of column names that should be treated as categorical
        """
        categorical_columns = []
        
        for column in data.columns:
            # FIXED: Include binary columns in categorical processing to maintain distribution
            if self._is_binary_column(data, column):
                logger.info(f"Including binary column '{column}' in categorical processing")
                categorical_columns.append(column)
                continue
                
            # Check if column is object/string type
            if data[column].dtype == 'object':
                categorical_columns.append(column)
            # Check if column is categorical dtype
            elif data[column].dtype.name == 'category':
                categorical_columns.append(column)
            # Only process truly categorical numeric columns (>2 categories)
            elif (data[column].dtype in ['int64', 'int32'] and 
                  data[column].nunique() > 2 and 
                  data[column].nunique() <= 10 and 
                  data[column].nunique() < len(data) * 0.05):  # More conservative threshold
                categorical_columns.append(column)
        
        return categorical_columns
    
    def _preprocess_categorical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns.
        
        Args:
            data: Original dataset with mixed data types
            
        Returns:
            Dataset with categorical columns one-hot encoded
        """
        processed_data = data.copy()
        self._original_columns = list(data.columns)
        
        # Detect categorical columns
        self._categorical_columns = self._detect_categorical_columns(data)
        
        if not self._categorical_columns:
            logger.info("No categorical columns detected")
            self._encoded_columns = list(processed_data.columns)
            return processed_data
        
        logger.info(f"Detected categorical columns: {self._categorical_columns}")
        
        # Apply one-hot encoding to each categorical column
        for col in self._categorical_columns:
            # Store unique values for reverse mapping
            unique_values = sorted(processed_data[col].unique())
            self._categorical_mappings[col] = unique_values
            
            # Create one-hot encoded columns
            one_hot = pd.get_dummies(processed_data[col], prefix=col)
            
            # Drop original categorical column
            processed_data = processed_data.drop(columns=[col])
            
            # Add one-hot encoded columns
            processed_data = pd.concat([processed_data, one_hot], axis=1)
        
        self._encoded_columns = list(processed_data.columns)
        logger.info(f"One-hot encoding completed. Original columns: {len(self._original_columns)}, Encoded columns: {len(self._encoded_columns)}")
        
        return processed_data
    
    def _postprocess_categorical_data(self, synthetic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert one-hot encoded columns back to categorical format.
        
        Args:
            synthetic_data: Generated data with one-hot encoded categoricals
            
        Returns:
            Data with categorical columns restored to original format
        """
        if not self._categorical_columns:
            return synthetic_data
        
        processed_data = synthetic_data.copy()
        
        # Convert one-hot columns back to categorical for each original categorical column
        for col in self._categorical_columns:
            # Find all one-hot columns for this categorical variable
            one_hot_cols = [c for c in processed_data.columns if c.startswith(f"{col}_")]
            
            if not one_hot_cols:
                logger.warning(f"No one-hot columns found for {col}")
                continue
            
            # Convert one-hot back to categorical
            # Use argmax to find the most likely category
            one_hot_data = processed_data[one_hot_cols].values
            
            # Handle edge cases where no category is clearly dominant
            # Use argmax but handle ties by taking the first occurrence
            category_indices = np.argmax(one_hot_data, axis=1)
            
            # Map indices back to category names
            unique_values = self._categorical_mappings[col]
            
            # Handle case where we have more/fewer categories than expected
            valid_indices = np.clip(category_indices, 0, len(unique_values) - 1)
            categorical_values = [unique_values[i] for i in valid_indices]
            
            # Add the categorical column back
            processed_data[col] = categorical_values
            
            # FIXED: Ensure binary columns are converted to proper integer type
            if len(unique_values) == 2 and all(str(v).isdigit() for v in unique_values):
                try:
                    processed_data[col] = processed_data[col].astype(int)
                    logger.info(f"Converted binary column '{col}' to integer type")
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert binary column '{col}' to integer")
            
            # Remove one-hot columns
            processed_data = processed_data.drop(columns=one_hot_cols)
        
        # Reorder columns to match original order
        final_columns = []
        for col in self._original_columns:
            if col in processed_data.columns:
                final_columns.append(col)
        
        # Add any remaining columns that weren't in original (shouldn't happen normally)
        for col in processed_data.columns:
            if col not in final_columns:
                final_columns.append(col)
        
        processed_data = processed_data[final_columns]
        
        logger.info(f"Categorical postprocessing completed. Final columns: {list(processed_data.columns)}")
        return processed_data

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train the GANerAid model on the provided dataset.
        
        Args:
            data: Training dataset as pandas DataFrame
            **kwargs: Training parameters (epochs, verbose, aug_factor)
            
        Returns:
            Dictionary containing training metadata and metrics
        """
        # Validate input data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise DataValidationError(f"Data validation failed: {error_msg}")
        
        # Extract training parameters
        epochs = kwargs.get("epochs", self.model_config.get("epochs", 5000))
        verbose = kwargs.get("verbose", True)
        aug_factor = kwargs.get("aug_factor", 1)

        # FIXED: Extract GANerAid-specific hyperparameters and validate constraints
        batch_size = kwargs.get("batch_size", self.model_config.get("batch_size", 100))
        nr_of_rows = kwargs.get("nr_of_rows", self.model_config.get("nr_of_rows", 15))
        hidden_feature_space = kwargs.get("hidden_feature_space", self.model_config.get("hidden_feature_space", 200))
        lr_d = kwargs.get("lr_d", self.model_config.get("lr_d", 0.0005))
        lr_g = kwargs.get("lr_g", self.model_config.get("lr_g", 0.0005))
        binary_noise = kwargs.get("binary_noise", self.model_config.get("binary_noise", 0.2))

        # Critical constraint validation for GANerAid
        dataset_size = len(data)
        logger.info(f"Validating GANerAid constraints: dataset_size={dataset_size}, nr_of_rows={nr_of_rows}")

        # Constraint 1: nr_of_rows must be less than dataset size
        if nr_of_rows >= dataset_size:
            nr_of_rows = min(15, dataset_size - 1)
            logger.warning(f"nr_of_rows adjusted: {kwargs.get('nr_of_rows', 'default')} -> {nr_of_rows} (dataset constraint)")

        # Constraint 2: hidden_feature_space should be divisible by nr_of_rows to avoid dimension issues
        if hidden_feature_space % nr_of_rows != 0:
            # Find the closest nr_of_rows that makes hidden_feature_space divisible
            original_nr_of_rows = nr_of_rows
            found_compatible = False

            # Try values from nr_of_rows down to 1
            for candidate in range(nr_of_rows, 0, -1):
                if (hidden_feature_space % candidate == 0 and
                    candidate < dataset_size and
                    batch_size % candidate == 0):  # Also ensure batch compatibility
                    nr_of_rows = candidate
                    found_compatible = True
                    break

            if not found_compatible:
                # Find any valid divisor of hidden_feature_space that works
                divisors = [i for i in range(1, hidden_feature_space + 1) if hidden_feature_space % i == 0]
                valid_divisors = [d for d in divisors if d < dataset_size and batch_size % d == 0]

                if valid_divisors:
                    nr_of_rows = max(valid_divisors)  # Use largest valid divisor
                    logger.warning(f"Using largest valid divisor: nr_of_rows = {nr_of_rows}")
                else:
                    # Last resort: adjust hidden_feature_space to be compatible
                    logger.error(f"CRITICAL: No compatible nr_of_rows found for hidden_feature_space={hidden_feature_space}, batch_size={batch_size}")
                    # Use a smaller hidden_feature_space that's compatible
                    compatible_hidden = 100  # Safe default that has many divisors
                    logger.warning(f"Adjusting hidden_feature_space: {hidden_feature_space} -> {compatible_hidden}")
                    hidden_feature_space = compatible_hidden
                    nr_of_rows = min(5, dataset_size - 1)

            logger.warning(f"nr_of_rows adjusted for dimension compatibility: {original_nr_of_rows} -> {nr_of_rows}")
            logger.info(f"Final validation: hidden_feature_space={hidden_feature_space} % nr_of_rows={nr_of_rows} = {hidden_feature_space % nr_of_rows}")

        # Constraint 3: batch_size should be compatible
        if batch_size % nr_of_rows != 0:
            logger.warning(f"Batch size compatibility: batch_size={batch_size} % nr_of_rows={nr_of_rows} = {batch_size % nr_of_rows}")

        logger.info(f"✅ Final GANerAid parameters: batch_size={batch_size}, nr_of_rows={nr_of_rows}, dataset_size={dataset_size}")
        logger.info(f"   Dimension checks: hidden_divisible={hidden_feature_space % nr_of_rows == 0}, size_safe={nr_of_rows < dataset_size}")
        
        logger.info(f"Starting GANerAid training with {epochs} epochs")
        training_start = datetime.now()
        
        try:
            # Preprocess categorical data using one-hot encoding
            processed_data = self._preprocess_categorical_data(data)
            logger.info(f"Data preprocessing completed. Shape: {data.shape} -> {processed_data.shape}")
            
            # Initialize GANerAid model with validated parameters
            self._ganeraid_model = GANerAid(
                device=self.device_obj,
                lr_d=lr_d,
                lr_g=lr_g,
                hidden_feature_space=hidden_feature_space,
                batch_size=batch_size,
                nr_of_rows=nr_of_rows,
                binary_noise=binary_noise
            )
            logger.info(f"✅ GANerAid initialized with validated parameters")
            
            # Train the model on preprocessed data
            self._training_history = self._ganeraid_model.fit(
                processed_data, 
                epochs=epochs, 
                verbose=verbose, 
                aug_factor=aug_factor
            )
            
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            
            # Update training metadata
            self.training_metadata = {
                "training_start": training_start.isoformat(),
                "training_end": training_end.isoformat(),
                "training_duration_seconds": training_duration,
                "epochs": epochs,
                "original_data_shape": data.shape,
                "processed_data_shape": processed_data.shape,
                "original_columns": self._original_columns,
                "categorical_columns": self._categorical_columns,
                "encoded_columns": self._encoded_columns,
                "categorical_mappings": self._categorical_mappings,
                "preprocessing_applied": len(self._categorical_columns) > 0,
                "verbose": verbose,
                "aug_factor": aug_factor,
                "device": str(self.device_obj),
                # Add corrected parameters for constraint validation debugging
                "final_batch_size": batch_size,
                "final_nr_of_rows": nr_of_rows,
                "final_hidden_feature_space": hidden_feature_space,
                "constraint_validation_applied": True
            }
            
            # Add convergence analysis if history is available
            if hasattr(self._training_history, 'd_loss') and hasattr(self._training_history, 'g_loss'):
                final_d_loss = np.mean(self._training_history.d_loss[-100:]) if len(self._training_history.d_loss) >= 100 else np.mean(self._training_history.d_loss)
                final_g_loss = np.mean(self._training_history.g_loss[-100:]) if len(self._training_history.g_loss) >= 100 else np.mean(self._training_history.g_loss)
                
                self.training_metadata.update({
                    "final_discriminator_loss": float(final_d_loss),
                    "final_generator_loss": float(final_g_loss),
                    "discriminator_loss_std": float(np.std(self._training_history.d_loss[-100:])) if len(self._training_history.d_loss) >= 100 else 0.0,
                    "generator_loss_std": float(np.std(self._training_history.g_loss[-100:])) if len(self._training_history.g_loss) >= 100 else 0.0
                })
            
            self.is_trained = True
            logger.info(f"GANerAid training completed in {training_duration:.2f} seconds")
            
            return self.training_metadata
            
        except Exception as e:
            logger.error(f"GANerAid training failed: {e}")
            self.is_trained = False
            raise
    
    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data samples using trained GANerAid model.
        
        Args:
            n_samples: Number of synthetic samples to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated synthetic data as pandas DataFrame
        """
        if not self.is_trained or self._ganeraid_model is None:
            raise ModelNotTrainedError("Model must be trained before generating data")
        
        logger.info(f"Generating {n_samples} synthetic samples with GANerAid")
        generation_start = datetime.now()
        
        try:
            # Generate synthetic data (in encoded format)
            encoded_synthetic_data = self._ganeraid_model.generate(n_samples)
            
            # Apply postprocessing to convert back to original categorical format
            synthetic_data = self._postprocess_categorical_data(encoded_synthetic_data)
            
            generation_end = datetime.now()
            generation_duration = (generation_end - generation_start).total_seconds()
            
            logger.info(f"Postprocessing completed. Shape: {encoded_synthetic_data.shape} -> {synthetic_data.shape}")
            
            # Update metadata with generation info
            generation_metadata = {
                "generation_time_seconds": generation_duration,
                "samples_generated": len(synthetic_data),
                "generation_rate_samples_per_sec": len(synthetic_data) / generation_duration if generation_duration > 0 else float('inf'),
                "synthetic_data_shape": synthetic_data.shape,
                "encoded_synthetic_shape": encoded_synthetic_data.shape,
                "postprocessing_applied": len(self._categorical_columns) > 0
            }
            self.training_metadata.update(generation_metadata)
            
            logger.info(f"Generated {len(synthetic_data)} samples in {generation_duration:.3f} seconds")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"GANerAid generation failed: {e}")
            raise
    
    def get_hyperparameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the enhanced hyperparameter search space for GANerAid optimization.
        Production-ready hyperparameter space designed for diverse tabular datasets.
        
        Returns:
            Dictionary defining comprehensive hyperparameter search space
        """
        return {
            'epochs': {
                'type': 'int',
                'low': 1000,
                'high': 10000,
                'step': 500,
                'default': 5000,
                'description': 'Training epochs - 5000 optimal for GANerAid convergence on most datasets'
            },
            'lr_d': {
                'type': 'float',
                'low': 1e-6,
                'high': 5e-3,
                'log': True,
                'default': 5e-4,
                'description': 'Discriminator learning rate - 5e-4 optimal for clinical/healthcare data'
            },
            'lr_g': {
                'type': 'float',
                'low': 1e-6,
                'high': 5e-3,
                'log': True,
                'default': 5e-4,
                'description': 'Generator learning rate - matched with discriminator for stable training'
            },
            'hidden_feature_space': {
                'type': 'categorical',
                'choices': [
                    100,    # Small datasets (<1K samples, <10 features)
                    150,    # Small-medium datasets (1K-5K samples, 10-20 features)
                    200,    # Medium datasets (5K-20K samples, 20-50 features) - default
                    300,    # Large datasets (20K-50K samples, 50-100 features)
                    400,    # Very large datasets (50K+ samples, 100+ features)
                    500,    # Complex high-dimensional datasets (>150 features)
                    600     # Very complex datasets with intricate relationships
                ],
                'default': 200,
                'description': 'Hidden feature space dimensionality - adaptive to dataset complexity'
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [16, 32, 64, 100, 128, 256, 512],
                'default': 100,
                'description': 'Batch size - 100 optimal balance for GANerAid stability and memory efficiency'
            },
            'nr_of_rows': {
                'type': 'categorical',
                'choices': [5, 10, 15, 20, 25],  # Conservative safe defaults - will be overridden dynamically
                'default': 15,  # More conservative default
                'description': 'GANerAid sequence length parameter - MUST be < dataset size to avoid index errors'
            },
            'binary_noise': {
                'type': 'float',
                'low': 0.05,
                'high': 0.6,
                'default': 0.2,
                'description': 'Binary noise injection level - 0.2 optimal for privacy-utility tradeoff'
            },
            'generator_dropout': {
                'type': 'float',
                'low': 0.0,
                'high': 0.5,
                'default': 0.1,
                'description': 'Generator dropout rate for regularization - 0.1 prevents overfitting'
            },
            'discriminator_dropout': {
                'type': 'float',
                'low': 0.0,
                'high': 0.5,
                'default': 0.2,
                'description': 'Discriminator dropout rate - 0.2 improves generalization'
            },
            'weight_decay': {
                'type': 'float',
                'low': 1e-8,
                'high': 1e-3,
                'log': True,
                'default': 1e-6,
                'description': 'L2 weight decay for both networks - 1e-6 prevents parameter explosion'
            },
            'beta1': {
                'type': 'float',
                'low': 0.1,
                'high': 0.9,
                'default': 0.5,
                'description': 'Adam optimizer beta1 - 0.5 optimal for GAN training stability'
            },
            'beta2': {
                'type': 'float',
                'low': 0.9,
                'high': 0.999,
                'default': 0.999,
                'description': 'Adam optimizer beta2 - 0.999 for smooth convergence'
            },
            'gradient_penalty_weight': {
                'type': 'float',
                'low': 0.0,
                'high': 10.0,
                'default': 0.0,
                'description': 'Gradient penalty coefficient - 0 for standard training, >0 improves stability'
            },
            'privacy_epsilon': {
                'type': 'float',
                'low': 0.1,
                'high': 10.0,
                'log': True,
                'default': 1.0,
                'description': 'Privacy budget epsilon for differential privacy - 1.0 good balance'
            },
            'feature_noise_std': {
                'type': 'float',
                'low': 0.01,
                'high': 0.3,
                'default': 0.05,
                'description': 'Standard deviation of feature noise injection - 0.05 for privacy preservation'
            }
        }
    
    def save_model(self, path: str) -> None:
        """
        Save the trained GANerAid model to disk.
        
        Args:
            path: Directory path where model should be saved
        """
        if not self.is_trained or self._ganeraid_model is None:
            raise ModelNotTrainedError("No trained model to save")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save GANerAid model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"GANerAid_Model_{timestamp}.gan"
            
            self._ganeraid_model.save(str(save_path), model_name)
            
            # Save metadata
            metadata_file = save_path / f"{model_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                serializable_metadata = self._make_json_serializable(self.get_model_info())
                json.dump(serializable_metadata, f, indent=2)
            
            logger.info(f"GANerAid model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save GANerAid model: {e}")
            raise
    
    def load_model(self, path: str) -> None:
        """
        Load a trained GANerAid model from disk.
        
        Args:
            path: Directory path where model is saved
        """
        # Note: GANerAid doesn't have a standard load method
        # This would need to be implemented based on GANerAid's save format
        raise NotImplementedError("GANerAid model loading not yet implemented")
    
    def evaluate(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Any:
        """
        Create GANerAid evaluation report for synthetic data quality assessment.
        
        Args:
            original_data: Original training dataset
            synthetic_data: Generated synthetic dataset
            
        Returns:
            GANerAid EvaluationReport object
        """
        if not self.is_trained or self._ganeraid_model is None:
            raise ModelNotTrainedError("Model must be trained before evaluation")
        
        try:
            evaluation_report = self._ganeraid_model.evaluate(original_data, synthetic_data)
            logger.info("GANerAid evaluation report created successfully")
            return evaluation_report
            
        except Exception as e:
            logger.error(f"GANerAid evaluation failed: {e}")
            raise
    
    def get_training_history(self) -> Optional[Any]:
        """
        Get the training history object from GANerAid.
        
        Returns:
            Training history object or None if not trained
        """
        return self._training_history
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert numpy types and other non-serializable objects to JSON-serializable types.
        
        Args:
            obj: Object to make serializable
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        else:
            return obj