"""
CTGAN model implementation for the synthetic tabular data framework.

This module wraps the CTGAN (Conditional Tabular GAN) model to work with
the unified framework interface. CTGAN is specifically designed for tabular
data with mixed data types and shows top performance in 2025 research.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import logging

from ..base_model import SyntheticDataModel, ModelNotTrainedError, DataValidationError

logger = logging.getLogger(__name__)

try:
    from ctgan import CTGAN
    CTGAN_AVAILABLE = True
except ImportError:
    CTGAN_AVAILABLE = False  
    logger.warning("CTGAN not available. Install with: pip install ctgan")


class CTGANModel(SyntheticDataModel):
    """
    CTGAN model implementation for synthetic tabular data generation.
    
    CTGAN (Conditional Tabular GAN) is a state-of-the-art deep generative model
    specifically designed for tabular data, handling mixed data types and 
    imbalanced categories effectively. Based on 2025 research, CTGAN shows
    the highest performance across multiple evaluation metrics.
    """
    
    def __init__(self, device: str = "cpu", random_state: int = 42):
        """
        Initialize CTGAN model wrapper.
        
        Args:
            device: Computing device ("cpu", "cuda") - CTGAN handles device internally
            random_state: Random seed for reproducibility
        """
        if not CTGAN_AVAILABLE:
            raise ImportError(
                "CTGAN is not available. Please install it with: pip install ctgan"
            )
        
        super().__init__(device, random_state)
        
        # CTGAN-specific initialization
        self._ctgan_model = None
        self._discrete_columns = []
        self._training_history = None
        
        # Default CTGAN parameters based on research best practices
        self.default_config = {
            "epochs": 300,
            "batch_size": 500,
            "generator_dim": (256, 256),
            "discriminator_dim": (256, 256),
            "generator_lr": 2e-4,
            "discriminator_lr": 2e-4,
            "discriminator_steps": 1,
            "log_frequency": True,
            "verbose": True,
            "pac": 10  # Pac size for discriminator
        }
        
        self.set_config(self.default_config)
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train the CTGAN model on the provided dataset.
        
        Args:
            data: Training dataset as pandas DataFrame
            **kwargs: Training parameters (epochs, discrete_columns, etc.)
            
        Returns:
            Dictionary containing training metadata and metrics
        """
        # Validate input data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise DataValidationError(f"Data validation failed: {error_msg}")
        
        # Extract training parameters
        epochs = kwargs.get("epochs", self.model_config.get("epochs", 300))
        discrete_columns = kwargs.get("discrete_columns", None)
        verbose = kwargs.get("verbose", self.model_config.get("verbose", True))
        
        # Auto-detect discrete columns if not provided
        if discrete_columns is None:
            discrete_columns = self._auto_detect_discrete_columns(data)
        
        self._discrete_columns = discrete_columns
        
        logger.info(f"Starting CTGAN training with {epochs} epochs")
        logger.info(f"Discrete columns detected: {discrete_columns}")
        training_start = datetime.now()
        
        try:
            # Preprocess data for CTGAN
            processed_data = self._preprocess_data(data)
            
            # Initialize CTGAN model with current configuration
            self._ctgan_model = CTGAN(
                epochs=epochs,
                batch_size=self.model_config["batch_size"],
                generator_dim=self.model_config["generator_dim"],
                discriminator_dim=self.model_config["discriminator_dim"],
                generator_lr=self.model_config["generator_lr"],
                discriminator_lr=self.model_config["discriminator_lr"],
                discriminator_steps=self.model_config["discriminator_steps"],
                log_frequency=self.model_config["log_frequency"],
                verbose=verbose,
                pac=self.model_config["pac"]
            )
            
            # Train the model
            self._ctgan_model.fit(processed_data, discrete_columns)
            
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            
            # Update training metadata
            self.training_metadata = {
                "training_start": training_start.isoformat(),
                "training_end": training_end.isoformat(),
                "training_duration_seconds": training_duration,
                "epochs": epochs,
                "data_shape": data.shape,
                "data_columns": list(data.columns),
                "discrete_columns": discrete_columns,
                "batch_size": self.model_config["batch_size"],
                "generator_dimensions": self.model_config["generator_dim"],
                "discriminator_dimensions": self.model_config["discriminator_dim"],
                "pac_size": self.model_config["pac"],
                "preprocessing_applied": True
            }
            
            self.is_trained = True
            logger.info(f"CTGAN training completed in {training_duration:.2f} seconds")
            
            return self.training_metadata
            
        except Exception as e:
            logger.error(f"CTGAN training failed: {e}")
            self.is_trained = False
            raise
    
    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data samples using trained CTGAN model.
        
        Args:
            n_samples: Number of synthetic samples to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated synthetic data as pandas DataFrame
        """
        if not self.is_trained or self._ctgan_model is None:
            raise ModelNotTrainedError("Model must be trained before generating data")
        
        logger.info(f"Generating {n_samples} synthetic samples with CTGAN")
        generation_start = datetime.now()
        
        try:
            synthetic_data = self._ctgan_model.sample(n_samples)
            
            generation_end = datetime.now()
            generation_duration = (generation_end - generation_start).total_seconds()
            
            # Post-process generated data
            processed_synthetic_data = self._postprocess_data(synthetic_data)
            
            # Update metadata with generation info
            generation_metadata = {
                "generation_time_seconds": generation_duration,
                "samples_generated": len(processed_synthetic_data),
                "generation_rate_samples_per_sec": len(processed_synthetic_data) / generation_duration if generation_duration > 0 else float('inf'),
                "synthetic_data_shape": processed_synthetic_data.shape,
                "postprocessing_applied": True
            }
            self.training_metadata.update(generation_metadata)
            
            logger.info(f"Generated {len(processed_synthetic_data)} samples in {generation_duration:.3f} seconds")
            return processed_synthetic_data
            
        except Exception as e:
            logger.error(f"CTGAN generation failed: {e}")
            raise
    
    def get_hyperparameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the enhanced hyperparameter search space for CTGAN optimization.
        Production-ready hyperparameter space designed for diverse tabular datasets.
        
        Returns:
            Dictionary defining comprehensive hyperparameter search space
        """
        return {
            'epochs': {
                'type': 'int',
                'low': 100,
                'high': 1000,
                'step': 50,
                'default': 300,
                'description': 'Training epochs - 300 optimal for most tabular datasets'
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [32, 64, 128, 256, 500, 1000],
                'default': 500,
                'description': 'Batch size - larger batches (500+) work well for CTGAN stability'
            },
            'generator_lr': {
                'type': 'float',
                'low': 5e-6,
                'high': 5e-3,
                'log': True,
                'default': 2e-4,
                'description': 'Generator learning rate - 2e-4 optimal for Adam optimizer'
            },
            'discriminator_lr': {
                'type': 'float',
                'low': 5e-6,
                'high': 5e-3,
                'log': True,
                'default': 2e-4,
                'description': 'Discriminator learning rate - balanced with generator'
            },
            'generator_dim': {
                'type': 'categorical',
                'choices': [
                    (128, 128),          # Small datasets (<1K samples, <20 features)
                    (256, 256),          # Medium datasets (1K-10K samples, 20-50 features) 
                    (512, 512),          # Large datasets (10K-100K samples, 50+ features)
                    (256, 512),          # Asymmetric for complex feature interactions
                    (512, 256),          # Bottleneck architecture for regularization
                    (128, 256, 128),     # Deep architecture for small-medium datasets
                    (256, 512, 256),     # Deep architecture for medium-large datasets
                    (512, 1024, 512)     # Deep architecture for very large/complex datasets
                ],
                'default': (256, 256),
                'description': 'Generator architecture - adaptive to dataset complexity'
            },
            'discriminator_dim': {
                'type': 'categorical',
                'choices': [
                    (128, 128),          # Matches small generator
                    (256, 256),          # Matches medium generator - most stable
                    (512, 512),          # Matches large generator
                    (256, 512),          # Stronger discriminator for hard datasets
                    (512, 256),          # Funnel architecture for feature selection
                    (128, 256, 128),     # Deep discriminator for small datasets
                    (256, 512, 256),     # Deep discriminator for medium datasets
                    (512, 1024, 512)     # Deep discriminator for complex datasets
                ],
                'default': (256, 256),
                'description': 'Discriminator architecture - balanced with generator complexity'
            },
            'pac': {
                'type': 'int',
                'low': 1,
                'high': 20,
                'step': 1,
                'default': 10,
                'description': 'PackedGAN discriminator group size - 10 optimal for most datasets'
            },
            'discriminator_steps': {
                'type': 'int',
                'low': 1,
                'high': 5,
                'step': 1,
                'default': 1,
                'description': 'Discriminator training steps per generator step - 1 for balanced training'
            },
            'generator_decay': {
                'type': 'float',
                'low': 1e-8,
                'high': 1e-4,
                'log': True,
                'default': 1e-6,
                'description': 'Generator L2 weight decay for regularization'
            },
            'discriminator_decay': {
                'type': 'float',
                'low': 1e-8,
                'high': 1e-4,
                'log': True,
                'default': 1e-6,
                'description': 'Discriminator L2 weight decay for regularization'
            }
        }
    
    def save_model(self, path: str) -> None:
        """
        Save the trained CTGAN model to disk.
        
        Args:
            path: Directory path where model should be saved
        """
        if not self.is_trained or self._ctgan_model is None:
            raise ModelNotTrainedError("No trained model to save")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save CTGAN model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"CTGAN_Model_{timestamp}.pkl"
            model_file = save_path / model_name
            
            # CTGAN uses pickle for model serialization
            self._ctgan_model.save(str(model_file))
            
            # Save metadata
            metadata_file = save_path / f"{model_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                serializable_metadata = self._make_json_serializable(self.get_model_info())
                json.dump(serializable_metadata, f, indent=2)
            
            # Save discrete columns info
            columns_file = save_path / f"{model_name}_discrete_columns.json"
            with open(columns_file, 'w') as f:
                json.dump(self._discrete_columns, f, indent=2)
            
            logger.info(f"CTGAN model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save CTGAN model: {e}")
            raise
    
    def load_model(self, path: str) -> None:
        """
        Load a trained CTGAN model from disk.
        
        Args:
            path: Directory path where model is saved
        """
        try:
            model_path = Path(path)
            
            # Find CTGAN model file
            model_files = list(model_path.glob("CTGAN_Model_*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No CTGAN model files found in {path}")
            
            # Load the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            # Load CTGAN model
            self._ctgan_model = CTGAN.load(str(latest_model))
            
            # Load discrete columns
            columns_file = latest_model.with_name(latest_model.name.replace('.pkl', '_discrete_columns.json'))
            if columns_file.exists():
                with open(columns_file, 'r') as f:
                    self._discrete_columns = json.load(f)
            
            # Load metadata
            metadata_file = latest_model.with_name(latest_model.name.replace('.pkl', '_metadata.json'))
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.training_metadata = json.load(f)
            
            self.is_trained = True
            logger.info(f"CTGAN model loaded from {latest_model}")
            
        except Exception as e:
            logger.error(f"Failed to load CTGAN model: {e}")
            raise
    
    def _auto_detect_discrete_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Automatically detect discrete columns in the dataset.
        
        Args:
            data: Input dataset
            
        Returns:
            List of column names that should be treated as discrete
        """
        discrete_columns = []
        
        for column in data.columns:
            # Check if column is categorical/object type
            if data[column].dtype == 'object' or data[column].dtype.name == 'category':
                discrete_columns.append(column)
            # Check if column is integer with limited unique values
            elif data[column].dtype in ['int64', 'int32'] and data[column].nunique() <= 20:
                discrete_columns.append(column)
            # Check if column appears to be binary
            elif data[column].nunique() == 2:
                discrete_columns.append(column)
        
        return discrete_columns
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for CTGAN training.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data suitable for CTGAN
        """
        processed_data = data.copy()
        
        # Handle missing values
        if processed_data.isnull().any().any():
            logger.warning("Data contains missing values. CTGAN requires complete data.")
            # Simple imputation strategy
            for column in processed_data.columns:
                if processed_data[column].isnull().any():
                    if processed_data[column].dtype in ['float64', 'int64']:
                        processed_data[column].fillna(processed_data[column].median(), inplace=True)
                    else:
                        processed_data[column].fillna(processed_data[column].mode()[0], inplace=True)
        
        # Ensure discrete columns are properly typed
        for column in self._discrete_columns:
            if column in processed_data.columns:
                if processed_data[column].dtype == 'float64':
                    processed_data[column] = processed_data[column].astype('int64')
        
        return processed_data
    
    def _postprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process generated synthetic data.
        
        Args:
            data: Raw synthetic data from CTGAN
            
        Returns:
            Post-processed synthetic data
        """
        processed_data = data.copy()
        
        # Ensure discrete columns maintain proper types
        for column in self._discrete_columns:
            if column in processed_data.columns:
                # Round discrete numerical columns
                if processed_data[column].dtype in ['float64', 'float32']:
                    processed_data[column] = processed_data[column].round().astype('int64')
        
        return processed_data
    
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
        elif isinstance(obj, tuple):
            return list(obj)  # Convert tuples to lists for JSON
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