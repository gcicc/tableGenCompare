"""
TVAE model implementation for the synthetic tabular data framework.

This module wraps the TVAE (Tabular Variational Autoencoder) model from SDV
to work with the unified framework interface. TVAE shows strong performance
in 2025 research as the second-best performing model after CTGAN.
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
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    TVAE_AVAILABLE = True
except ImportError:
    TVAE_AVAILABLE = False
    # Create dummy class for type hints when SDV not available
    class SingleTableMetadata:
        pass
    logger.warning("TVAE (SDV) not available. Install with: pip install sdv")


class TVAEModel(SyntheticDataModel):
    """
    TVAE model implementation for synthetic tabular data generation.
    
    TVAE (Tabular Variational Autoencoder) uses VAE-based neural network techniques
    to model tabular data. It extends conventional VAE with regularization in the 
    latent space and mode-specific normalization for handling non-Gaussian and 
    multimodal distributions. Based on 2025 research, TVAE is the second-best 
    performing model after CTGAN.
    """
    
    def __init__(self, device: str = "cpu", random_state: int = 42):
        """
        Initialize TVAE model wrapper.
        
        Args:
            device: Computing device (SDV handles device internally)
            random_state: Random seed for reproducibility
        """
        if not TVAE_AVAILABLE:
            raise ImportError(
                "TVAE (SDV) is not available. Please install it with: pip install sdv"
            )
        
        super().__init__(device, random_state)
        
        # TVAE-specific initialization
        self._tvae_model = None
        self._metadata = None
        self._training_history = None
        
        # Default TVAE parameters based on research best practices
        self.default_config = {
            "epochs": 300,
            "enforce_min_max_values": True,
            "enforce_rounding": False,
            "compress_dims": (128, 128),
            "decompress_dims": (128, 128),
            "l2scale": 1e-5,
            "batch_size": 500,
            "loss_factor": 2,
            "cuda": device.lower() == "cuda"
        }
        
        self.set_config(self.default_config)
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train the TVAE model on the provided dataset.
        
        Args:
            data: Training dataset as pandas DataFrame
            **kwargs: Training parameters (epochs, metadata, etc.)
            
        Returns:
            Dictionary containing training metadata and metrics
        """
        # Validate input data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise DataValidationError(f"Data validation failed: {error_msg}")
        
        # Extract training parameters
        epochs = kwargs.get("epochs", self.model_config.get("epochs", 300))
        enforce_min_max = kwargs.get("enforce_min_max_values", self.model_config.get("enforce_min_max_values", True))
        enforce_rounding = kwargs.get("enforce_rounding", self.model_config.get("enforce_rounding", False))
        custom_metadata = kwargs.get("metadata", None)
        
        logger.info(f"Starting TVAE training with {epochs} epochs")
        training_start = datetime.now()
        
        try:
            # Preprocess data for TVAE
            processed_data = self._preprocess_data(data)
            
            # Create or use custom metadata
            if custom_metadata is not None:
                self._metadata = custom_metadata
            else:
                self._metadata = self._create_metadata(processed_data)
            
            logger.info(f"Metadata created with {len(self._metadata.columns)} columns")
            
            # Initialize TVAE model with current configuration
            self._tvae_model = TVAESynthesizer(
                metadata=self._metadata,
                enforce_min_max_values=enforce_min_max,
                enforce_rounding=enforce_rounding,
                epochs=epochs,
                compress_dims=self.model_config["compress_dims"],
                decompress_dims=self.model_config["decompress_dims"],
                l2scale=self.model_config["l2scale"],
                batch_size=self.model_config["batch_size"],
                loss_factor=self.model_config["loss_factor"],
                cuda=self.model_config["cuda"]
            )
            
            # Train the model
            self._tvae_model.fit(processed_data)
            
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            
            # Extract training history if available
            training_losses = []
            if hasattr(self._tvae_model, '_model') and hasattr(self._tvae_model._model, 'loss_values'):
                raw_losses = self._tvae_model._model.loss_values
                # Handle case where loss_values might be a DataFrame
                if hasattr(raw_losses, 'values'):  # pandas DataFrame/Series
                    training_losses = raw_losses.values.flatten().tolist()
                elif isinstance(raw_losses, (list, tuple)):
                    training_losses = list(raw_losses)
                else:
                    training_losses = []
            
            # Update training metadata
            self.training_metadata = {
                "training_start": training_start.isoformat(),
                "training_end": training_end.isoformat(),
                "training_duration_seconds": training_duration,
                "epochs": epochs,
                "data_shape": data.shape,
                "data_columns": list(data.columns),
                "compress_dimensions": self.model_config["compress_dims"],
                "decompress_dimensions": self.model_config["decompress_dims"],
                "l2_regularization": self.model_config["l2scale"],
                "batch_size": self.model_config["batch_size"],
                "loss_factor": self.model_config["loss_factor"],
                "enforce_min_max_values": enforce_min_max,
                "enforce_rounding": enforce_rounding,
                "cuda_enabled": self.model_config["cuda"],
                "training_losses": training_losses[:100] if training_losses else [],  # Store first 100 for space
                "final_loss": training_losses[-1] if training_losses else None,
                "convergence_achieved": len(training_losses) < epochs if training_losses else False
            }
            
            self.is_trained = True
            logger.info(f"TVAE training completed in {training_duration:.2f} seconds")
            
            if training_losses:
                logger.info(f"Final training loss: {training_losses[-1]:.6f}")
            
            return self.training_metadata
            
        except Exception as e:
            logger.error(f"TVAE training failed: {e}")
            self.is_trained = False
            raise
    
    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data samples using trained TVAE model.
        
        Args:
            n_samples: Number of synthetic samples to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated synthetic data as pandas DataFrame
        """
        if not self.is_trained or self._tvae_model is None:
            raise ModelNotTrainedError("Model must be trained before generating data")
        
        logger.info(f"Generating {n_samples} synthetic samples with TVAE")
        generation_start = datetime.now()
        
        try:
            # Generate synthetic data using SDV TVAE
            synthetic_data = self._tvae_model.sample(num_rows=n_samples)
            
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
            logger.error(f"TVAE generation failed: {e}")
            raise
    
    def get_hyperparameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the hyperparameter search space for TVAE optimization.
        
        Returns:
            Dictionary defining hyperparameter search space
        """
        return {
            'epochs': {
                'type': 'int',
                'low': 100,
                'high': 1000,
                'step': 50,
                'description': 'Number of training epochs'
            },
            'compress_dims': {
                'type': 'categorical',
                'choices': [(64, 64), (128, 128), (256, 256), (128, 256), (256, 128)],
                'description': 'Encoder (compression) network dimensions'
            },
            'decompress_dims': {
                'type': 'categorical',
                'choices': [(64, 64), (128, 128), (256, 256), (128, 256), (256, 128)],
                'description': 'Decoder (decompression) network dimensions'
            },
            'l2scale': {
                'type': 'float',
                'low': 1e-6,
                'high': 1e-3,
                'log': True,
                'description': 'L2 regularization scale (log scale)'
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [100, 250, 500, 1000],
                'description': 'Training batch size'
            },
            'loss_factor': {
                'type': 'int',
                'low': 1,
                'high': 10,
                'step': 1,
                'description': 'Loss factor for VAE training'
            },
            'enforce_min_max_values': {
                'type': 'categorical',
                'choices': [True, False],
                'description': 'Enforce min/max value constraints'
            },
            'enforce_rounding': {
                'type': 'categorical',
                'choices': [True, False],
                'description': 'Enforce rounding for integer columns'
            }
        }
    
    def save_model(self, path: str) -> None:
        """
        Save the trained TVAE model to disk.
        
        Args:
            path: Directory path where model should be saved
        """
        if not self.is_trained or self._tvae_model is None:
            raise ModelNotTrainedError("No trained model to save")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save TVAE model using SDV's built-in save method
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"TVAE_Model_{timestamp}.pkl"
            model_file = save_path / model_name
            
            self._tvae_model.save(str(model_file))
            
            # Save metadata separately for reference
            metadata_file = save_path / f"{model_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                serializable_metadata = self._make_json_serializable(self.get_model_info())
                json.dump(serializable_metadata, f, indent=2)
            
            # Save SDV metadata
            sdv_metadata_file = save_path / f"{model_name}_sdv_metadata.json"
            with open(sdv_metadata_file, 'w') as f:
                json.dump(self._metadata.to_dict(), f, indent=2)
            
            logger.info(f"TVAE model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save TVAE model: {e}")
            raise
    
    def load_model(self, path: str) -> None:
        """
        Load a trained TVAE model from disk.
        
        Args:
            path: Directory path where model is saved
        """
        try:
            model_path = Path(path)
            
            # Find TVAE model file
            model_files = list(model_path.glob("TVAE_Model_*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No TVAE model files found in {path}")
            
            # Load the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            # Load TVAE model using SDV's built-in load method
            self._tvae_model = TVAESynthesizer.load(str(latest_model))
            
            # Load SDV metadata
            sdv_metadata_file = latest_model.with_name(latest_model.name.replace('.pkl', '_sdv_metadata.json'))
            if sdv_metadata_file.exists():
                with open(sdv_metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    self._metadata = SingleTableMetadata.load_from_dict(metadata_dict)
            
            # Load our training metadata
            metadata_file = latest_model.with_name(latest_model.name.replace('.pkl', '_metadata.json'))
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.training_metadata = json.load(f)
            
            self.is_trained = True
            logger.info(f"TVAE model loaded from {latest_model}")
            
        except Exception as e:
            logger.error(f"Failed to load TVAE model: {e}")
            raise
    
    def _create_metadata(self, data: pd.DataFrame) -> SingleTableMetadata:
        """
        Create SDV metadata from the dataset.
        
        Args:
            data: Input dataset
            
        Returns:
            SDV SingleTableMetadata object
        """
        try:
            # SDV can auto-detect metadata from the dataframe
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data)
            
            # Log detected column types for debugging
            column_info = {}
            for column_name, column_info_obj in metadata.columns.items():
                column_info[column_name] = column_info_obj.get('sdtype', 'unknown')
            
            logger.info(f"SDV metadata detected column types: {column_info}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to create SDV metadata: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for TVAE training.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data suitable for TVAE
        """
        processed_data = data.copy()
        
        # Handle missing values - SDV TVAE can handle some missing values with RDT transforms
        # but it's better to handle them explicitly
        if processed_data.isnull().any().any():
            logger.warning("Data contains missing values. Applying simple imputation.")
            for column in processed_data.columns:
                if processed_data[column].isnull().any():
                    if processed_data[column].dtype in ['float64', 'int64']:
                        processed_data[column].fillna(processed_data[column].median(), inplace=True)
                    else:
                        processed_data[column].fillna(processed_data[column].mode()[0], inplace=True)
        
        # Ensure reasonable data types
        for column in processed_data.columns:
            # Convert boolean to int for consistency
            if processed_data[column].dtype == 'bool':
                processed_data[column] = processed_data[column].astype('int64')
        
        return processed_data
    
    def _postprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process generated synthetic data.
        
        Args:
            data: Raw synthetic data from TVAE
            
        Returns:
            Post-processed synthetic data
        """
        # SDV TVAE handles most post-processing automatically through RDTs
        # We'll do minimal post-processing here
        processed_data = data.copy()
        
        # Ensure integer columns are properly typed
        for column in processed_data.columns:
            if processed_data[column].dtype == 'float64':
                # Check if all values are close to integers
                if processed_data[column].apply(lambda x: x == int(x) if pd.notnull(x) else True).all():
                    processed_data[column] = processed_data[column].astype('int64')
        
        return processed_data
    
    def get_training_losses(self) -> List[float]:
        """
        Get training loss history if available.
        
        Returns:
            List of loss values from training
        """
        if hasattr(self._tvae_model, '_model') and hasattr(self._tvae_model._model, 'loss_values'):
            return self._tvae_model._model.loss_values
        else:
            return self.training_metadata.get('training_losses', [])
    
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