"""
GANerAid model implementation for the synthetic tabular data framework.

This module wraps the GANerAid synthetic data generation model to work with
the unified framework interface.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import json
import logging

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
        
        logger.info(f"Starting GANerAid training with {epochs} epochs")
        training_start = datetime.now()
        
        try:
            # Initialize GANerAid model with current configuration
            self._ganeraid_model = GANerAid(
                device=self.device_obj,
                lr_d=self.model_config["lr_d"],
                lr_g=self.model_config["lr_g"],  
                hidden_feature_space=self.model_config["hidden_feature_space"],
                batch_size=self.model_config["batch_size"],
                nr_of_rows=self.model_config["nr_of_rows"],
                binary_noise=self.model_config["binary_noise"]
            )
            
            # Train the model
            self._training_history = self._ganeraid_model.fit(
                data, 
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
                "data_shape": data.shape,
                "data_columns": list(data.columns),
                "verbose": verbose,
                "aug_factor": aug_factor,
                "device": str(self.device_obj)
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
            synthetic_data = self._ganeraid_model.generate(n_samples)
            
            generation_end = datetime.now()
            generation_duration = (generation_end - generation_start).total_seconds()
            
            # Update metadata with generation info
            generation_metadata = {
                "generation_time_seconds": generation_duration,
                "samples_generated": len(synthetic_data),
                "generation_rate_samples_per_sec": len(synthetic_data) / generation_duration if generation_duration > 0 else float('inf'),
                "synthetic_data_shape": synthetic_data.shape
            }
            self.training_metadata.update(generation_metadata)
            
            logger.info(f"Generated {len(synthetic_data)} samples in {generation_duration:.3f} seconds")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"GANerAid generation failed: {e}")
            raise
    
    def get_hyperparameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the hyperparameter search space for GANerAid optimization.
        
        Returns:
            Dictionary defining hyperparameter search space
        """
        return {
            'lr_d': {
                'type': 'float',
                'low': 1e-5,
                'high': 1e-3,
                'log': True,
                'description': 'Discriminator learning rate (log scale)'
            },
            'lr_g': {
                'type': 'float', 
                'low': 1e-5,
                'high': 1e-3,
                'log': True,
                'description': 'Generator learning rate (log scale)'
            },
            'hidden_feature_space': {
                'type': 'int',
                'low': 100,
                'high': 400,
                'step': 50,
                'description': 'Hidden feature dimensions'
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [32, 64, 100, 128, 256],
                'description': 'Training batch size'
            },
            'nr_of_rows': {
                'type': 'int',
                'low': 25,
                'high': 25,  # Limited to prevent dimension issues
                'step': 5,
                'description': 'Number of rows parameter'
            },
            'binary_noise': {
                'type': 'float',
                'low': 0.1,
                'high': 0.4,
                'log': False,
                'description': 'Binary noise level'
            },
            'epochs': {
                'type': 'int',
                'low': 1000,
                'high': 8000,
                'step': 500,
                'description': 'Training epochs'
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