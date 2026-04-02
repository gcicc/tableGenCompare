"""
TabDDPM model implementation — Diffusion-based synthetic tabular data generation.

Uses the Synthcity library's diffusion plugin to generate high-quality synthetic data.
TabDDPM (Tab Denoising Diffusion Probabilistic Model) is a state-of-the-art diffusion
model specifically designed for tabular data as of 2024-2025.

Phase 5 - April 2026
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import logging
import cloudpickle

from ..base_model import SyntheticDataModel, ModelNotTrainedError, DataValidationError

logger = logging.getLogger(__name__)

try:
    from synthcity.plugins import Plugins
    from synthcity.utils.data_loader import GenericDataLoader
    TABDDPM_AVAILABLE = True
except ImportError:
    TABDDPM_AVAILABLE = False
    logger.debug("TabDDPM not available. Install with: pip install synthcity")


class TabDDPMModel(SyntheticDataModel):
    """
    TabDDPM model implementation for synthetic tabular data generation.

    Wraps Synthcity's diffusion-based tabular model to work with the unified
    framework interface. Diffusion models are among the best-performing synthetic
    data generators as of 2024-2025, with strong fidelity and utility metrics.
    """

    def __init__(self, device: str = "cpu", random_state: int = 42):
        """
        Initialize TabDDPM model wrapper.

        Args:
            device: Computing device ("cpu", "cuda"). Synthcity uses torch internally.
            random_state: Random seed for reproducibility
        """
        if not TABDDPM_AVAILABLE:
            raise ImportError(
                "TabDDPM is not available. Please install it with: pip install synthcity"
            )

        super().__init__(device, random_state)

        # TabDDPM-specific initialization
        self._plugin = None
        self._loader_kwargs = {}
        self._training_history = None
        self._plugin_name = None  # Track actual plugin key used

        # Default TabDDPM parameters
        self.default_config = {
            "n_iter": 100,                # Diffusion iterations / epochs
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 64,
            "num_timesteps": 1000,        # Number of diffusion timesteps
            "dim_embed": 128,             # Embedding dimension
            "n_layers_hidden": 2,
            "n_units_hidden": 128,
        }

        self.set_config(self.default_config)

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train TabDDPM on the provided dataset.

        Args:
            data: Training dataset as pandas DataFrame
            **kwargs: Training parameters (n_iter, lr, batch_size, etc.)
                     discrete_columns: List of categorical column names
                     target_column: Target column name (optional, for sensitive feature handling)

        Returns:
            Dictionary containing training metadata
        """
        # Validate input
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise DataValidationError(f"Data validation failed: {error_msg}")

        # Extract kwargs
        n_iter = kwargs.get("n_iter", self.model_config.get("n_iter", 100))
        discrete_columns = kwargs.get("discrete_columns", [])
        target_column = kwargs.get("target_column", None)

        logger.info(f"Starting TabDDPM training with {n_iter} iterations")
        logger.info(f"Discrete columns: {discrete_columns}")
        training_start = datetime.now()

        try:
            # Build GenericDataLoader for Synthcity
            # Synthcity expects categorical columns to be marked via 'sensitive_features'
            sensitive_features = discrete_columns if discrete_columns else []

            loader = GenericDataLoader(
                data.copy(),
                target_column=target_column,
                sensitive_features=sensitive_features
            )

            # Probe for available diffusion plugin
            plugins = Plugins()
            available_plugins = plugins.list()
            logger.info(f"Available Synthcity plugins: {available_plugins}")

            # Try canonical names in order of preference
            plugin_names_to_try = ["ddpm", "tab_ddpm", "tabddpm"]
            self._plugin = None
            self._plugin_name = None

            for pname in plugin_names_to_try:
                if pname in available_plugins:
                    try:
                        self._plugin = plugins.get(pname)
                        self._plugin_name = pname
                        logger.info(f"Using Synthcity plugin: {pname}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to get plugin {pname}: {e}")
                        continue

            if self._plugin is None:
                raise RuntimeError(
                    f"No diffusion plugin found. Available: {available_plugins}. "
                    f"Expected one of: {plugin_names_to_try}"
                )

            # Train the plugin
            self._plugin.fit(loader)

            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()

            # Update metadata
            self.training_metadata = {
                "training_start": training_start.isoformat(),
                "training_end": training_end.isoformat(),
                "training_duration_seconds": training_duration,
                "n_iter": n_iter,
                "data_shape": data.shape,
                "data_columns": list(data.columns),
                "discrete_columns": discrete_columns,
                "target_column": target_column,
                "plugin_name": self._plugin_name,
                "plugin_type": "diffusion",
            }

            self.is_trained = True
            logger.info(f"TabDDPM training completed in {training_duration:.2f} seconds")

            return self.training_metadata

        except Exception as e:
            logger.error(f"TabDDPM training failed: {e}")
            self.is_trained = False
            raise

    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data samples using trained TabDDPM.

        Args:
            n_samples: Number of synthetic samples to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated synthetic data as pandas DataFrame
        """
        if not self.is_trained or self._plugin is None:
            raise ModelNotTrainedError("Model must be trained before generating data")

        logger.info(f"Generating {n_samples} synthetic samples with TabDDPM")
        generation_start = datetime.now()

        try:
            # Synthcity's generate returns a GenericDataLoader object
            synthetic_loader = self._plugin.generate(count=n_samples)

            # Extract DataFrame from loader
            if hasattr(synthetic_loader, 'dataframe'):
                synthetic_data = synthetic_loader.dataframe()
            elif hasattr(synthetic_loader, 'data'):
                synthetic_data = synthetic_loader.data
            else:
                # Fallback: try to convert directly to DataFrame
                synthetic_data = pd.DataFrame(synthetic_loader)

            generation_end = datetime.now()
            generation_duration = (generation_end - generation_start).total_seconds()

            # Post-process
            processed_synthetic_data = self._postprocess_data(synthetic_data)

            # Update metadata
            generation_metadata = {
                "generation_time_seconds": generation_duration,
                "samples_generated": len(processed_synthetic_data),
                "generation_rate_samples_per_sec": (
                    len(processed_synthetic_data) / generation_duration
                    if generation_duration > 0 else float('inf')
                ),
                "synthetic_data_shape": processed_synthetic_data.shape,
            }
            self.training_metadata.update(generation_metadata)

            logger.info(f"Generated {len(processed_synthetic_data)} samples in {generation_duration:.3f} seconds")

            # Sanitize and return
            return self.sanitize_output(processed_synthetic_data)

        except Exception as e:
            logger.error(f"TabDDPM generation failed: {e}")
            raise

    def get_hyperparameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the hyperparameter search space for TabDDPM optimization.

        Returns:
            Dictionary mapping hyperparameter names to search specs for Optuna
        """
        return {
            'n_iter': {
                'type': 'int',
                'low': 50,
                'high': 200,
                'step': 10,
                'default': 100,
                'description': 'Diffusion iterations / training epochs'
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [32, 64, 128, 256],
                'default': 64,
                'description': 'Batch size for training'
            },
            'lr': {
                'type': 'float',
                'low': 1e-4,
                'high': 1e-2,
                'log': True,
                'default': 1e-3,
                'description': 'Learning rate'
            },
            'weight_decay': {
                'type': 'float',
                'low': 1e-6,
                'high': 1e-4,
                'log': True,
                'default': 1e-5,
                'description': 'L2 weight decay for regularization'
            },
            'num_timesteps': {
                'type': 'int',
                'low': 500,
                'high': 1500,
                'step': 250,
                'default': 1000,
                'description': 'Number of diffusion timesteps'
            },
            'dim_embed': {
                'type': 'categorical',
                'choices': [64, 128, 256],
                'default': 128,
                'description': 'Embedding dimension'
            },
            'n_layers_hidden': {
                'type': 'int',
                'low': 1,
                'high': 4,
                'step': 1,
                'default': 2,
                'description': 'Number of hidden layers'
            },
            'n_units_hidden': {
                'type': 'categorical',
                'choices': [64, 128, 256, 512],
                'default': 128,
                'description': 'Number of units per hidden layer'
            },
        }

    def save_model(self, path: str) -> None:
        """
        Save the trained TabDDPM model to disk.

        Args:
            path: Directory path where model should be saved
        """
        if not self.is_trained or self._plugin is None:
            raise ModelNotTrainedError("No trained model to save")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save plugin using cloudpickle
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"TabDDPM_Model_{timestamp}.pkl"
            model_file = save_path / model_name

            with open(model_file, 'wb') as f:
                cloudpickle.dump(self._plugin, f)

            # Save metadata
            metadata_file = save_path / f"{model_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                serializable_metadata = self._make_json_serializable(self.get_model_info())
                json.dump(serializable_metadata, f, indent=2)

            logger.info(f"TabDDPM model saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save TabDDPM model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """
        Load a trained TabDDPM model from disk.

        Args:
            path: Directory path where model is saved
        """
        try:
            model_path = Path(path)

            # Find model file
            model_files = list(model_path.glob("TabDDPM_Model_*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No TabDDPM model files found in {path}")

            # Load the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

            with open(latest_model, 'rb') as f:
                self._plugin = cloudpickle.load(f)

            # Load metadata
            metadata_file = latest_model.with_name(latest_model.name.replace('.pkl', '_metadata.json'))
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.training_metadata = json.load(f)

            self.is_trained = True
            logger.info(f"TabDDPM model loaded from {latest_model}")

        except Exception as e:
            logger.error(f"Failed to load TabDDPM model: {e}")
            raise

    def _postprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process generated synthetic data.

        Args:
            data: Raw synthetic data from TabDDPM

        Returns:
            Post-processed synthetic data
        """
        processed_data = data.copy()

        # Ensure categorical columns have proper types
        for col in processed_data.columns:
            if col in self._loader_kwargs.get("discrete_columns", []):
                # Round to nearest integer for categorical columns that came out as floats
                if processed_data[col].dtype in ['float64', 'float32']:
                    processed_data[col] = processed_data[col].round().astype('int64')

        return processed_data

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert non-JSON-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
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
