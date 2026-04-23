"""
GReaT model implementation — LLM-based synthetic tabular data generation.

Uses the be_great library to generate synthetic tabular data using large language models.
GReaT (Generating tabulaR data with REcurrent Autoencoders and Transformers) leverages
the expressiveness of LLMs to produce high-quality, semantically coherent synthetic data.

Phase 5 - April 2026
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import logging
import tempfile
import torch

from ..base_model import SyntheticDataModel, ModelNotTrainedError, DataValidationError

logger = logging.getLogger(__name__)

try:
    from be_great import GReaT
    GREAT_AVAILABLE = True
except ImportError:
    GREAT_AVAILABLE = False
    logger.debug("GReaT not available. Install with: pip install be_great")


class GReaTModel(SyntheticDataModel):
    """
    GReaT model implementation for synthetic tabular data generation.

    Wraps the be_great library's LLM-based tabular data generator to work with
    the unified framework interface. GReaT uses transformer-based language models
    to generate realistic, high-utility synthetic tabular data.
    """

    def __init__(self, device: str = "cpu", random_state: int = 42):
        """
        Initialize GReaT model wrapper.

        Args:
            device: Computing device ("cpu", "cuda"). GReaT uses torch internally.
            random_state: Random seed for reproducibility
        """
        if not GREAT_AVAILABLE:
            raise ImportError(
                "GReaT is not available. Please install it with: pip install be_great"
            )

        super().__init__(device, random_state)

        # GReaT-specific initialization
        self._model = None
        self._training_history = None

        # Detect actual device available
        self.actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"GReaT using device: {self.actual_device}")

        # Default GReaT parameters
        self.default_config = {
            "llm": "distilgpt2",      # Lightweight HF model
            "batch_size": 32,
            "epochs": 25,
            "lr": 5e-4,
            "warmup_steps": 100,
            "verbose": True,
        }

        self.set_config(self.default_config)

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train GReaT on the provided dataset.

        Args:
            data: Training dataset as pandas DataFrame
            **kwargs: Training parameters (all optional, have defaults):
                     - llm: HuggingFace model name (default: distilgpt2)
                     - batch_size: Training batch size (default: 32)
                     - epochs: Number of training epochs (default: 25)
                     - lr: Learning rate (default: 5e-4)
                     - warmup_steps: Warmup steps (default: 100)
                     - target_column: Target column name (optional, for metadata only)

        Returns:
            Dictionary containing training metadata
        """
        # Validate input
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise DataValidationError(f"Data validation failed: {error_msg}")

        # Extract kwargs
        llm = kwargs.get("llm", self.model_config.get("llm", "distilgpt2"))
        batch_size = kwargs.get("batch_size", self.model_config.get("batch_size", 32))
        epochs = kwargs.get("epochs", self.model_config.get("epochs", 25))
        lr = kwargs.get("lr", self.model_config.get("lr", 5e-4))
        warmup_steps = kwargs.get("warmup_steps", self.model_config.get("warmup_steps", 100))
        target_column = kwargs.get("target_column", None)
        verbose = kwargs.get("verbose", self.model_config.get("verbose", True))

        logger.info(f"Starting GReaT training with {llm} on {self.actual_device}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
        training_start = datetime.now()

        try:
            # Initialize GReaT model
            # GReaT accepts training parameters as **kwargs in __init__
            self._model = GReaT(
                llm=llm,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=lr,
                warmup_steps=warmup_steps,
            )

            # Train on the data
            # GReaT handles categorical/numerical detection internally
            self._model.fit(data)

            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()

            # Update metadata
            self.training_metadata = {
                "training_start": training_start.isoformat(),
                "training_end": training_end.isoformat(),
                "training_duration_seconds": training_duration,
                "llm": llm,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr,
                "warmup_steps": warmup_steps,
                "data_shape": data.shape,
                "data_columns": list(data.columns),
                "target_column": target_column,
                "device": self.actual_device,
            }

            self.is_trained = True
            logger.info(f"GReaT training completed in {training_duration:.2f} seconds")

            return self.training_metadata

        except Exception as e:
            logger.error(f"GReaT training failed: {e}")
            self.is_trained = False
            raise

    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data samples using trained GReaT.

        Args:
            n_samples: Number of synthetic samples to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated synthetic data as pandas DataFrame
        """
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError("Model must be trained before generating data")

        logger.info(f"Generating {n_samples} synthetic samples with GReaT")
        generation_start = datetime.now()

        try:
            # GReaT's sample method returns a DataFrame directly
            synthetic_data = self._model.sample(n_samples)

            generation_end = datetime.now()
            generation_duration = (generation_end - generation_start).total_seconds()

            # Ensure DataFrame structure
            if not isinstance(synthetic_data, pd.DataFrame):
                synthetic_data = pd.DataFrame(synthetic_data)

            # Update metadata
            generation_metadata = {
                "generation_time_seconds": generation_duration,
                "samples_generated": len(synthetic_data),
                "generation_rate_samples_per_sec": (
                    len(synthetic_data) / generation_duration
                    if generation_duration > 0 else float('inf')
                ),
                "synthetic_data_shape": synthetic_data.shape,
            }
            self.training_metadata.update(generation_metadata)

            logger.info(f"Generated {len(synthetic_data)} samples in {generation_duration:.3f} seconds")

            # Sanitize and return
            return self.sanitize_output(synthetic_data)

        except Exception as e:
            logger.error(f"GReaT generation failed: {e}")
            raise

    def get_hyperparameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the hyperparameter search space for GReaT optimization.

        Returns:
            Dictionary mapping hyperparameter names to search specs for Optuna
        """
        return {
            'llm': {
                'type': 'categorical',
                'choices': ['distilgpt2', 'gpt2'],
                'default': 'distilgpt2',
                'description': 'Hugging Face LLM model name (distilgpt2 is faster, gpt2 more capable)'
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [16, 32, 64, 128],
                'default': 32,
                'description': 'Training batch size'
            },
            'epochs': {
                'type': 'int',
                'low': 10,
                'high': 50,
                'step': 5,
                'default': 25,
                'description': 'Number of training epochs'
            },
            'lr': {
                'type': 'float',
                'low': 1e-5,
                'high': 1e-3,
                'log': True,
                'default': 5e-4,
                'description': 'Learning rate for optimizer'
            },
            'warmup_steps': {
                'type': 'int',
                'low': 0,
                'high': 500,
                'step': 50,
                'default': 100,
                'description': 'Warmup steps for learning rate scheduler'
            },
        }

    def save_model(self, path: str) -> None:
        """
        Save the trained GReaT model to disk.

        Args:
            path: Directory path where model should be saved
        """
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError("No trained model to save")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            # GReaT provides a .save() method
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_dir = save_path / f"GReaT_Model_{timestamp}"
            model_dir.mkdir(parents=True, exist_ok=True)

            self._model.save(str(model_dir))

            # Save metadata
            metadata_file = save_path / f"GReaT_Model_{timestamp}_metadata.json"
            with open(metadata_file, 'w') as f:
                serializable_metadata = self._make_json_serializable(self.get_model_info())
                json.dump(serializable_metadata, f, indent=2)

            logger.info(f"GReaT model saved to {model_dir}")

        except Exception as e:
            logger.error(f"Failed to save GReaT model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """
        Load a trained GReaT model from disk.

        Args:
            path: Directory path where model is saved
        """
        try:
            model_path = Path(path)

            # Find model directories
            model_dirs = list(model_path.glob("GReaT_Model_*"))
            if not model_dirs:
                raise FileNotFoundError(f"No GReaT model directories found in {path}")

            # Load the most recent model
            latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)

            # GReaT provides a .load() class method
            self._model = GReaT.load(str(latest_model_dir))

            # Load metadata
            metadata_file = model_path / f"{latest_model_dir.name}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.training_metadata = json.load(f)

            self.is_trained = True
            logger.info(f"GReaT model loaded from {latest_model_dir}")

        except Exception as e:
            logger.error(f"Failed to load GReaT model: {e}")
            raise

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
