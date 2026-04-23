"""
TabDiffusion model implementation — Diffusion-based synthetic tabular data generation.

Uses the Hugging Face diffusers library to implement a diffusion model for tabular data.
TabDiffusion applies diffusion probabilistic models to tabular synthesis, offering
state-of-the-art fidelity and utility.

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

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from ..base_model import SyntheticDataModel, ModelNotTrainedError, DataValidationError

logger = logging.getLogger(__name__)

try:
    from diffusers import DDPMScheduler
    TABDIFFUSION_AVAILABLE = True
except ImportError:
    TABDIFFUSION_AVAILABLE = False
    logger.debug("TabDiffusion not available. Install with: pip install diffusers")


class SimpleTabularDiffusionNetwork(nn.Module):
    """
    Simple denoising network for tabular diffusion.

    Processes concatenated features (numerical + categorical embeddings)
    with a series of dense layers and positional time embeddings.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Time embedding (sinusoidal)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Main network
        layers = []
        layers.append(nn.Linear(input_dim + hidden_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, input_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy features [batch_size, input_dim]
            t: Timesteps [batch_size] in range [0, 1]

        Returns:
            Predicted noise [batch_size, input_dim]
        """
        # Embed time
        t_emb = self.time_embedding(t.unsqueeze(-1))

        # Concatenate with input
        x_t = torch.cat([x, t_emb], dim=1)

        # Denoise
        return self.network(x_t)


class TabDiffusionModel(SyntheticDataModel):
    """
    TabDiffusion model implementation for synthetic tabular data generation.

    Uses diffusion probabilistic models (via diffusers library) to generate
    high-quality synthetic tabular data. Handles mixed data types (numerical
    and categorical) transparently.
    """

    def __init__(self, device: str = "cpu", random_state: int = 42):
        """
        Initialize TabDiffusion model wrapper.

        Args:
            device: Computing device ("cpu", "cuda")
            random_state: Random seed for reproducibility
        """
        if not TABDIFFUSION_AVAILABLE:
            raise ImportError(
                "TabDiffusion is not available. Please install diffusers: pip install diffusers"
            )

        super().__init__(device, random_state)

        # TabDiffusion-specific initialization
        self._model = None
        self._scheduler = None
        self._scaler = None
        self._categorical_columns = []
        self._categorical_encoders = {}
        self._numerical_columns = []
        self._training_history = None

        # Default TabDiffusion parameters
        self.default_config = {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "num_diffusion_steps": 1000,
            "hidden_dim": 128,
            "num_layers": 3,
            "noise_schedule": "linear",  # or "cosine"
        }

        self.set_config(self.default_config)

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train TabDiffusion on the provided dataset.

        Args:
            data: Training dataset as pandas DataFrame
            **kwargs: Training parameters (epochs, batch_size, learning_rate, etc.)
                     discrete_columns: List of categorical column names
                     target_column: Target column name (optional)

        Returns:
            Dictionary containing training metadata
        """
        # Validate input
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise DataValidationError(f"Data validation failed: {error_msg}")

        # Extract kwargs
        epochs = kwargs.get("epochs", self.model_config.get("epochs", 100))
        batch_size = kwargs.get("batch_size", self.model_config.get("batch_size", 64))
        learning_rate = kwargs.get("learning_rate", self.model_config.get("learning_rate", 1e-3))
        num_diffusion_steps = kwargs.get("num_diffusion_steps", self.model_config.get("num_diffusion_steps", 1000))
        hidden_dim = kwargs.get("hidden_dim", self.model_config.get("hidden_dim", 128))
        num_layers = kwargs.get("num_layers", self.model_config.get("num_layers", 3))
        discrete_columns = kwargs.get("discrete_columns", [])
        target_column = kwargs.get("target_column", None)

        logger.info(f"Starting TabDiffusion training for {epochs} epochs")
        logger.info(f"Dataset shape: {data.shape}, Categorical cols: {discrete_columns}")
        training_start = datetime.now()

        try:
            # Preprocess data
            data_processed = self._preprocess_data(data, discrete_columns)

            # Convert to tensor
            X = torch.FloatTensor(data_processed.values).to(self.device)
            dataset = TensorDataset(X)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Initialize model and scheduler
            self._model = SimpleTabularDiffusionNetwork(
                input_dim=X.shape[1],
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(self.device)

            self._scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_steps)
            # Move scheduler tensors to device (critical for CUDA support)
            self._scheduler.alphas_cumprod = self._scheduler.alphas_cumprod.to(self.device)
            self._scheduler.alphas = self._scheduler.alphas.to(self.device)
            optimizer = Adam(self._model.parameters(), lr=learning_rate)

            # Training loop
            self._model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch_idx, (x_batch,) in enumerate(dataloader):
                    x_batch = x_batch.to(self.device)

                    # Sample random timesteps
                    batch_size_actual = x_batch.shape[0]
                    t = torch.randint(0, num_diffusion_steps, (batch_size_actual,)).to(self.device)
                    t_normalized = t.float() / num_diffusion_steps

                    # Add noise
                    noise = torch.randn_like(x_batch)
                    sqrt_alpha_prod = torch.sqrt(
                        self._scheduler.alphas_cumprod[t].view(-1, 1)
                    )
                    sqrt_one_minus_alpha_prod = torch.sqrt(
                        1 - self._scheduler.alphas_cumprod[t].view(-1, 1)
                    )

                    x_noisy = sqrt_alpha_prod * x_batch + sqrt_one_minus_alpha_prod * noise

                    # Predict noise
                    noise_pred = self._model(x_noisy, t_normalized)

                    # Loss
                    loss = nn.MSELoss()(noise_pred, noise)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if (epoch + 1) % max(1, epochs // 10) == 0:
                    avg_loss = epoch_loss / len(dataloader)
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()

            # Update metadata
            self.training_metadata = {
                "training_start": training_start.isoformat(),
                "training_end": training_end.isoformat(),
                "training_duration_seconds": training_duration,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_diffusion_steps": num_diffusion_steps,
                "data_shape": data.shape,
                "data_columns": list(data.columns),
                "categorical_columns": discrete_columns,
                "numerical_columns": self._numerical_columns,
                "target_column": target_column,
            }

            self.is_trained = True
            logger.info(f"TabDiffusion training completed in {training_duration:.2f} seconds")

            return self.training_metadata

        except Exception as e:
            logger.error(f"TabDiffusion training failed: {e}")
            self.is_trained = False
            raise

    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data samples using trained TabDiffusion.

        Args:
            n_samples: Number of synthetic samples to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated synthetic data as pandas DataFrame
        """
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError("Model must be trained before generating data")

        logger.info(f"Generating {n_samples} synthetic samples with TabDiffusion")
        generation_start = datetime.now()

        try:
            self._model.eval()
            num_diffusion_steps = self.training_metadata.get("num_diffusion_steps", 1000)

            with torch.no_grad():
                # Start from pure noise
                x = torch.randn(n_samples, self._model.input_dim).to(self.device)

                # Reverse diffusion process
                for t_idx in range(num_diffusion_steps - 1, -1, -1):
                    t = torch.full((n_samples,), t_idx).to(self.device)
                    t_normalized = t.float() / num_diffusion_steps

                    # Predict noise
                    noise_pred = self._model(x, t_normalized)

                    # Update x
                    alpha = self._scheduler.alphas[t_idx].to(self.device)
                    alpha_prod = self._scheduler.alphas_cumprod[t_idx].to(self.device)
                    alpha_prod_prev = (
                        self._scheduler.alphas_cumprod[t_idx - 1].to(self.device)
                        if t_idx > 0
                        else torch.tensor(1.0, device=self.device)
                    )

                    beta = 1 - alpha
                    variance = (1 - alpha_prod_prev) / (1 - alpha_prod) * beta

                    mean = (x - beta / torch.sqrt(1 - alpha_prod) * noise_pred) / torch.sqrt(alpha)
                    x = mean + torch.sqrt(variance + 1e-7) * torch.randn_like(x)

            # Convert to numpy and denormalize
            synthetic_np = x.cpu().numpy()
            synthetic_data = pd.DataFrame(
                synthetic_np,
                columns=self._numerical_columns + self._categorical_columns
            )

            # Denormalize numerical columns
            if self._scaler is not None:
                numerical_data = synthetic_data[self._numerical_columns].values
                numerical_data = self._scaler.inverse_transform(numerical_data)
                synthetic_data[self._numerical_columns] = numerical_data

            generation_end = datetime.now()
            generation_duration = (generation_end - generation_start).total_seconds()

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
            logger.error(f"TabDiffusion generation failed: {e}")
            raise

    def get_hyperparameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the hyperparameter search space for TabDiffusion optimization.

        Returns:
            Dictionary mapping hyperparameter names to search specs for Optuna
        """
        return {
            'epochs': {
                'type': 'int',
                'low': 50,
                'high': 200,
                'step': 10,
                'default': 100,
                'description': 'Number of training epochs'
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [32, 64, 128, 256],
                'default': 64,
                'description': 'Batch size for training'
            },
            'learning_rate': {
                'type': 'float',
                'low': 1e-4,
                'high': 1e-2,
                'log': True,
                'default': 1e-3,
                'description': 'Learning rate for optimizer'
            },
            'num_diffusion_steps': {
                'type': 'int',
                'low': 500,
                'high': 1500,
                'step': 250,
                'default': 1000,
                'description': 'Number of diffusion timesteps'
            },
            'hidden_dim': {
                'type': 'categorical',
                'choices': [64, 128, 256],
                'default': 128,
                'description': 'Hidden dimension of denoising network'
            },
            'num_layers': {
                'type': 'int',
                'low': 2,
                'high': 5,
                'step': 1,
                'default': 3,
                'description': 'Number of layers in denoising network'
            },
        }

    def save_model(self, path: str) -> None:
        """
        Save the trained TabDiffusion model to disk.

        Args:
            path: Directory path where model should be saved
        """
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError("No trained model to save")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"TabDiffusion_Model_{timestamp}"

            # Save model state
            model_file = save_path / f"{model_name}.pkl"
            torch.save(self._model.state_dict(), str(model_file))

            # Save scheduler
            scheduler_file = save_path / f"{model_name}_scheduler.pkl"
            with open(scheduler_file, 'wb') as f:
                cloudpickle.dump(self._scheduler, f)

            # Save scaler and encoders
            scaler_file = save_path / f"{model_name}_scaler.pkl"
            with open(scaler_file, 'wb') as f:
                cloudpickle.dump(self._scaler, f)

            encoders_file = save_path / f"{model_name}_encoders.pkl"
            with open(encoders_file, 'wb') as f:
                cloudpickle.dump(self._categorical_encoders, f)

            # Save metadata
            metadata_file = save_path / f"{model_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                serializable_metadata = self._make_json_serializable(self.get_model_info())
                json.dump(serializable_metadata, f, indent=2)

            logger.info(f"TabDiffusion model saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save TabDiffusion model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """
        Load a trained TabDiffusion model from disk.

        Args:
            path: Directory path where model is saved
        """
        try:
            model_path = Path(path)

            # Find model files
            model_files = list(model_path.glob("TabDiffusion_Model_*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No TabDiffusion model files found in {path}")

            # Get the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            base_name = latest_model.name.replace('.pkl', '')

            # Load model state
            self._model = SimpleTabularDiffusionNetwork(input_dim=1)  # Dummy for loading
            self._model.load_state_dict(torch.load(str(latest_model), map_location=self.device))
            self._model.to(self.device)

            # Load scheduler
            scheduler_file = model_path / f"{base_name}_scheduler.pkl"
            if scheduler_file.exists():
                with open(scheduler_file, 'rb') as f:
                    self._scheduler = cloudpickle.load(f)

            # Load scaler
            scaler_file = model_path / f"{base_name}_scaler.pkl"
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    self._scaler = cloudpickle.load(f)

            # Load encoders
            encoders_file = model_path / f"{base_name}_encoders.pkl"
            if encoders_file.exists():
                with open(encoders_file, 'rb') as f:
                    self._categorical_encoders = cloudpickle.load(f)

            # Load metadata
            metadata_file = model_path / f"{base_name}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.training_metadata = json.load(f)

            self.is_trained = True
            logger.info(f"TabDiffusion model loaded from {latest_model}")

        except Exception as e:
            logger.error(f"Failed to load TabDiffusion model: {e}")
            raise

    def _preprocess_data(self, data: pd.DataFrame, discrete_columns: List[str]) -> pd.DataFrame:
        """
        Preprocess data for TabDiffusion training.

        Standardizes numerical columns and encodes categorical columns.

        Args:
            data: Raw input data
            discrete_columns: List of categorical column names

        Returns:
            Preprocessed data ready for training
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder

        processed_data = data.copy()
        self._categorical_columns = discrete_columns
        self._numerical_columns = [col for col in data.columns if col not in discrete_columns]

        # Standardize numerical columns
        if self._numerical_columns:
            self._scaler = StandardScaler()
            processed_data[self._numerical_columns] = self._scaler.fit_transform(
                processed_data[self._numerical_columns]
            )
        else:
            self._scaler = None

        # Encode categorical columns
        for col in self._categorical_columns:
            if col in processed_data.columns:
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                self._categorical_encoders[col] = le

        # Reorder columns: numerical first, then categorical
        processed_data = processed_data[self._numerical_columns + self._categorical_columns]

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
