"""
MEDGAN model implementation for the synthetic tabular data framework.

This module wraps the MEDGAN (Medical Generative Adversarial Network) model
to work with the unified framework interface. MEDGAN is designed specifically
for generating synthetic medical/health records with discrete features.

Phase 5 - January 2026
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

# Try to import MEDGAN dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    MEDGAN_AVAILABLE = True
except ImportError:
    MEDGAN_AVAILABLE = False
    logger.warning("MEDGAN dependencies not available. Install with: pip install torch")


class MEDGANModel(SyntheticDataModel):
    """
    MEDGAN model implementation for synthetic medical data generation.

    MEDGAN is specifically designed for generating synthetic medical records,
    particularly suited for discrete/binary features common in healthcare data.
    It uses an autoencoder to learn a continuous representation of discrete data,
    then trains a GAN in that latent space.

    Key features:
    - Autoencoder for handling discrete/binary features
    - Pre-training phase for stable latent representations
    - Suitable for EHR (Electronic Health Records) data
    - Handles high-dimensional sparse binary data
    """

    def __init__(self, device: str = "cpu", random_state: int = 42):
        """
        Initialize MEDGAN model wrapper.

        Args:
            device: Computing device ("cpu", "cuda")
            random_state: Random seed for reproducibility
        """
        if not MEDGAN_AVAILABLE:
            raise ImportError(
                "MEDGAN dependencies not available. "
                "Please install with: pip install torch"
            )

        super().__init__(device, random_state)

        # MEDGAN specific initialization
        self._autoencoder = None
        self._generator = None
        self._discriminator = None
        self._discrete_columns = []
        self._data_dim = None
        self._latent_dim = None

        # Default MEDGAN parameters
        self.default_config = {
            "epochs": 300,
            "pretrain_epochs": 100,
            "batch_size": 128,
            "autoencoder_dim": (128, 128),
            "generator_dim": (128, 128),
            "discriminator_dim": (256, 128),
            "latent_dim": 128,
            "autoencoder_lr": 1e-3,
            "generator_lr": 1e-3,
            "discriminator_lr": 1e-3,
            "l2_reg": 1e-3,
            "bn_decay": 0.99,
            "verbose": True
        }

        self.set_config(self.default_config)

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train the MEDGAN model on the provided dataset.

        Training consists of two phases:
        1. Pretrain autoencoder to learn latent representations
        2. Train GAN in the latent space

        Args:
            data: Training dataset as pandas DataFrame
            **kwargs: Training parameters

        Returns:
            Dictionary containing training metadata
        """
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise DataValidationError(f"Data validation failed: {error_msg}")

        # Extract parameters
        epochs = kwargs.get("epochs", self.model_config.get("epochs", 300))
        pretrain_epochs = kwargs.get("pretrain_epochs", self.model_config.get("pretrain_epochs", 100))
        discrete_columns = kwargs.get("discrete_columns", None)

        if discrete_columns is None:
            discrete_columns = self._auto_detect_discrete_columns(data)

        self._discrete_columns = discrete_columns

        logger.info(f"Starting MEDGAN training: {pretrain_epochs} pretrain + {epochs} GAN epochs")
        training_start = datetime.now()

        try:
            # Preprocess data
            processed_data = self._preprocess_data(data)
            self._data_dim = processed_data.shape[1]
            self._latent_dim = self.model_config["latent_dim"]

            # Initialize networks
            self._initialize_networks()

            # Phase 1: Pretrain autoencoder
            logger.info("Phase 1: Pretraining autoencoder...")
            self._pretrain_autoencoder(processed_data, pretrain_epochs)

            # Phase 2: Train GAN
            logger.info("Phase 2: Training GAN in latent space...")
            self._train_gan(processed_data, epochs)

            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()

            self.training_metadata = {
                "training_start": training_start.isoformat(),
                "training_end": training_end.isoformat(),
                "training_duration_seconds": training_duration,
                "epochs": epochs,
                "pretrain_epochs": pretrain_epochs,
                "data_shape": data.shape,
                "data_columns": list(data.columns),
                "discrete_columns": discrete_columns,
                "latent_dim": self._latent_dim,
                "data_dim": self._data_dim
            }

            self.is_trained = True
            logger.info(f"MEDGAN training completed in {training_duration:.2f} seconds")

            return self.training_metadata

        except Exception as e:
            logger.error(f"MEDGAN training failed: {e}")
            raise

    def _initialize_networks(self):
        """Initialize autoencoder, generator, and discriminator networks."""
        device = torch.device(self.device)
        ae_dim = self.model_config["autoencoder_dim"]
        gen_dim = self.model_config["generator_dim"]
        disc_dim = self.model_config["discriminator_dim"]

        # Autoencoder (encoder + decoder)
        self._autoencoder = Autoencoder(
            input_dim=self._data_dim,
            hidden_dims=ae_dim,
            latent_dim=self._latent_dim
        ).to(device)

        # Generator: latent noise -> latent code
        self._generator = Generator(
            noise_dim=self._latent_dim,
            hidden_dims=gen_dim,
            output_dim=self._latent_dim
        ).to(device)

        # Discriminator: latent code -> real/fake
        self._discriminator = Discriminator(
            input_dim=self._latent_dim,
            hidden_dims=disc_dim
        ).to(device)

    def _pretrain_autoencoder(self, data: np.ndarray, epochs: int):
        """Pretrain the autoencoder for stable latent representations."""
        device = torch.device(self.device)
        batch_size = self.model_config["batch_size"]
        optimizer = torch.optim.Adam(
            self._autoencoder.parameters(),
            lr=self.model_config["autoencoder_lr"],
            weight_decay=self.model_config["l2_reg"]
        )

        n_samples = len(data)
        n_batches = max(1, n_samples // batch_size)

        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = np.random.permutation(n_samples)

            for i in range(n_batches):
                batch_idx = indices[i * batch_size:(i + 1) * batch_size]
                batch = torch.FloatTensor(data[batch_idx]).to(device)

                optimizer.zero_grad()
                reconstructed, _ = self._autoencoder(batch)

                # Reconstruction loss (binary cross-entropy for discrete data)
                loss = F.binary_cross_entropy(reconstructed, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if epoch % 20 == 0 and self.model_config.get("verbose", True):
                logger.info(f"Pretrain Epoch {epoch}: Loss={epoch_loss / n_batches:.4f}")

    def _train_gan(self, data: np.ndarray, epochs: int):
        """Train GAN in the latent space of the autoencoder."""
        device = torch.device(self.device)
        batch_size = self.model_config["batch_size"]

        g_optimizer = torch.optim.Adam(
            self._generator.parameters(),
            lr=self.model_config["generator_lr"]
        )
        d_optimizer = torch.optim.Adam(
            self._discriminator.parameters(),
            lr=self.model_config["discriminator_lr"]
        )

        n_samples = len(data)
        n_batches = max(1, n_samples // batch_size)

        for epoch in range(epochs):
            g_loss_epoch = 0.0
            d_loss_epoch = 0.0
            indices = np.random.permutation(n_samples)

            for i in range(n_batches):
                batch_idx = indices[i * batch_size:(i + 1) * batch_size]
                real_batch = torch.FloatTensor(data[batch_idx]).to(device)

                # Get real latent codes from autoencoder
                with torch.no_grad():
                    _, real_latent = self._autoencoder(real_batch)

                # Generate fake latent codes
                noise = torch.randn(len(batch_idx), self._latent_dim).to(device)
                fake_latent = self._generator(noise)

                # Train Discriminator
                d_optimizer.zero_grad()
                real_pred = self._discriminator(real_latent)
                fake_pred = self._discriminator(fake_latent.detach())
                d_loss = -torch.mean(torch.log(real_pred + 1e-8) + torch.log(1 - fake_pred + 1e-8))
                d_loss.backward()
                d_optimizer.step()

                # Train Generator
                g_optimizer.zero_grad()
                fake_pred = self._discriminator(fake_latent)
                g_loss = -torch.mean(torch.log(fake_pred + 1e-8))
                g_loss.backward()
                g_optimizer.step()

                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()

            if epoch % 50 == 0 and self.model_config.get("verbose", True):
                logger.info(
                    f"GAN Epoch {epoch}: D_loss={d_loss_epoch / n_batches:.4f}, "
                    f"G_loss={g_loss_epoch / n_batches:.4f}"
                )

    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data samples.

        Args:
            n_samples: Number of synthetic samples to generate
            **kwargs: Generation parameters

        Returns:
            Generated synthetic data as pandas DataFrame
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Model must be trained before generating data")

        logger.info(f"Generating {n_samples} synthetic samples with MEDGAN")

        device = torch.device(self.device)
        self._generator.eval()
        self._autoencoder.eval()

        with torch.no_grad():
            # Generate in latent space
            noise = torch.randn(n_samples, self._latent_dim).to(device)
            fake_latent = self._generator(noise)

            # Decode to data space
            synthetic_data = self._autoencoder.decode(fake_latent).cpu().numpy()

        # Post-process
        synthetic_df = self._postprocess_data(synthetic_data)

        # Apply sanitization
        from src.data.target_integrity import sanitize_numeric
        synthetic_df = sanitize_numeric(synthetic_df, verbose=False)

        return synthetic_df

    def get_hyperparameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the hyperparameter search space for MEDGAN optimization.
        Practical guidance for 1k–5k rows:
          - batch_size: include smaller values; 512 is often too large unless N is near the top of your range.
          - learning rates: 1e-2 is frequently unstable for GAN portions; narrow to a more typical stable band.
          - latent_dim / network sizes: keep modest to avoid overfitting and AE dominating.
          - l2_reg: very large regularization can underfit; cap upper range.
        """
        return {
            "epochs": {
                "type": "int",
                "low": 200,
                "high": 500,
                "step": 50,
                "default": 300,
                "description": "GAN training epochs (use Optuna pruning if available)"
            },
            "pretrain_epochs": {
                "type": "int",
                "low": 50,
                "high": 200,
                "step": 25,
                "default": 100,
                "description": "Autoencoder pretraining epochs (pretrain helps a lot on small tabular data)"
            },
            "batch_size": {
                "type": "categorical",
                "choices": [32, 64, 128, 256, 512],
                "default": 64,
                "description": "Training batch size (for 1k–5k rows: 32/64/128 usually best; avoid 512 unless N is large enough)"
            },
            "latent_dim": {
                "type": "int",
                "low": 32,
                "high": 192,
                "step": 32,
                "default": 96,
                "description": "Latent space dimensionality (smaller often generalizes better on 1k–5k rows)"
            },
            "autoencoder_dim": {
                "type": "categorical",
                "choices": [(64, 64), (128, 128), (256, 128), (128, 256, 128)],
                "default": (128, 128),
                "description": "Autoencoder hidden dimensions (keep modest for small data)"
            },
            "generator_dim": {
                "type": "categorical",
                "choices": [(64, 64), (128, 128), (256, 128), (128, 256, 128)],
                "default": (128, 128),
                "description": "Generator hidden dimensions (avoid over-large nets on small data)"
            },
            "discriminator_dim": {
                "type": "categorical",
                "choices": [(128, 64), (256, 128), (256, 256), (256, 128, 64)],
                "default": (256, 128),
                "description": "Discriminator hidden dimensions (keep balanced; overly strong discriminator can destabilize)"
            },
            "autoencoder_lr": {
                "type": "float",
                "low": 3e-4,
                "high": 3e-3,
                "log": True,
                "default": 1e-3,
                "description": "Autoencoder learning rate (narrowed for stability)"
            },
            "generator_lr": {
                "type": "float",
                "low": 1e-4,
                "high": 3e-3,
                "log": True,
                "default": 5e-4,
                "description": "Generator learning rate (narrowed; 1e-2 often unstable)"
            },
            "discriminator_lr": {
                "type": "float",
                "low": 1e-4,
                "high": 3e-3,
                "log": True,
                "default": 5e-4,
                "description": "Discriminator learning rate (narrowed; consider <= generator_lr if unstable)"
            },
            "l2_reg": {
                "type": "float",
                "low": 1e-6,
                "high": 1e-3,
                "log": True,
                "default": 1e-4,
                "description": "L2 regularization weight (tightened; too large can underfit on small data)"
            }
        }

    def save_model(self, path: str) -> None:
        """Save the trained MEDGAN model to disk."""
        if not self.is_trained:
            raise ModelNotTrainedError("Cannot save untrained model")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save networks
        torch.save(self._autoencoder.state_dict(), save_path / "autoencoder.pt")
        torch.save(self._generator.state_dict(), save_path / "generator.pt")
        torch.save(self._discriminator.state_dict(), save_path / "discriminator.pt")

        # Save metadata
        metadata = {
            "model_type": "MEDGAN",
            "training_metadata": self.training_metadata,
            "model_config": self.model_config,
            "discrete_columns": self._discrete_columns,
            "data_dim": self._data_dim,
            "latent_dim": self._latent_dim,
            "column_names": self._column_names
        }

        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"MEDGAN model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a trained MEDGAN model from disk."""
        load_path = Path(path)

        # Load metadata
        with open(load_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.training_metadata = metadata["training_metadata"]
        self.model_config = metadata["model_config"]
        self._discrete_columns = metadata["discrete_columns"]
        self._data_dim = metadata["data_dim"]
        self._latent_dim = metadata["latent_dim"]
        self._column_names = metadata["column_names"]

        # Initialize and load networks
        self._initialize_networks()
        self._autoencoder.load_state_dict(
            torch.load(load_path / "autoencoder.pt", map_location=self.device)
        )
        self._generator.load_state_dict(
            torch.load(load_path / "generator.pt", map_location=self.device)
        )
        self._discriminator.load_state_dict(
            torch.load(load_path / "discriminator.pt", map_location=self.device)
        )

        self.is_trained = True
        logger.info(f"MEDGAN model loaded from {path}")

    def _auto_detect_discrete_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect discrete columns based on data types and cardinality."""
        discrete_columns = []
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                discrete_columns.append(col)
            elif data[col].dtype in ['int64', 'int32'] and data[col].nunique() <= 20:
                discrete_columns.append(col)
            elif data[col].dtype == 'bool':
                discrete_columns.append(col)
        return discrete_columns

    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess data for MEDGAN training."""
        processed = data.copy()

        # Encode categorical columns
        for col in self._discrete_columns:
            if col in processed.columns and processed[col].dtype == 'object':
                processed[col] = pd.factorize(processed[col])[0]

        # Fill missing values
        processed = processed.fillna(processed.median(numeric_only=True))
        for col in processed.select_dtypes(include=['object']).columns:
            mode_val = processed[col].mode()
            processed[col] = processed[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')

        # Store normalization parameters before normalizing
        self._norm_params = {}
        numeric_cols = processed.select_dtypes(include=[np.number]).columns

        # Normalize to [0, 1] for sigmoid output
        for col in numeric_cols:
            col_min, col_max = processed[col].min(), processed[col].max()
            self._norm_params[col] = {'min': col_min, 'max': col_max}
            if col_max > col_min:
                processed[col] = (processed[col] - col_min) / (col_max - col_min)
            else:
                processed[col] = 0.0

        # Store for postprocessing
        self._column_names = list(data.columns)
        self._original_dtypes = data.dtypes.to_dict()

        return processed.values.astype(np.float32)

    def _postprocess_data(self, data: np.ndarray) -> pd.DataFrame:
        """Postprocess generated data back to DataFrame format."""
        # Clip to [0, 1] range (sigmoid output)
        data = np.clip(data, 0, 1)
        df = pd.DataFrame(data, columns=self._column_names)

        # Denormalize numeric columns back to original scale
        if hasattr(self, '_norm_params') and self._norm_params:
            for col in df.columns:
                if col in self._norm_params:
                    p = self._norm_params[col]
                    col_range = p['max'] - p['min']
                    if col_range > 0:
                        df[col] = df[col] * col_range + p['min']
                    else:
                        df[col] = p['min']

        return df


# ============================================================================
# NEURAL NETWORK COMPONENTS
# ============================================================================

if MEDGAN_AVAILABLE:
    class Autoencoder(nn.Module):
        """Autoencoder for learning latent representations of discrete data."""

        def __init__(self, input_dim: int, hidden_dims: tuple, latent_dim: int):
            super().__init__()

            # Encoder
            encoder_layers = []
            in_dim = input_dim
            for h_dim in hidden_dims:
                encoder_layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                ])
                in_dim = h_dim
            encoder_layers.append(nn.Linear(in_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            # Decoder
            decoder_layers = []
            in_dim = latent_dim
            for h_dim in reversed(hidden_dims):
                decoder_layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                ])
                in_dim = h_dim
            decoder_layers.append(nn.Linear(in_dim, input_dim))
            decoder_layers.append(nn.Sigmoid())  # Output in [0, 1]
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent

        def decode(self, latent):
            return self.decoder(latent)


    class Generator(nn.Module):
        """Generator network for MEDGAN."""

        def __init__(self, noise_dim: int, hidden_dims: tuple, output_dim: int):
            super().__init__()

            layers = []
            in_dim = noise_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                ])
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


    class Discriminator(nn.Module):
        """Discriminator network for MEDGAN."""

        def __init__(self, input_dim: int, hidden_dims: tuple):
            super().__init__()

            layers = []
            in_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.LeakyReLU(0.2)
                ])
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, 1))
            layers.append(nn.Sigmoid())
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)
