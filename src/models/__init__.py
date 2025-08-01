"""
Model implementations and interfaces for synthetic tabular data generation.

This module provides a unified interface for different synthetic data generation
models, including GANerAid, CTGAN, TVAE, CopulaGAN, and TableGAN.
"""

from .base_model import SyntheticDataModel
from .model_factory import ModelFactory

__all__ = [
    "SyntheticDataModel",
    "ModelFactory",
]