"""
Dataset handling and preprocessing for synthetic tabular data benchmarking.

This module provides unified interfaces for loading, preprocessing, and
validating datasets across different domains and formats.
"""

from .dataset_handler import DatasetHandler
from .data_validator import DataValidator

__all__ = [
    "DatasetHandler",
    "DataValidator",
]