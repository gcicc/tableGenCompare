"""
Synthetic Tabular Data Benchmark Framework

A comprehensive framework for comparing synthetic tabular data generation models
across multiple datasets with research-grade evaluation metrics.
"""

__version__ = "0.1.0"
__author__ = "Clinical Research Team"

# Core imports for easy access
from .models import ModelFactory
from .datasets import DatasetHandler
from .evaluation import UnifiedEvaluator
from .experiments import ExperimentRunner

__all__ = [
    "ModelFactory",
    "DatasetHandler", 
    "UnifiedEvaluator",
    "ExperimentRunner",
]