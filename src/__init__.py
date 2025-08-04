"""
Synthetic Tabular Data Benchmark Framework

A comprehensive framework for comparing synthetic tabular data generation models
across multiple datasets with research-grade evaluation metrics.
"""

__version__ = "0.1.0"
__author__ = "Clinical Research Team"

# Core imports for easy access
try:
    from .models import ModelFactory
except ImportError:
    ModelFactory = None

try:
    from .datasets import DatasetHandler
except ImportError:
    DatasetHandler = None

try:
    from .evaluation import UnifiedEvaluator
except ImportError:
    UnifiedEvaluator = None

# ExperimentRunner not yet implemented
ExperimentRunner = None

__all__ = [
    "ModelFactory",
    "DatasetHandler", 
    "UnifiedEvaluator",
    "ExperimentRunner",
]