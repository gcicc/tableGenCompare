"""
Multi-dataset benchmarking framework for synthetic data generation models.

This module provides comprehensive benchmarking capabilities across multiple
datasets and models, enabling systematic comparison and evaluation of synthetic
data generation performance.
"""

from .dataset_manager import DatasetManager, DatasetConfig, DatasetType, DatasetSize
from .benchmark_pipeline import BenchmarkPipeline, BenchmarkConfig, BenchmarkResult
from .benchmark_reporter import BenchmarkReporter, ModelPerformanceStats, DatasetDifficultyStats

__version__ = "0.2.0"

__all__ = [
    'DatasetManager',
    'DatasetConfig',
    'DatasetType',
    'DatasetSize',
    'BenchmarkPipeline',
    'BenchmarkConfig',
    'BenchmarkResult',
    'BenchmarkReporter',
    'ModelPerformanceStats',
    'DatasetDifficultyStats'
]