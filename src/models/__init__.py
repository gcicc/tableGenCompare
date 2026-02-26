"""
Model implementations and interfaces for synthetic tabular data generation.

This module provides a unified interface for different synthetic data generation
models, including GANerAid, CTGAN, TVAE, CopulaGAN, CTABGAN, CTABGANPLUS,
PATEGAN, and MEDGAN.

Phase 5 - January 2026: Added batch training and optimization functions.
"""

from .base_model import SyntheticDataModel
from .model_factory import ModelFactory

# Batch training (Phase 5)
from .batch_training import (
    train_models_batch,
    extract_synthetic_data_to_globals,
)

# Search spaces for HPO (Phase 5)
from .search_spaces import (
    get_search_space,
    get_pruner_config,
    list_supported_models,
)

# Batch optimization (Phase 5)
from .batch_optimization import (
    optimize_models_batch,
    extract_studies_to_globals,
)

# Staged optimization (Phase 5 - February 2026)
from .staged_optimization import (
    StagedOptimizationConfig,
    ModelOptimizationState,
    TrialTimeTracker,
    ConvergenceAnalyzer,
    StudyPersistence,
    ConciseTrialCallback,
    StagedOptimizationManager,
)

__all__ = [
    # Base classes
    "SyntheticDataModel",
    "ModelFactory",
    # Batch training
    "train_models_batch",
    "extract_synthetic_data_to_globals",
    # Search spaces
    "get_search_space",
    "get_pruner_config",
    "list_supported_models",
    # Batch optimization
    "optimize_models_batch",
    "extract_studies_to_globals",
    # Staged optimization
    "StagedOptimizationConfig",
    "ModelOptimizationState",
    "TrialTimeTracker",
    "ConvergenceAnalyzer",
    "StudyPersistence",
    "ConciseTrialCallback",
    "StagedOptimizationManager",
]