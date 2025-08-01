"""
Optimization framework for automated hyperparameter tuning.

This module provides comprehensive hyperparameter optimization capabilities
for synthetic data generation models using Optuna and other optimization engines.
"""

from .optuna_optimizer import OptunaOptimizer
from .objective_functions import (
    TRTSObjective,
    SimilarityObjective, 
    DataQualityObjective,
    StatisticalObjective,
    MultiObjective,
    CompositeObjective,
    create_trts_objective,
    create_similarity_objective,
    create_quality_objective,
    create_balanced_multi_objective,
    create_comprehensive_multi_objective,
    create_composite_objective
)

__version__ = "0.2.0"

__all__ = [
    'OptunaOptimizer',
    'TRTSObjective',
    'SimilarityObjective',
    'DataQualityObjective', 
    'StatisticalObjective',
    'MultiObjective',
    'CompositeObjective',
    'create_trts_objective',
    'create_similarity_objective',
    'create_quality_objective',
    'create_balanced_multi_objective',
    'create_comprehensive_multi_objective',
    'create_composite_objective'
]