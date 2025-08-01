"""
Objective functions for hyperparameter optimization.

This module defines various objective functions that can be used for
optimizing synthetic data model hyperparameters, including TRTS-based
objectives, similarity objectives, and composite multi-objective functions.
"""

from typing import Dict, Any, List, Optional, Callable, Union
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ObjectiveFunction(ABC):
    """
    Abstract base class for optimization objective functions.
    
    Defines the interface that all objective functions must implement
    for use with the OptunaOptimizer.
    """
    
    def __init__(self, name: str, direction: str = "maximize"):
        """
        Initialize objective function.
        
        Args:
            name: Name of the objective function
            direction: Optimization direction ("maximize" or "minimize")
        """
        self.name = name
        self.direction = direction
        self.weight = 1.0
    
    @abstractmethod
    def evaluate(self, evaluation_results: Dict[str, Any]) -> float:
        """
        Evaluate the objective function given evaluation results.
        
        Args:
            evaluation_results: Results from UnifiedEvaluator
            
        Returns:
            Objective value (higher is better for maximize, lower for minimize)
        """
        pass
    
    def set_weight(self, weight: float) -> None:
        """Set the weight for this objective in multi-objective optimization."""
        self.weight = weight


class TRTSObjective(ObjectiveFunction):
    """
    TRTS-based objective function.
    
    Optimizes based on TRTS framework scores (utility, quality, or overall).
    """
    
    def __init__(self, metric: str = "overall", direction: str = "maximize"):
        """
        Initialize TRTS objective.
        
        Args:
            metric: TRTS metric to optimize ("overall", "utility", "quality")
            direction: Optimization direction
        """
        super().__init__(f"trts_{metric}", direction)
        self.metric = metric.lower()
        
        if self.metric not in ["overall", "utility", "quality"]:
            raise ValueError(f"Invalid TRTS metric: {metric}. Choose from: overall, utility, quality")
    
    def evaluate(self, evaluation_results: Dict[str, Any]) -> float:
        """
        Evaluate TRTS objective.
        
        Args:
            evaluation_results: Results from UnifiedEvaluator
            
        Returns:
            TRTS score normalized to [0, 1]
        """
        try:
            trts_results = evaluation_results.get('trts_results', {})
            
            if self.metric == "overall":
                score = trts_results.get('overall_score_percent', 0.0)
            elif self.metric == "utility":
                score = trts_results.get('utility_score_percent', 0.0)
            elif self.metric == "quality":
                score = trts_results.get('quality_score_percent', 0.0)
            else:
                score = 0.0
            
            # Normalize to [0, 1] (TRTS scores are percentages)
            normalized_score = score / 100.0
            
            return normalized_score * self.weight
            
        except Exception as e:
            logger.warning(f"Failed to evaluate TRTS objective: {e}")
            return 0.0


class SimilarityObjective(ObjectiveFunction):
    """
    Similarity-based objective function.
    
    Optimizes based on similarity metrics between original and synthetic data.
    """
    
    def __init__(self, metric: str = "final", direction: str = "maximize"):
        """
        Initialize similarity objective.
        
        Args:
            metric: Similarity metric to optimize ("final", "univariate", "bivariate")
            direction: Optimization direction
        """
        super().__init__(f"similarity_{metric}", direction)
        self.metric = metric.lower()
        
        if self.metric not in ["final", "univariate", "bivariate"]:
            raise ValueError(f"Invalid similarity metric: {metric}. Choose from: final, univariate, bivariate")
    
    def evaluate(self, evaluation_results: Dict[str, Any]) -> float:
        """
        Evaluate similarity objective.
        
        Args:
            evaluation_results: Results from UnifiedEvaluator
            
        Returns:
            Similarity score (already normalized to [0, 1])
        """
        try:
            similarity_results = evaluation_results.get('similarity_analysis', {})
            
            if self.metric == "final":
                score = similarity_results.get('final_similarity', 0.0)
            elif self.metric == "univariate":
                score = similarity_results.get('univariate_similarity', 0.0)
            elif self.metric == "bivariate":
                score = similarity_results.get('bivariate_similarity', 0.0)
            else:
                score = 0.0
            
            return score * self.weight
            
        except Exception as e:
            logger.warning(f"Failed to evaluate similarity objective: {e}")
            return 0.0


class DataQualityObjective(ObjectiveFunction):
    """
    Data quality-based objective function.
    
    Optimizes based on synthetic data quality metrics.
    """
    
    def __init__(self, metric: str = "overall", direction: str = "maximize"):
        """
        Initialize data quality objective.
        
        Args:
            metric: Quality metric to optimize ("overall", "type_consistency", "range_validity")
            direction: Optimization direction
        """
        super().__init__(f"data_quality_{metric}", direction)
        self.metric = metric.lower()
        
        if self.metric not in ["overall", "type_consistency", "range_validity"]:
            raise ValueError(f"Invalid data quality metric: {metric}")
    
    def evaluate(self, evaluation_results: Dict[str, Any]) -> float:
        """
        Evaluate data quality objective.
        
        Args:
            evaluation_results: Results from UnifiedEvaluator
            
        Returns:
            Data quality score normalized to [0, 1]
        """
        try:
            quality_results = evaluation_results.get('data_quality', {})
            
            if self.metric == "overall":
                # Composite score from multiple quality metrics
                type_consistency = quality_results.get('data_type_consistency', 0.0) / 100.0
                range_validity = quality_results.get('range_validity_percentage', 0.0) / 100.0
                column_match = 1.0 if quality_results.get('column_match', False) else 0.0
                shape_match = 1.0 if quality_results.get('shape_match', False) else 0.0
                
                score = (type_consistency + range_validity + column_match + shape_match) / 4.0
            elif self.metric == "type_consistency":
                score = quality_results.get('data_type_consistency', 0.0) / 100.0
            elif self.metric == "range_validity":
                score = quality_results.get('range_validity_percentage', 0.0) / 100.0
            else:
                score = 0.0
            
            return score * self.weight
            
        except Exception as e:
            logger.warning(f"Failed to evaluate data quality objective: {e}")
            return 0.0


class StatisticalObjective(ObjectiveFunction):
    """
    Statistical similarity-based objective function.
    
    Optimizes based on statistical comparison metrics.
    """
    
    def __init__(self, metric: str = "similarity_percentage", direction: str = "maximize"):
        """
        Initialize statistical objective.
        
        Args:
            metric: Statistical metric to optimize
            direction: Optimization direction
        """
        super().__init__(f"statistical_{metric}", direction)
        self.metric = metric.lower()
    
    def evaluate(self, evaluation_results: Dict[str, Any]) -> float:
        """
        Evaluate statistical objective.
        
        Args:
            evaluation_results: Results from UnifiedEvaluator
            
        Returns:
            Statistical metric score
        """
        try:
            stats_results = evaluation_results.get('statistical_analysis', {})
            summary_stats = stats_results.get('summary_statistics', {})
            
            if self.metric == "similarity_percentage":
                score = summary_stats.get('similarity_percentage', 0.0) / 100.0
            elif self.metric == "mean_error":
                # Lower mean error is better, so invert for maximization
                mean_error = summary_stats.get('average_mean_error', float('inf'))
                score = 1.0 / (1.0 + mean_error) if mean_error != float('inf') else 0.0
            else:
                score = 0.0
            
            return score * self.weight
            
        except Exception as e:
            logger.warning(f"Failed to evaluate statistical objective: {e}")
            return 0.0


class MultiObjective:
    """
    Multi-objective function composer.
    
    Combines multiple objective functions for multi-objective optimization.
    """
    
    def __init__(self, objectives: List[ObjectiveFunction]):
        """
        Initialize multi-objective function.
        
        Args:
            objectives: List of objective functions to combine
        """
        self.objectives = objectives
        self.weights = [obj.weight for obj in objectives]
    
    def evaluate(self, evaluation_results: Dict[str, Any]) -> List[float]:
        """
        Evaluate all objectives.
        
        Args:
            evaluation_results: Results from UnifiedEvaluator
            
        Returns:
            List of objective values
        """
        values = []
        for objective in self.objectives:
            try:
                value = objective.evaluate(evaluation_results)
                values.append(value)
            except Exception as e:
                logger.warning(f"Failed to evaluate objective {objective.name}: {e}")
                values.append(0.0)
        
        return values
    
    def get_objective_names(self) -> List[str]:
        """Get names of all objectives."""
        return [obj.name for obj in self.objectives]
    
    def get_directions(self) -> List[str]:
        """Get optimization directions for all objectives."""
        return [obj.direction for obj in self.objectives]
    
    def set_weights(self, weights: List[float]) -> None:
        """
        Set weights for all objectives.
        
        Args:
            weights: List of weights (must match number of objectives)
        """
        if len(weights) != len(self.objectives):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of objectives ({len(self.objectives)})")
        
        for obj, weight in zip(self.objectives, weights):
            obj.set_weight(weight)
        
        self.weights = weights


class CompositeObjective(ObjectiveFunction):
    """
    Composite objective function.
    
    Combines multiple objectives into a single weighted score for
    single-objective optimization.
    """
    
    def __init__(
        self, 
        objectives: List[ObjectiveFunction], 
        weights: Optional[List[float]] = None,
        aggregation: str = "weighted_sum"
    ):
        """
        Initialize composite objective.
        
        Args:
            objectives: List of objective functions to combine
            weights: Weights for each objective (default: equal weights)
            aggregation: Aggregation method ("weighted_sum", "geometric_mean")
        """
        super().__init__("composite", "maximize")
        self.objectives = objectives
        self.aggregation = aggregation.lower()
        
        if weights is None:
            self.weights = [1.0 / len(objectives)] * len(objectives)
        else:
            if len(weights) != len(objectives):
                raise ValueError("Number of weights must match number of objectives")
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        if self.aggregation not in ["weighted_sum", "geometric_mean"]:
            raise ValueError("Aggregation must be 'weighted_sum' or 'geometric_mean'")
    
    def evaluate(self, evaluation_results: Dict[str, Any]) -> float:
        """
        Evaluate composite objective.
        
        Args:
            evaluation_results: Results from UnifiedEvaluator
            
        Returns:
            Aggregated objective value
        """
        values = []
        for objective in self.objectives:
            try:
                value = objective.evaluate(evaluation_results)
                values.append(max(0.0, value))  # Ensure non-negative for geometric mean
            except Exception as e:
                logger.warning(f"Failed to evaluate objective {objective.name}: {e}")
                values.append(0.0)
        
        if self.aggregation == "weighted_sum":
            return sum(w * v for w, v in zip(self.weights, values))
        elif self.aggregation == "geometric_mean":
            # Geometric mean of weighted values
            if any(v == 0 for v in values):
                return 0.0
            product = 1.0
            for w, v in zip(self.weights, values):
                if v > 0:
                    product *= v ** w
            return product
        else:
            return 0.0


# Predefined objective function factories
def create_trts_objective(metric: str = "overall") -> TRTSObjective:
    """Create a TRTS-based objective function."""
    return TRTSObjective(metric=metric)


def create_similarity_objective(metric: str = "final") -> SimilarityObjective:
    """Create a similarity-based objective function.""" 
    return SimilarityObjective(metric=metric)


def create_quality_objective(metric: str = "overall") -> DataQualityObjective:
    """Create a data quality-based objective function."""
    return DataQualityObjective(metric=metric)


def create_balanced_multi_objective() -> MultiObjective:
    """Create a balanced multi-objective function with TRTS and similarity."""
    objectives = [
        TRTSObjective(metric="overall"),
        SimilarityObjective(metric="final")
    ]
    return MultiObjective(objectives)


def create_comprehensive_multi_objective() -> MultiObjective:
    """Create a comprehensive multi-objective function with all key metrics."""
    objectives = [
        TRTSObjective(metric="overall"),
        SimilarityObjective(metric="final"),
        DataQualityObjective(metric="overall")
    ]
    return MultiObjective(objectives)


def create_composite_objective(
    include_trts: bool = True,
    include_similarity: bool = True,
    include_quality: bool = True,
    trts_weight: float = 0.5,
    similarity_weight: float = 0.3,
    quality_weight: float = 0.2
) -> CompositeObjective:
    """
    Create a composite objective function with customizable components.
    
    Args:
        include_trts: Include TRTS objective
        include_similarity: Include similarity objective
        include_quality: Include data quality objective
        trts_weight: Weight for TRTS objective
        similarity_weight: Weight for similarity objective
        quality_weight: Weight for quality objective
        
    Returns:
        Composite objective function
    """
    objectives = []
    weights = []
    
    if include_trts:
        objectives.append(TRTSObjective(metric="overall"))
        weights.append(trts_weight)
    
    if include_similarity:
        objectives.append(SimilarityObjective(metric="final"))
        weights.append(similarity_weight)
    
    if include_quality:
        objectives.append(DataQualityObjective(metric="overall"))
        weights.append(quality_weight)
    
    if not objectives:
        raise ValueError("At least one objective must be included")
    
    return CompositeObjective(objectives, weights)