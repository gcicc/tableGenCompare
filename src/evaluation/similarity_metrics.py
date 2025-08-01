"""
Advanced similarity metrics for synthetic data quality assessment.

This module provides sophisticated similarity calculations including
Wasserstein distance, Jensen-Shannon divergence, and bivariate analysis.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

# Check for optimal transport library
try:
    import ot  # POT (Python Optimal Transport) library
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False

# Check for wasserstein distance
try:
    from scipy.stats import wasserstein_distance
    WASSERSTEIN_AVAILABLE = True
except ImportError:
    WASSERSTEIN_AVAILABLE = False


class SimilarityCalculator:
    """
    Advanced similarity calculator for synthetic data evaluation.
    
    Provides sophisticated similarity metrics including Wasserstein distance,
    Jensen-Shannon divergence, and bivariate feature relationship analysis.
    """
    
    def __init__(self):
        """Initialize similarity calculator."""
        self.wasserstein_available = WASSERSTEIN_AVAILABLE
        self.ot_available = OT_AVAILABLE
        
        if not self.wasserstein_available:
            logger.warning("Wasserstein distance not available - using fallback methods")
        if not self.ot_available:
            logger.warning("Optimal transport library not available - bivariate analysis will use correlation-based fallback")
    
    def evaluate_univariate_similarity(
        self, 
        original: pd.DataFrame, 
        synthetic: pd.DataFrame, 
        target_col: Optional[str] = None
    ) -> float:
        """
        Evaluate univariate similarity using Wasserstein distance for numerical features
        and Jensen-Shannon divergence for categorical features.
        
        Args:
            original: Original dataset  
            synthetic: Synthetic dataset
            target_col: Target column to exclude from evaluation
            
        Returns:
            Average univariate similarity score (higher = better)
        """
        logger.info("Calculating univariate similarity")
        
        try:
            # Remove target column
            orig_features = original.drop(columns=[target_col]) if target_col and target_col in original.columns else original
            synth_features = synthetic.drop(columns=[target_col]) if target_col and target_col in synthetic.columns else synthetic
            
            similarities = []
            
            for col in orig_features.columns:
                if col in synth_features.columns:
                    orig_data = orig_features[col].dropna()
                    synth_data = synth_features[col].dropna()
                    
                    if len(orig_data) == 0 or len(synth_data) == 0:
                        continue
                    
                    if orig_data.dtype in ['int64', 'float64', 'int32', 'float32']:
                        # Numerical feature: Wasserstein distance
                        similarity = self._calculate_wasserstein_similarity(orig_data, synth_data)
                    else:
                        # Categorical feature: Jensen-Shannon divergence
                        similarity = self._calculate_js_similarity(orig_data, synth_data)
                    
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.error(f"Univariate similarity calculation failed: {e}")
            return 0.5
    
    def evaluate_bivariate_similarity(
        self, 
        original: pd.DataFrame, 
        synthetic: pd.DataFrame, 
        target_col: Optional[str] = None
    ) -> float:
        """
        Evaluate bivariate similarity using Wasserstein distance on feature pairs.
        Uses optimal transport if available, otherwise falls back to correlation analysis.
        
        Args:
            original: Original dataset
            synthetic: Synthetic dataset
            target_col: Target column to exclude from evaluation
            
        Returns:
            Average bivariate similarity score (higher = better)
        """
        logger.info("Calculating bivariate similarity")
        
        try:
            # Remove target column
            orig_features = original.drop(columns=[target_col]) if target_col and target_col in original.columns else original
            synth_features = synthetic.drop(columns=[target_col]) if target_col and target_col in synthetic.columns else synthetic
            
            # Get numerical columns only
            numeric_cols = orig_features.select_dtypes(include=[np.number]).columns
            common_numeric = [col for col in numeric_cols if col in synth_features.columns]
            
            if len(common_numeric) < 2:
                return 0.5  # Not enough features for bivariate analysis
            
            similarities = []
            
            if self.ot_available:
                # Advanced bivariate analysis using optimal transport
                similarities = self._calculate_ot_bivariate_similarity(orig_features, synth_features, common_numeric)
            else:
                # Fallback: correlation-based bivariate analysis
                similarities = self._calculate_correlation_bivariate_similarity(orig_features, synth_features, common_numeric)
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.error(f"Bivariate similarity calculation failed: {e}")
            return 0.5
    
    def evaluate_overall_similarity(
        self, 
        original: pd.DataFrame, 
        synthetic: pd.DataFrame, 
        target_col: Optional[str] = None
    ) -> Tuple[float, float, float]:
        """
        Evaluate overall similarity combining univariate and bivariate metrics.
        Returns comprehensive similarity assessment.
        
        Args:
            original: Original dataset
            synthetic: Synthetic dataset
            target_col: Target column to exclude from evaluation
            
        Returns:
            Tuple of (final_similarity, univariate_similarity, bivariate_similarity)
        """
        logger.info(f"Evaluating overall similarity for {len(original)} vs {len(synthetic)} samples")
        
        try:
            # Calculate univariate similarity
            univariate_sim = self.evaluate_univariate_similarity(original, synthetic, target_col)
            logger.info(f"Univariate similarity: {univariate_sim:.4f}")
            
            # Calculate bivariate similarity
            bivariate_sim = self.evaluate_bivariate_similarity(original, synthetic, target_col)
            logger.info(f"Bivariate similarity: {bivariate_sim:.4f}")
            
            # Combined similarity (weighted average)
            # Weight univariate more heavily as it's more reliable
            final_similarity = (0.7 * univariate_sim) + (0.3 * bivariate_sim)
            logger.info(f"Final similarity: {final_similarity:.4f}")
            
            return final_similarity, univariate_sim, bivariate_sim
            
        except Exception as e:
            logger.error(f"Overall similarity evaluation failed: {e}")
            # Return safe default values
            return 0.5, 0.5, 0.5
    
    def _calculate_wasserstein_similarity(self, orig_data: pd.Series, synth_data: pd.Series) -> float:
        """
        Calculate similarity using Wasserstein distance.
        
        Args:
            orig_data: Original data series
            synth_data: Synthetic data series
            
        Returns:
            Similarity score (0-1, higher better)
        """
        try:
            if self.wasserstein_available:
                wd = wasserstein_distance(orig_data, synth_data)
                # Convert to similarity (0-1, higher better)
                # Normalize by data range
                data_range = max(orig_data.max() - orig_data.min(), 1e-10)
                similarity = 1 / (1 + wd / data_range)
                return similarity
            else:
                # Fallback: inverse normalized mean difference
                mean_diff = abs(orig_data.mean() - synth_data.mean())
                data_std = max(orig_data.std(), 1e-10)
                similarity = 1 / (1 + mean_diff / data_std)
                return similarity
        except Exception:
            # Final fallback: simple mean comparison
            mean_diff = abs(orig_data.mean() - synth_data.mean())
            data_range = max(orig_data.max() - orig_data.min(), 1e-10)
            return 1 / (1 + mean_diff / data_range)
    
    def _calculate_js_similarity(self, orig_data: pd.Series, synth_data: pd.Series) -> float:
        """
        Calculate similarity using Jensen-Shannon divergence for categorical data.
        
        Args:
            orig_data: Original data series
            synth_data: Synthetic data series
            
        Returns:
            Similarity score (0-1, higher better)
        """
        try:
            # Get value counts and align
            orig_counts = orig_data.value_counts(normalize=True)
            synth_counts = synth_data.value_counts(normalize=True)
            
            # Align indices
            all_values = set(orig_counts.index) | set(synth_counts.index)
            p = np.array([orig_counts.get(v, 0) for v in all_values])
            q = np.array([synth_counts.get(v, 0) for v in all_values])
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            p = p + epsilon
            q = q + epsilon
            p = p / p.sum()
            q = q / q.sum()
            
            # Jensen-Shannon divergence
            m = (p + q) / 2
            js_div = (stats.entropy(p, m) + stats.entropy(q, m)) / 2
            
            # Convert to similarity (0-1, higher better)
            similarity = 1 / (1 + js_div)
            return similarity
        except Exception:
            # Fallback: overlap ratio
            orig_unique = set(orig_data.unique())
            synth_unique = set(synth_data.unique())
            if len(orig_unique) > 0:
                overlap = len(orig_unique.intersection(synth_unique))
                similarity = overlap / len(orig_unique)
            else:
                similarity = 1.0
            return similarity
    
    def _calculate_ot_bivariate_similarity(
        self, 
        orig_features: pd.DataFrame, 
        synth_features: pd.DataFrame, 
        common_numeric: list
    ) -> list:
        """
        Calculate bivariate similarity using optimal transport.
        
        Args:
            orig_features: Original features dataframe
            synth_features: Synthetic features dataframe
            common_numeric: List of common numeric columns
            
        Returns:
            List of similarity scores
        """
        similarities = []
        
        for i in range(len(common_numeric)):
            for j in range(i + 1, len(common_numeric)):
                col1, col2 = common_numeric[i], common_numeric[j]
                
                try:
                    # Extract feature pairs
                    orig_pair = orig_features[[col1, col2]].dropna()
                    synth_pair = synth_features[[col1, col2]].dropna()
                    
                    if len(orig_pair) < 10 or len(synth_pair) < 10:
                        continue
                    
                    # Normalize features
                    orig_normalized = (orig_pair - orig_pair.mean()) / (orig_pair.std() + 1e-10)
                    synth_normalized = (synth_pair - synth_pair.mean()) / (synth_pair.std() + 1e-10)
                    
                    # Calculate 2D Wasserstein distance using optimal transport
                    cost_matrix = ot.dist(orig_normalized.values, synth_normalized.values)
                    
                    # Uniform distributions
                    orig_weights = np.ones(len(orig_normalized)) / len(orig_normalized)
                    synth_weights = np.ones(len(synth_normalized)) / len(synth_normalized)
                    
                    # Compute optimal transport distance
                    ot_distance = ot.emd2(orig_weights, synth_weights, cost_matrix)
                    
                    # Convert to similarity
                    similarity = 1 / (1 + ot_distance)
                    similarities.append(similarity)
                    
                except Exception:
                    continue
        
        return similarities
    
    def _calculate_correlation_bivariate_similarity(
        self, 
        orig_features: pd.DataFrame, 
        synth_features: pd.DataFrame, 
        common_numeric: list
    ) -> list:
        """
        Calculate bivariate similarity using correlation analysis (fallback method).
        
        Args:
            orig_features: Original features dataframe
            synth_features: Synthetic features dataframe
            common_numeric: List of common numeric columns
            
        Returns:
            List of similarity scores
        """
        similarities = []
        
        for i in range(len(common_numeric)):
            for j in range(i + 1, len(common_numeric)):
                col1, col2 = common_numeric[i], common_numeric[j]
                
                try:
                    # Calculate correlations
                    orig_corr = orig_features[col1].corr(orig_features[col2])
                    synth_corr = synth_features[col1].corr(synth_features[col2])
                    
                    # Handle NaN correlations
                    if pd.isna(orig_corr) or pd.isna(synth_corr):
                        continue
                    
                    # Correlation similarity
                    corr_diff = abs(orig_corr - synth_corr)
                    similarity = 1 / (1 + corr_diff)
                    similarities.append(similarity)
                    
                except Exception:
                    continue
        
        return similarities
    
    def calculate_feature_wise_similarities(
        self, 
        original: pd.DataFrame, 
        synthetic: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate similarity scores for each feature individually.
        
        Args:
            original: Original dataset
            synthetic: Synthetic dataset
            target_col: Target column to exclude
            
        Returns:
            Dictionary mapping feature names to similarity scores
        """
        logger.info("Calculating feature-wise similarities")
        
        feature_similarities = {}
        
        # Remove target column
        orig_features = original.drop(columns=[target_col]) if target_col and target_col in original.columns else original
        synth_features = synthetic.drop(columns=[target_col]) if target_col and target_col in synthetic.columns else synthetic
        
        for col in orig_features.columns:
            if col in synth_features.columns:
                orig_data = orig_features[col].dropna()
                synth_data = synth_features[col].dropna()
                
                if len(orig_data) == 0 or len(synth_data) == 0:
                    feature_similarities[col] = 0.0
                    continue
                
                if orig_data.dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Numerical feature
                    similarity = self._calculate_wasserstein_similarity(orig_data, synth_data)
                else:
                    # Categorical feature
                    similarity = self._calculate_js_similarity(orig_data, synth_data)
                
                feature_similarities[col] = similarity
        
        return feature_similarities