"""
Statistical analysis module for synthetic data quality assessment.

This module provides comprehensive statistical comparison functions between
original and synthetic datasets, including distribution tests and similarity metrics.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Comprehensive statistical analyzer for comparing original and synthetic data.
    
    Provides statistical tests, distribution comparisons, and quality metrics
    to assess how well synthetic data preserves original data characteristics.
    """
    
    def __init__(self):
        """Initialize statistical analyzer."""
        pass
    
    def comprehensive_statistical_comparison(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform comprehensive statistical comparison between original and synthetic data.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            target_column: Target column to include/exclude from analysis
            
        Returns:
            Tuple of (comparison_dataframe, summary_statistics)
        """
        logger.info("Starting comprehensive statistical comparison")
        
        # Get columns to analyze
        if target_column and target_column in original_data.columns:
            columns_to_analyze = original_data.columns.tolist()
        else:
            columns_to_analyze = [col for col in original_data.columns if col != target_column]
        
        statistical_comparison = []
        
        for col in columns_to_analyze:
            if col in synthetic_data.columns:
                orig_data = original_data[col]
                synth_data = synthetic_data[col]
                
                # Calculate comprehensive statistics
                stats_dict = self._calculate_column_statistics(col, orig_data, synth_data)
                statistical_comparison.append(stats_dict)
        
        # Create comparison dataframe
        stats_comparison_df = pd.DataFrame(statistical_comparison)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(stats_comparison_df)
        
        logger.info(f"Statistical comparison completed for {len(statistical_comparison)} features")
        return stats_comparison_df, summary_stats
    
    def _calculate_column_statistics(
        self, 
        column_name: str, 
        orig_data: pd.Series, 
        synth_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for a single column.
        
        Args:
            column_name: Name of the column
            orig_data: Original data series
            synth_data: Synthetic data series
            
        Returns:
            Dictionary containing all statistics for the column
        """
        stats_dict = {
            'Feature': column_name,
            'Original_Mean': orig_data.mean() if orig_data.dtype in ['int64', 'float64', 'int32', 'float32'] else np.nan,
            'Synthetic_Mean': synth_data.mean() if synth_data.dtype in ['int64', 'float64', 'int32', 'float32'] else np.nan,
            'Original_Std': orig_data.std() if orig_data.dtype in ['int64', 'float64', 'int32', 'float32'] else np.nan,
            'Synthetic_Std': synth_data.std() if synth_data.dtype in ['int64', 'float64', 'int32', 'float32'] else np.nan,
            'Original_Min': orig_data.min(),
            'Synthetic_Min': synth_data.min(),
            'Original_Max': orig_data.max(),
            'Synthetic_Max': synth_data.max(),
        }
        
        # Calculate differences
        if not pd.isna(stats_dict['Original_Mean']) and not pd.isna(stats_dict['Synthetic_Mean']):
            stats_dict['Mean_Diff'] = abs(stats_dict['Original_Mean'] - stats_dict['Synthetic_Mean'])
            stats_dict['Std_Diff'] = abs(stats_dict['Original_Std'] - stats_dict['Synthetic_Std'])
        else:
            stats_dict['Mean_Diff'] = np.nan
            stats_dict['Std_Diff'] = np.nan
        
        # Range overlap analysis
        stats_dict['Range_Overlap'] = self._calculate_range_overlap(
            orig_data.min(), orig_data.max(),
            synth_data.min(), synth_data.max()
        )
        
        # Statistical tests
        try:
            # Kolmogorov-Smirnov test
            if orig_data.dtype in ['int64', 'float64', 'int32', 'float32'] and synth_data.dtype in ['int64', 'float64', 'int32', 'float32']:
                ks_stat, ks_pvalue = stats.ks_2samp(orig_data.dropna(), synth_data.dropna())
                stats_dict['KS_Statistic'] = ks_stat
                stats_dict['KS_PValue'] = ks_pvalue
                stats_dict['KS_Similar'] = 'Yes' if ks_pvalue > 0.05 else 'No'
            else:
                # For categorical data, use chi-square test
                stats_dict['KS_Statistic'] = np.nan
                stats_dict['KS_PValue'] = np.nan
                stats_dict['KS_Similar'] = self._categorical_similarity_test(orig_data, synth_data)
        except Exception as e:
            logger.warning(f"Statistical test failed for {column_name}: {e}")
            stats_dict['KS_Statistic'] = np.nan
            stats_dict['KS_PValue'] = np.nan
            stats_dict['KS_Similar'] = 'Unknown'
        
        # Data quality metrics
        stats_dict['Original_Missing'] = orig_data.isnull().sum()
        stats_dict['Synthetic_Missing'] = synth_data.isnull().sum()
        stats_dict['Original_Unique'] = orig_data.nunique()
        stats_dict['Synthetic_Unique'] = synth_data.nunique()
        
        return stats_dict
    
    def _calculate_range_overlap(
        self, 
        orig_min: float, 
        orig_max: float, 
        synth_min: float, 
        synth_max: float
    ) -> str:
        """
        Calculate range overlap between original and synthetic data.
        
        Args:
            orig_min, orig_max: Original data range
            synth_min, synth_max: Synthetic data range
            
        Returns:
            String indicating overlap type
        """
        if synth_min >= orig_min and synth_max <= orig_max:
            return 'Yes'  # Synthetic range fully within original
        elif (synth_min <= orig_max and synth_max >= orig_min):
            return 'Partial'  # Some overlap
        else:
            return 'No'  # No overlap
    
    def _categorical_similarity_test(self, orig_data: pd.Series, synth_data: pd.Series) -> str:
        """
        Test similarity for categorical data using value distributions.
        
        Args:
            orig_data: Original categorical data
            synth_data: Synthetic categorical data
            
        Returns:
            String indicating similarity
        """
        try:
            orig_counts = orig_data.value_counts(normalize=True)
            synth_counts = synth_data.value_counts(normalize=True)
            
            # Check if all original categories are present in synthetic
            missing_categories = set(orig_counts.index) - set(synth_counts.index)
            extra_categories = set(synth_counts.index) - set(orig_counts.index)
            
            if len(missing_categories) == 0 and len(extra_categories) == 0:
                return 'Yes'
            elif len(missing_categories) <= len(orig_counts.index) * 0.1:  # Allow 10% missing
                return 'Partial'
            else:
                return 'No'
        except Exception:
            return 'Unknown'
    
    def _calculate_summary_statistics(self, stats_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics across all features.
        
        Args:
            stats_df: DataFrame containing feature-wise statistics
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {}
        
        # Overall similarity metrics
        if 'Mean_Diff' in stats_df.columns:
            summary['average_mean_error'] = stats_df['Mean_Diff'].mean()
            summary['max_mean_error'] = stats_df['Mean_Diff'].max()
        
        if 'Std_Diff' in stats_df.columns:
            summary['average_std_error'] = stats_df['Std_Diff'].mean()
            summary['max_std_error'] = stats_df['Std_Diff'].max()
        
        # Statistical test results
        if 'KS_Similar' in stats_df.columns:
            similar_count = (stats_df['KS_Similar'] == 'Yes').sum()
            total_count = len(stats_df)
            summary['statistically_similar_features'] = similar_count
            summary['total_features'] = total_count
            summary['similarity_percentage'] = (similar_count / total_count) * 100
        
        # Range overlap results
        if 'Range_Overlap' in stats_df.columns:
            full_overlap = (stats_df['Range_Overlap'] == 'Yes').sum()
            partial_overlap = (stats_df['Range_Overlap'] == 'Partial').sum()
            summary['full_range_overlap_count'] = full_overlap
            summary['partial_range_overlap_count'] = partial_overlap
            summary['no_range_overlap_count'] = total_count - full_overlap - partial_overlap
        
        # Data quality summary
        if 'Original_Missing' in stats_df.columns:
            summary['total_original_missing'] = stats_df['Original_Missing'].sum()
            summary['total_synthetic_missing'] = stats_df['Synthetic_Missing'].sum()
        
        return summary
    
    def correlation_analysis(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        method: str = 'pearson'
    ) -> Dict[str, Any]:
        """
        Perform correlation analysis between original and synthetic data.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary containing correlation analysis results
        """
        logger.info(f"Performing correlation analysis using {method} method")
        
        try:
            # Get numeric columns only
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
            common_numeric = [col for col in numeric_cols if col in synthetic_data.columns]
            
            if len(common_numeric) < 2:
                return {'error': 'Not enough numeric columns for correlation analysis'}
            
            # Calculate correlation matrices
            orig_corr = original_data[common_numeric].corr(method=method)
            synth_corr = synthetic_data[common_numeric].corr(method=method)
            
            # Calculate correlation difference matrix
            corr_diff = np.abs(orig_corr - synth_corr)
            
            # Summary metrics
            avg_corr_diff = corr_diff.values[np.triu_indices_from(corr_diff, k=1)].mean()
            max_corr_diff = corr_diff.values[np.triu_indices_from(corr_diff, k=1)].max()
            
            return {
                'original_correlation_matrix': orig_corr,
                'synthetic_correlation_matrix': synth_corr,
                'correlation_difference_matrix': corr_diff,
                'average_correlation_difference': avg_corr_diff,
                'max_correlation_difference': max_corr_diff,
                'correlation_preservation_score': 1 - avg_corr_diff  # Higher is better
            }
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {'error': f'Correlation analysis failed: {e}'}
    
    def distribution_overlap_analysis(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        bins: int = 30
    ) -> Dict[str, float]:
        """
        Calculate distribution overlap scores for numeric features.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            bins: Number of bins for histogram comparison
            
        Returns:
            Dictionary mapping feature names to overlap scores
        """
        logger.info("Calculating distribution overlap scores")
        
        overlap_scores = {}
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in synthetic_data.columns:
                try:
                    orig_values = original_data[col].dropna()
                    synth_values = synthetic_data[col].dropna()
                    
                    if len(orig_values) == 0 or len(synth_values) == 0:
                        overlap_scores[col] = 0.0
                        continue
                    
                    # Create common bins
                    min_val = min(orig_values.min(), synth_values.min())
                    max_val = max(orig_values.max(), synth_values.max())
                    bin_edges = np.linspace(min_val, max_val, bins + 1)
                    
                    # Calculate histograms
                    orig_hist, _ = np.histogram(orig_values, bins=bin_edges, density=True)
                    synth_hist, _ = np.histogram(synth_values, bins=bin_edges, density=True)
                    
                    # Calculate overlap using histogram intersection
                    overlap = np.sum(np.minimum(orig_hist, synth_hist)) / np.sum(np.maximum(orig_hist, synth_hist))
                    overlap_scores[col] = overlap
                    
                except Exception as e:
                    logger.warning(f"Overlap calculation failed for {col}: {e}")
                    overlap_scores[col] = 0.0
        
        return overlap_scores
    
    def data_quality_assessment(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Assess overall data quality of synthetic data.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            
        Returns:
            Dictionary containing data quality metrics
        """
        logger.info("Performing data quality assessment")
        
        quality_metrics = {}
        
        # Basic shape comparison
        quality_metrics['shape_match'] = original_data.shape == synthetic_data.shape
        quality_metrics['column_match'] = list(original_data.columns) == list(synthetic_data.columns)
        
        # Missing values comparison
        orig_missing = original_data.isnull().sum().sum()
        synth_missing = synthetic_data.isnull().sum().sum()
        quality_metrics['missing_values_original'] = orig_missing
        quality_metrics['missing_values_synthetic'] = synth_missing
        
        # Data type consistency
        type_matches = 0
        total_columns = len(original_data.columns)
        
        for col in original_data.columns:
            if col in synthetic_data.columns:
                if original_data[col].dtype == synthetic_data[col].dtype:
                    type_matches += 1
        
        quality_metrics['data_type_consistency'] = (type_matches / total_columns) * 100
        
        # Range validity (synthetic values within original ranges)
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        range_valid_count = 0
        
        for col in numeric_cols:
            if col in synthetic_data.columns:
                orig_min, orig_max = original_data[col].min(), original_data[col].max()
                synth_min, synth_max = synthetic_data[col].min(), synthetic_data[col].max()
                
                # Check if synthetic range is within reasonable bounds of original
                range_expansion_factor = 1.2  # Allow 20% expansion
                if (synth_min >= orig_min * range_expansion_factor and 
                    synth_max <= orig_max * range_expansion_factor):
                    range_valid_count += 1
        
        if len(numeric_cols) > 0:
            quality_metrics['range_validity_percentage'] = (range_valid_count / len(numeric_cols)) * 100
        else:
            quality_metrics['range_validity_percentage'] = 100.0
        
        # Duplicate analysis
        quality_metrics['duplicates_original'] = original_data.duplicated().sum()
        quality_metrics['duplicates_synthetic'] = synthetic_data.duplicated().sum()
        
        return quality_metrics