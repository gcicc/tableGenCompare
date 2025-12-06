"""
Data Quality Evaluation Functions

This module contains functions for evaluating synthetic data quality,
including statistical similarity metrics and higher-order feature interactions.
"""

import pandas as pd
import numpy as np


def calculate_mutual_information(real_data, synthetic_data, target_column,
                                 max_features=10, verbose=True):
    """
    Calculate mutual information preservation between real and synthetic data.

    Mutual information captures non-linear dependencies between features and the target,
    going beyond simple correlation to measure shared information content.

    Parameters:
    -----------
    real_data : pd.DataFrame
        Real dataset
    synthetic_data : pd.DataFrame
        Synthetic dataset
    target_column : str
        Target column name
    max_features : int
        Maximum number of features to analyze (MI is computationally expensive)
    verbose : bool
        Print progress messages

    Returns:
    --------
    dict : {
        'mi_preservation': float,
        'mi_real': np.array,
        'mi_synth': np.array,
        'mi_features': list,
        'mi_correlation': float
    }
    """
    if verbose:
        print(f"\n[MI] Calculating Mutual Information preservation...")

    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        # Get numeric columns excluding target
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        # Limit to max_features for performance
        mi_cols = numeric_cols[:min(max_features, len(numeric_cols))]

        if not mi_cols:
            return {
                'mi_preservation': np.nan,
                'mi_real': np.array([]),
                'mi_synth': np.array([]),
                'mi_features': [],
                'mi_correlation': np.nan,
                'error': 'No numeric features to analyze'
            }

        if target_column not in real_data.columns:
            return {
                'mi_preservation': np.nan,
                'mi_real': np.array([]),
                'mi_synth': np.array([]),
                'mi_features': [],
                'mi_correlation': np.nan,
                'error': 'Target column not found'
            }

        # Prepare data
        X_real = real_data[mi_cols].fillna(0)
        X_synth = synthetic_data[mi_cols].fillna(0)
        y_real = real_data[target_column]
        y_synth = synthetic_data[target_column] if target_column in synthetic_data.columns else None

        if y_synth is None:
            return {
                'mi_preservation': np.nan,
                'mi_real': np.array([]),
                'mi_synth': np.array([]),
                'mi_features': [],
                'mi_correlation': np.nan,
                'error': 'Target column not in synthetic data'
            }

        # Determine if classification or regression
        is_classification = y_real.nunique() <= 10

        if verbose:
            print(f"   [MI] Analyzing {len(mi_cols)} features...")
            print(f"   [MI] Task type: {'Classification' if is_classification else 'Regression'}")

        # Calculate MI
        if is_classification:
            mi_real = mutual_info_classif(X_real, y_real, random_state=42)
            mi_synth = mutual_info_classif(X_synth, y_synth, random_state=42)
        else:
            mi_real = mutual_info_regression(X_real, y_real, random_state=42)
            mi_synth = mutual_info_regression(X_synth, y_synth, random_state=42)

        # Calculate preservation score (correlation between MI vectors)
        from scipy.stats import pearsonr
        if len(mi_real) > 1:
            mi_correlation = pearsonr(mi_real, mi_synth)[0]
            mi_correlation = max(0, mi_correlation)  # Clip to [0, 1]
        else:
            mi_correlation = 0.0

        if verbose:
            print(f"   [METRIC] MI Preservation: {mi_correlation:.3f}")

        return {
            'mi_preservation': mi_correlation,
            'mi_real': mi_real,
            'mi_synth': mi_synth,
            'mi_features': mi_cols,
            'mi_correlation': mi_correlation
        }

    except ImportError as e:
        if verbose:
            print(f"   [ERROR] sklearn.feature_selection not available: {e}")
        return {
            'mi_preservation': np.nan,
            'mi_real': np.array([]),
            'mi_synth': np.array([]),
            'mi_features': [],
            'mi_correlation': np.nan,
            'error': 'sklearn not available'
        }
    except Exception as e:
        if verbose:
            print(f"   [ERROR] MI calculation failed: {e}")
        return {
            'mi_preservation': np.nan,
            'mi_real': np.array([]),
            'mi_synth': np.array([]),
            'mi_features': [],
            'mi_correlation': np.nan,
            'error': str(e)
        }
