"""
Mode Collapse Detection Functions

This module contains functions for detecting mode collapse in synthetic data,
particularly for categorical features where diversity is lost.
"""

import pandas as pd
import numpy as np
from scipy import stats


def detect_mode_collapse(real_data, synthetic_data, target_column=None, verbose=True):
    """
    Detect mode collapse in categorical variables.

    Mode collapse occurs when a generative model loses diversity and generates
    limited variety in categorical features, indicating the model has "collapsed"
    to a few modes instead of capturing the full distribution.

    Parameters:
    -----------
    real_data : pd.DataFrame
        Real dataset
    synthetic_data : pd.DataFrame
        Synthetic dataset
    target_column : str, optional
        Target column to exclude from analysis
    verbose : bool
        Print warnings for detected collapses

    Returns:
    --------
    dict : {
        'mode_collapse_detected': bool,
        'mode_collapse_count': int,
        'mode_collapse_df': pd.DataFrame,
        'summary': str
    }
    """
    mode_collapse_results = []

    # Get categorical columns
    categorical_cols = [col for col in real_data.columns
                       if real_data[col].dtype in ['object', 'category']]

    # Add integer columns with ≤10 unique values (likely categorical)
    for col in real_data.select_dtypes(include=['int64', 'int32']).columns:
        if col != target_column and real_data[col].nunique() <= 10:
            categorical_cols.append(col)

    # Remove target if present
    if target_column and target_column in categorical_cols:
        categorical_cols.remove(target_column)

    if verbose and categorical_cols:
        print(f"\n[MODE COLLAPSE] Analyzing {len(categorical_cols)} categorical features...")

    for col in categorical_cols:
        if col not in synthetic_data.columns:
            continue

        real_unique = set(real_data[col].dropna().unique())
        synth_unique = set(synthetic_data[col].dropna().unique())

        # Coverage: % of real categories present in synthetic
        coverage = len(synth_unique & real_unique) / len(real_unique) if len(real_unique) > 0 else 0

        # Flag mode collapse based on severity
        mode_collapse_flag = False
        collapse_severity = "None"

        if len(real_unique) > 1 and len(synth_unique) == 1:
            mode_collapse_flag = True
            collapse_severity = "Severe"  # Total collapse to single mode
        elif len(real_unique) > 2 and coverage < 0.5:
            mode_collapse_flag = True
            collapse_severity = "Moderate"  # Lost >50% of modes
        elif coverage < 0.8:
            mode_collapse_flag = True
            collapse_severity = "Mild"  # Lost 20-50% of modes

        # Calculate distribution divergence (Total Variation Distance)
        real_freq = real_data[col].value_counts(normalize=True).to_dict()
        synth_freq = synthetic_data[col].value_counts(normalize=True).to_dict()
        all_categories = set(real_freq.keys()) | set(synth_freq.keys())

        real_vec = np.array([real_freq.get(cat, 0) for cat in all_categories])
        synth_vec = np.array([synth_freq.get(cat, 0) for cat in all_categories])
        tv_distance = 0.5 * np.sum(np.abs(real_vec - synth_vec))
        distribution_similarity = 1 - tv_distance

        # Identify missing and extra categories
        missing_categories = list(real_unique - synth_unique)
        extra_categories = list(synth_unique - real_unique)

        mode_collapse_results.append({
            'column': col,
            'real_unique_count': len(real_unique),
            'synthetic_unique_count': len(synth_unique),
            'category_coverage': coverage,
            'distribution_similarity': distribution_similarity,
            'mode_collapse_flag': mode_collapse_flag,
            'collapse_severity': collapse_severity,
            'missing_categories': str(missing_categories) if missing_categories else '',
            'extra_categories': str(extra_categories) if extra_categories else ''
        })

        if verbose and mode_collapse_flag:
            print(f"   [WARNING] {col}: {collapse_severity} mode collapse detected")
            print(f"      Real: {len(real_unique)} categories, Synthetic: {len(synth_unique)}")
            if missing_categories and len(missing_categories) <= 5:
                print(f"      Missing: {missing_categories}")

    # Create results dictionary
    if mode_collapse_results:
        mode_collapse_df = pd.DataFrame(mode_collapse_results)
        overall_collapse = mode_collapse_df['mode_collapse_flag'].any()
        collapse_count = mode_collapse_df['mode_collapse_flag'].sum()

        summary = f"Mode collapse detected in {collapse_count}/{len(mode_collapse_results)} categorical features"

        if verbose:
            if overall_collapse:
                print(f"\n[RESULT] {summary}")
            else:
                print(f"\n[RESULT] No mode collapse detected in categorical features")

        return {
            'mode_collapse_detected': overall_collapse,
            'mode_collapse_count': collapse_count,
            'mode_collapse_df': mode_collapse_df,
            'summary': summary
        }
    else:
        if verbose:
            print("\n[RESULT] No categorical variables to analyze")

        return {
            'mode_collapse_detected': False,
            'mode_collapse_count': 0,
            'mode_collapse_df': pd.DataFrame(),
            'summary': "No categorical variables to analyze"
        }
