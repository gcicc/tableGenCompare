"""
Comprehensive EDA (Exploratory Data Analysis) Module

This module consolidates EDA functionality from CHUNK_2_1_4_A through CHUNK_2_1_4_F
into a single function call for streamlined notebook usage.

Phase 5 - January 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.utils.paths import get_results_path
from src.visualization.section2 import (
    create_mixed_association_heatmap,
    create_feature_distributions
)


# ---- association primitives ----
# Copied verbatim from multi-table-gen-compare/src/data/eda.py (keep in sync).

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V for two categorical series."""
    ct = pd.crosstab(x, y)
    if ct.size == 0:
        return np.nan
    chi2 = stats.chi2_contingency(ct, correction=False).statistic
    n = ct.values.sum()
    r, c = ct.shape
    denom = n * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan


def _eta(num: pd.Series, cat: pd.Series) -> float:
    """Correlation ratio η between numeric and categorical series."""
    df = pd.DataFrame({"n": num, "c": cat}).dropna()
    if df.empty or df["c"].nunique() < 2:
        return np.nan
    overall_mean = df["n"].mean()
    grp = df.groupby("c")["n"]
    ss_between = float(((grp.mean() - overall_mean) ** 2 * grp.count()).sum())
    ss_total = float(((df["n"] - overall_mean) ** 2).sum())
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else np.nan


def mixed_association_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise mixed-association matrix.

    Pearson for numeric×numeric (signed, [-1, 1]).
    Cramér's V for categorical×categorical (unsigned, [0, 1]).
    Correlation ratio η for numeric×categorical (unsigned, [0, 1]).

    Note: this is the sibling-aligned implementation, distinct from
    ``src.evaluation.association.compute_mixed_association_matrix`` (which
    wraps dython). Both coexist; use this one when you need parity with
    multi-table-gen-compare's fidelity metrics.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(
        include=["object", "category", "string"],
    ).columns.tolist()
    cols = num_cols + cat_cols
    out = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    for a in cols:
        for b in cols:
            if a == b:
                out.loc[a, b] = 1.0
                continue
            if a in num_cols and b in num_cols:
                out.loc[a, b] = float(df[a].corr(df[b]))
            elif a in cat_cols and b in cat_cols:
                out.loc[a, b] = _cramers_v(df[a], df[b])
            else:
                num_s = df[a] if a in num_cols else df[b]
                cat_s = df[b] if a in num_cols else df[a]
                out.loc[a, b] = _eta(num_s, cat_s)
    return out


def run_comprehensive_eda(
    data: pd.DataFrame,
    target_column: str,
    dataset_identifier: str,
    dataset_name: str = None,
    categorical_columns: list = None,
    verbose: bool = True,
    flagged_columns: set = None,
    full_data: pd.DataFrame = None,
) -> dict:
    """
    Run comprehensive EDA on a dataset, consolidating all Section 2 EDA chunks.

    This function combines:
    - CHUNK_2_1_4_A: Dataset overview stats
    - CHUNK_2_1_4_B: Column analysis -> column_analysis.csv
    - CHUNK_2_1_4_C: Target variable analysis -> target_analysis.csv, target_balance_metrics.csv
    - CHUNK_2_1_4_D: Feature distributions -> feature_distributions.png
    - CHUNK_2_1_4_E: Association analysis -> mixed_association_heatmap.png, association_matrix.csv
    - CHUNK_2_1_4_F: Global config validation

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to analyze
    target_column : str
        Name of the target column
    dataset_identifier : str
        Identifier for the dataset (used in folder structure)
    dataset_name : str, optional
        Display name for the dataset (defaults to dataset_identifier.title())
    categorical_columns : list, optional
        List of categorical column names. If None, auto-detected.
    verbose : bool
        Whether to print progress messages (default: True)

    Returns:
    --------
    dict : Results dictionary with structure:
        {
            'overview_stats': dict,
            'column_analysis': pd.DataFrame,
            'target_analysis': pd.DataFrame,
            'target_balance_metrics': pd.DataFrame,
            'correlation_matrix': pd.DataFrame,
            'target_correlations': pd.DataFrame,
            'categorical_columns': list,
            'files_generated': list,
            'results_path': str
        }

    Example:
    --------
    >>> from src.data.eda import run_comprehensive_eda
    >>> eda_results = run_comprehensive_eda(
    ...     data=data,
    ...     target_column='diagnosis',
    ...     dataset_identifier='breast-cancer-data',
    ...     verbose=True
    ... )
    >>> print(f"Generated {len(eda_results['files_generated'])} files")
    """

    if dataset_name is None:
        dataset_name = dataset_identifier.replace('-', ' ').replace('_', ' ').title()

    # Get results path
    results_path = get_results_path(dataset_identifier, section_number=2)
    os.makedirs(results_path, exist_ok=True)

    files_generated = []
    results = {
        'results_path': results_path,
        'dataset_identifier': dataset_identifier,
        'dataset_name': dataset_name,
        'target_column': target_column
    }

    if verbose:
        print("=" * 60)
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        print(f"Dataset: {dataset_name}")
        print(f"Target column: {target_column}")
        print(f"Results path: {results_path}")
        print("=" * 60)

    # =========================================================================
    # CHUNK_2_1_4_A: Dataset Overview Stats
    # =========================================================================
    if verbose:
        print("\n[1/6] Dataset Overview...")

    overview_stats = {
        'Dataset Name': dataset_name,
        'Shape': f"{data.shape[0]} rows x {data.shape[1]} columns",
        'Memory Usage': f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'Total Missing Values': data.isnull().sum().sum(),
        'Missing Percentage': f"{(data.isnull().sum().sum() / data.size) * 100:.2f}%",
        'Duplicate Rows': data.duplicated().sum(),
        'Numeric Columns': len(data.select_dtypes(include=[np.number]).columns),
        'Categorical Columns': len(data.select_dtypes(include=['object']).columns)
    }

    results['overview_stats'] = overview_stats

    if verbose:
        for key, value in overview_stats.items():
            print(f"   {key:.<30} {value}")

    # =========================================================================
    # CHUNK_2_1_4_B: Column Analysis
    # =========================================================================
    if verbose:
        print("\n[2/6] Column Analysis...")

    column_analysis = pd.DataFrame({
        'Column': data.columns,
        'Data_Type': data.dtypes.astype(str),
        'Unique_Values': [data[col].nunique() for col in data.columns],
        'Missing_Count': [data[col].isnull().sum() for col in data.columns],
        'Missing_Percent': [f"{(data[col].isnull().sum()/len(data)*100):.2f}%" for col in data.columns],
        'Min_Value': [data[col].min() if data[col].dtype in ['int64', 'float64'] else 'N/A' for col in data.columns],
        'Max_Value': [data[col].max() if data[col].dtype in ['int64', 'float64'] else 'N/A' for col in data.columns]
    })

    column_analysis_file = os.path.join(results_path, 'column_analysis.csv')
    column_analysis.to_csv(column_analysis_file, index=False)
    files_generated.append(column_analysis_file)
    results['column_analysis'] = column_analysis

    if verbose:
        print(f"   Saved: column_analysis.csv ({len(data.columns)} columns)")

    # =========================================================================
    # CHUNK_2_1_4_C: Target Variable Analysis
    # =========================================================================
    if verbose:
        print("\n[3/6] Target Variable Analysis...")

    target_analysis = None
    target_balance_metrics = None

    if target_column in data.columns:
        target_counts = data[target_column].value_counts().sort_index()
        target_props = data[target_column].value_counts(normalize=True).sort_index() * 100

        # Create class descriptions
        if len(target_counts) == 2:
            descriptions = ['Class 0', 'Class 1']
        else:
            descriptions = [f'Class {i}' for i in target_counts.index]

        target_analysis = pd.DataFrame({
            'Class': target_counts.index,
            'Count': target_counts.values,
            'Percentage': [f"{prop:.1f}%" for prop in target_props.values],
            'Description': descriptions
        })

        target_analysis_file = os.path.join(results_path, 'target_analysis.csv')
        target_analysis.to_csv(target_analysis_file, index=False)
        files_generated.append(target_analysis_file)
        results['target_analysis'] = target_analysis

        # Calculate class balance metrics
        balance_ratio = target_counts.min() / target_counts.max()
        balance_category = 'Balanced' if balance_ratio > 0.8 else 'Moderately Imbalanced' if balance_ratio > 0.5 else 'Highly Imbalanced'

        target_balance_metrics = pd.DataFrame({
            'Metric': ['Class_Balance_Ratio', 'Dataset_Balance_Category'],
            'Value': [f"{balance_ratio:.3f}", balance_category]
        })

        balance_file = os.path.join(results_path, 'target_balance_metrics.csv')
        target_balance_metrics.to_csv(balance_file, index=False)
        files_generated.append(balance_file)
        results['target_balance_metrics'] = target_balance_metrics
        results['balance_ratio'] = balance_ratio
        results['balance_category'] = balance_category

        if verbose:
            print(f"   Saved: target_analysis.csv")
            print(f"   Saved: target_balance_metrics.csv")
            print(f"   Balance Ratio: {balance_ratio:.3f} ({balance_category})")
    else:
        if verbose:
            print(f"   WARNING: Target column '{target_column}' not found!")

    # =========================================================================
    # CHUNK_2_1_4_D: Feature Distribution Visualizations
    # =========================================================================
    if verbose:
        print("\n[4/6] Feature Distributions...")

    # Set non-interactive backend
    matplotlib.use('Agg')
    plt.ioff()

    # Get numeric columns excluding target
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_no_target = [col for col in numeric_cols if col != target_column]

    if numeric_cols_no_target:
        # Use the existing visualization function
        dist_files = create_feature_distributions(
            data=data,
            target_column=target_column,
            results_path=results_path,
            plots_per_file=6,
            verbose=False
        )
        files_generated.extend(dist_files)
        results['distribution_files'] = dist_files

        if verbose:
            print(f"   Saved: {len(dist_files)} distribution file(s) ({len(numeric_cols_no_target)} features)")
    else:
        if verbose:
            print("   WARNING: No numeric features found for visualization")
        results['distribution_files'] = []

    # =========================================================================
    # CHUNK_2_1_4_E: Correlation Analysis
    # =========================================================================
    if verbose:
        print("\n[5/6] Correlation Analysis...")

    correlation_matrix = None
    target_correlations = None

    if len(numeric_cols_no_target) > 1:
        # Mixed-association matrix: Pearson (num-num), Cramér's V (cat-cat),
        # correlation ratio η (num-cat) — replaces numeric-only Pearson
        from src.evaluation.association import compute_mixed_association_matrix

        # Include all feature columns + target for association analysis
        cols_for_corr = [c for c in data.columns if c != target_column]
        if target_column in data.columns:
            cols_for_corr.append(target_column)

        correlation_matrix = compute_mixed_association_matrix(data[cols_for_corr])

        # When full_data is provided, emit two standalone heatmaps: full (pre-
        # collinearity-reduction) and reduced (post). Otherwise emit the single
        # heatmap under the legacy filename to stay backward compatible.
        if full_data is not None:
            reduced_filename = 'mixed_association_heatmap_reduced.png'
            reduced_csv = 'association_matrix_reduced.csv'
        else:
            reduced_filename = 'mixed_association_heatmap.png'
            reduced_csv = 'association_matrix.csv'

        heatmap_path = create_mixed_association_heatmap(
            association_matrix=correlation_matrix,
            results_path=results_path,
            filename=reduced_filename,
            verbose=False,
            flagged_columns=flagged_columns,
        )
        files_generated.append(heatmap_path)

        corr_matrix_file = os.path.join(results_path, reduced_csv)
        correlation_matrix.to_csv(corr_matrix_file)
        files_generated.append(corr_matrix_file)

        results['correlation_matrix'] = correlation_matrix

        if verbose:
            print(f"   Saved: {reduced_filename}")
            print(f"   Saved: {reduced_csv}")

        # Full (pre-reduction) view for side-by-side diagnosis.
        if full_data is not None:
            full_cols = [c for c in full_data.columns if c != target_column]
            if target_column in full_data.columns:
                full_cols.append(target_column)
            full_matrix = compute_mixed_association_matrix(full_data[full_cols])

            full_heatmap = create_mixed_association_heatmap(
                association_matrix=full_matrix,
                results_path=results_path,
                filename='mixed_association_heatmap_full.png',
                verbose=False,
                flagged_columns=flagged_columns,
            )
            files_generated.append(full_heatmap)

            full_csv = os.path.join(results_path, 'association_matrix_full.csv')
            full_matrix.to_csv(full_csv)
            files_generated.append(full_csv)

            results['correlation_matrix_full'] = full_matrix

            if verbose:
                print(f"   Saved: mixed_association_heatmap_full.png")
                print(f"   Saved: association_matrix_full.csv")

        # Target correlation analysis
        if target_column in correlation_matrix.columns:
            target_corrs = correlation_matrix[target_column].abs().sort_values(ascending=False)
            target_corrs = target_corrs[target_corrs.index != target_column]

            target_correlations = pd.DataFrame({
                'Feature': target_corrs.index,
                'Absolute_Correlation': target_corrs.values,
                'Raw_Correlation': [correlation_matrix.loc[feat, target_column] for feat in target_corrs.index],
                'Strength': ['Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
                            for corr in target_corrs.values]
            })

            target_corr_file = os.path.join(results_path, 'target_associations.csv')
            target_correlations.to_csv(target_corr_file, index=False)
            files_generated.append(target_corr_file)
            results['target_correlations'] = target_correlations

            if verbose:
                print(f"   Saved: target_associations.csv ({len(target_corrs)} features)")
    else:
        if verbose:
            print("   WARNING: Insufficient numeric features for correlation analysis")

    # =========================================================================
    # CHUNK_2_1_4_F: Global Config Validation & Categorical Column Detection
    # =========================================================================
    if verbose:
        print("\n[6/6] Configuration Validation...")

    # Auto-detect categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        if target_column in categorical_columns:
            categorical_columns.remove(target_column)

    results['categorical_columns'] = categorical_columns
    results['numeric_columns'] = numeric_cols_no_target
    results['files_generated'] = files_generated

    if verbose:
        print(f"   Categorical columns: {categorical_columns if categorical_columns else 'None'}")
        print(f"   Numeric columns: {len(numeric_cols_no_target)}")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("EDA COMPLETE")
        print("=" * 60)
        print(f"Files generated: {len(files_generated)}")
        for f in files_generated:
            print(f"   - {os.path.basename(f)}")
        print(f"Results saved to: {results_path}")
        print("=" * 60)

    return results


print("[OK] EDA module loaded from src/data/eda.py")
