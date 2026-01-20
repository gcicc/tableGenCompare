"""
Target Integrity and Data Sanitization

This module provides functions to ensure synthetic data maintains target column
integrity and handles numeric edge cases (inf, NaN) that can break downstream
evaluation metrics.

Addresses known issues:
- CTABGAN producing continuous target values (breaks classification metrics)
- TVAE producing NaN/inf values (breaks correlation calculations)
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Literal
import warnings
import logging

logger = logging.getLogger(__name__)


def sanitize_numeric(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    inf_replacement: Literal["nan", "clip", "median"] = "nan",
    nan_strategy: Literal["median", "mean", "drop", "none"] = "median",
    clip_percentile: float = 99.9,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Sanitize numeric columns by handling inf and NaN values.

    This function addresses TVAE and other models that may produce inf/-inf
    or NaN values which break correlation and other metric calculations.

    Parameters:
        df: DataFrame to sanitize
        columns: Specific columns to sanitize (None = all numeric columns)
        inf_replacement: How to handle inf values:
            - "nan": Replace inf with NaN (then handle via nan_strategy)
            - "clip": Clip to percentile bounds
            - "median": Replace with column median
        nan_strategy: How to handle NaN values:
            - "median": Replace with column median
            - "mean": Replace with column mean
            - "drop": Drop rows with NaN
            - "none": Leave NaN as-is
        clip_percentile: Percentile for clipping (default 99.9)
        verbose: Print information about sanitization

    Returns:
        Sanitized DataFrame
    """
    result = df.copy()

    # Determine columns to process
    if columns is None:
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [c for c in columns if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]

    if not numeric_cols:
        return result

    inf_count = 0
    nan_count_before = result[numeric_cols].isna().sum().sum()

    # Handle infinities
    for col in numeric_cols:
        col_inf_mask = np.isinf(result[col])
        col_inf_count = col_inf_mask.sum()

        if col_inf_count > 0:
            inf_count += col_inf_count

            if inf_replacement == "nan":
                result.loc[col_inf_mask, col] = np.nan
            elif inf_replacement == "clip":
                finite_vals = result.loc[~col_inf_mask, col]
                if len(finite_vals) > 0:
                    lower = np.percentile(finite_vals, 100 - clip_percentile)
                    upper = np.percentile(finite_vals, clip_percentile)
                    result.loc[result[col] == np.inf, col] = upper
                    result.loc[result[col] == -np.inf, col] = lower
            elif inf_replacement == "median":
                finite_vals = result.loc[~col_inf_mask, col]
                if len(finite_vals) > 0:
                    result.loc[col_inf_mask, col] = finite_vals.median()

    # Handle NaN values
    nan_count_after_inf = result[numeric_cols].isna().sum().sum()

    if nan_strategy == "median":
        for col in numeric_cols:
            if result[col].isna().any():
                median_val = result[col].median()
                if pd.notna(median_val):
                    result[col] = result[col].fillna(median_val)
    elif nan_strategy == "mean":
        for col in numeric_cols:
            if result[col].isna().any():
                mean_val = result[col].mean()
                if pd.notna(mean_val):
                    result[col] = result[col].fillna(mean_val)
    elif nan_strategy == "drop":
        rows_before = len(result)
        result = result.dropna(subset=numeric_cols)
        rows_dropped = rows_before - len(result)
        if verbose and rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped} rows with NaN values")
    # "none" - do nothing

    nan_count_final = result[numeric_cols].isna().sum().sum()

    if verbose and (inf_count > 0 or nan_count_before > 0):
        logger.info(f"Sanitized {len(numeric_cols)} numeric columns:")
        logger.info(f"  - Infinities replaced: {inf_count}")
        logger.info(f"  - NaN before: {nan_count_before}, after inf handling: {nan_count_after_inf}, final: {nan_count_final}")

    return result


def enforce_target_schema(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_column: str,
    task_type: Optional[str] = None,
    max_classes_for_classification: int = 20,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Enforce that synthetic data target column matches the schema of real data.

    This function addresses CTABGAN and other models that may produce continuous
    values for classification targets, which breaks downstream metrics.

    Parameters:
        real_df: Real/original DataFrame (used as schema reference)
        synth_df: Synthetic DataFrame to fix
        target_column: Name of the target column
        task_type: Task type - "classification", "regression", or "auto" (default)
            If "auto", infers from real data characteristics
        max_classes_for_classification: Max unique values to consider as classification
        verbose: Print information about schema enforcement

    Returns:
        Synthetic DataFrame with corrected target column

    Raises:
        ValueError: If target_column not found in either DataFrame
    """
    if target_column not in real_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in real DataFrame")
    if target_column not in synth_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in synthetic DataFrame")

    result = synth_df.copy()
    real_target = real_df[target_column]
    synth_target = result[target_column]

    # Determine task type
    if task_type is None or task_type == "auto":
        n_unique = real_target.nunique()
        real_dtype = real_target.dtype

        # Classification if: categorical, object, bool, or few unique values
        if real_dtype == 'object' or real_dtype.name == 'category' or real_dtype == bool:
            inferred_task = "classification"
        elif n_unique <= max_classes_for_classification:
            inferred_task = "classification"
        else:
            inferred_task = "regression"

        if verbose:
            logger.info(f"Inferred task type: {inferred_task} (unique values: {n_unique}, dtype: {real_dtype})")
    else:
        inferred_task = task_type

    # Get valid class labels from real data
    valid_labels = real_target.dropna().unique()
    original_dtype = real_target.dtype

    if inferred_task == "classification":
        # Handle classification targets

        # Check if we're dealing with binary classification
        is_binary = len(valid_labels) == 2

        # Check for numeric labels
        try:
            numeric_labels = pd.to_numeric(valid_labels, errors='coerce')
            is_numeric = not np.isnan(numeric_labels).all()
        except:
            is_numeric = False

        if is_numeric:
            numeric_labels = sorted([float(x) for x in valid_labels if pd.notna(x)])

            if is_binary and set(numeric_labels) == {0.0, 1.0}:
                # Binary 0/1 case - clamp and round
                synth_numeric = pd.to_numeric(synth_target, errors='coerce')
                result[target_column] = synth_numeric.clip(0, 1).round().astype(int)

                if verbose:
                    changes = (synth_target != result[target_column]).sum()
                    logger.info(f"Binary target: clamped/rounded {changes} values to {{0, 1}}")
            else:
                # Multi-class numeric - map to nearest valid label
                synth_numeric = pd.to_numeric(synth_target, errors='coerce')

                def map_to_nearest(val):
                    if pd.isna(val):
                        return numeric_labels[0]  # Default to first class
                    distances = [abs(val - label) for label in numeric_labels]
                    return numeric_labels[np.argmin(distances)]

                result[target_column] = synth_numeric.apply(map_to_nearest)

                if verbose:
                    changes = (synth_target != result[target_column]).sum()
                    logger.info(f"Multiclass target: mapped {changes} values to valid classes {numeric_labels}")
        else:
            # Categorical/string labels - map invalid to mode
            mode_label = real_target.mode()[0] if len(real_target.mode()) > 0 else valid_labels[0]
            valid_set = set(valid_labels)

            def map_to_valid(val):
                if val in valid_set:
                    return val
                return mode_label

            result[target_column] = synth_target.apply(map_to_valid)

            if verbose:
                invalid_count = (~synth_target.isin(valid_set)).sum()
                logger.info(f"Categorical target: replaced {invalid_count} invalid values with mode '{mode_label}'")

        # Ensure dtype matches
        try:
            if original_dtype == 'int64' or original_dtype == 'int32':
                result[target_column] = result[target_column].astype(int)
            elif original_dtype == 'float64' or original_dtype == 'float32':
                result[target_column] = result[target_column].astype(float)
            elif original_dtype == 'object':
                result[target_column] = result[target_column].astype(str)
        except (ValueError, TypeError) as e:
            if verbose:
                logger.warning(f"Could not convert target to original dtype {original_dtype}: {e}")

    else:
        # Regression - ensure numeric type and handle outliers
        synth_numeric = pd.to_numeric(synth_target, errors='coerce')

        # Replace NaN with median of real data
        if synth_numeric.isna().any():
            fill_value = real_target.median()
            synth_numeric = synth_numeric.fillna(fill_value)
            if verbose:
                nan_count = synth_target.isna().sum() if hasattr(synth_target, 'isna') else 0
                logger.info(f"Regression target: filled {nan_count} NaN values with median {fill_value}")

        # Optional: clip extreme outliers based on real data range
        real_min, real_max = real_target.min(), real_target.max()
        real_range = real_max - real_min
        tolerance = 0.5  # Allow 50% beyond real range
        clip_min = real_min - tolerance * real_range
        clip_max = real_max + tolerance * real_range

        clipped = synth_numeric.clip(clip_min, clip_max)
        clip_count = ((synth_numeric < clip_min) | (synth_numeric > clip_max)).sum()

        if clip_count > 0 and verbose:
            logger.info(f"Regression target: clipped {clip_count} extreme outliers to [{clip_min:.2f}, {clip_max:.2f}]")

        result[target_column] = clipped.astype(float)

    return result


def sanitize_synthetic_data(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_column: str,
    task_type: Optional[str] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Convenience function that applies both sanitize_numeric and enforce_target_schema.

    This is the recommended function to call after generating synthetic data
    to ensure it's safe for all downstream evaluation metrics.

    Parameters:
        real_df: Real/original DataFrame
        synth_df: Synthetic DataFrame to sanitize
        target_column: Name of the target column
        task_type: Task type - "classification", "regression", or "auto"
        verbose: Print information about sanitization

    Returns:
        Fully sanitized synthetic DataFrame
    """
    # First sanitize numeric columns (handles inf/NaN)
    result = sanitize_numeric(synth_df, verbose=verbose)

    # Then enforce target schema
    result = enforce_target_schema(
        real_df=real_df,
        synth_df=result,
        target_column=target_column,
        task_type=task_type,
        verbose=verbose
    )

    return result


# Module-level print for import confirmation
print("[OK] Target integrity functions loaded from src/data/target_integrity.py")
