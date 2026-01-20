"""
Data Preprocessing Functions

Handles data cleaning, categorical encoding, and preprocessing for model training.
Migrated from setup.py Phase 2 (Task 4.3 Migration Plan).

Functions migrated:
- get_categorical_columns_for_models() - from setup.py line 231
- clean_and_preprocess_data() - from setup.py line 261
- prepare_data_for_any_model() - from setup.py line 407
- prepare_data_for_hyperparameter_optimization() - from setup.py line 3300
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_categorical_columns_for_models():
    """
    Robust categorical column detection for consistent model training.
    Works across Section 3 demos AND Section 4 hyperparameter optimization.

    Returns:
        list: List of categorical column names, or empty list if no categorical columns
              NEVER returns None to prevent 'NoneType' object is not iterable errors
    """
    # Try to get from global scope first (set in CHUNK_014)
    if 'categorical_columns' in globals():
        cats = globals()['categorical_columns']
        # Return the list if it has items, empty list if None or empty
        return cats if cats is not None else []

    # Fallback: auto-detect from data
    if 'data' in globals() and globals()['data'] is not None:
        try:
            auto_detected = globals()['data'].select_dtypes(include=['object']).columns.tolist()
            # Remove target column if it's in categorical columns
            if 'TARGET_COLUMN' in globals() and globals()['TARGET_COLUMN'] and globals()['TARGET_COLUMN'] in auto_detected:
                auto_detected.remove(globals()['TARGET_COLUMN'])
            return auto_detected
        except Exception as e:
            print(f"[WARNING] Auto-detection of categorical columns failed: {e}")
            return []

    # Always return empty list, never None
    return []


def clean_and_preprocess_data(data, categorical_columns=None):
    """
    Comprehensive data cleaning and preprocessing to prevent model training errors.
    Handles NaN/None values, categorical encoding, and data type validation.

    Args:
        data (pd.DataFrame): Input dataset
        categorical_columns (list, optional): List of categorical column names

    Returns:
        tuple: (cleaned_data, categorical_columns, encoders_dict)
    """
    # Work on a copy to avoid modifying original data
    cleaned_data = data.copy()
    encoders_dict = {}

    # Auto-detect categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = get_categorical_columns_for_models()

    print(f"[DATA_CLEANING] Processing {len(cleaned_data)} rows, {len(cleaned_data.columns)} columns")
    print(f"[DATA_CLEANING] Categorical columns: {categorical_columns}")

    # Step 1: Handle missing values
    missing_counts = cleaned_data.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"[DATA_CLEANING] Found {missing_counts.sum()} missing values across {(missing_counts > 0).sum()} columns")

        for col in cleaned_data.columns:
            if missing_counts[col] > 0:
                if col in categorical_columns or cleaned_data[col].dtype == 'object':
                    # Fill categorical missing values with mode or 'Unknown'
                    mode_val = cleaned_data[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    cleaned_data[col].fillna(fill_val, inplace=True)
                    print(f"[DATA_CLEANING] Filled {missing_counts[col]} missing values in categorical column '{col}' with '{fill_val}'")
                else:
                    # Fill numerical missing values with median
                    median_val = cleaned_data[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0  # Fallback if all values are NaN
                    cleaned_data[col].fillna(median_val, inplace=True)
                    print(f"[DATA_CLEANING] Filled {missing_counts[col]} missing values in numerical column '{col}' with {median_val}")

    # Step 2: Smart categorical encoding (binary vs one-hot vs label encoding)
    for col in categorical_columns:
        if col in cleaned_data.columns and cleaned_data[col].dtype == 'object':
            try:
                # Convert to string to handle any remaining None/NaN values
                cleaned_data[col] = cleaned_data[col].astype(str)

                # Get unique values count for encoding strategy
                unique_count = cleaned_data[col].nunique()
                unique_values = cleaned_data[col].unique()

                if unique_count == 2:
                    # Binary encoding: Use LabelEncoder for 0/1 encoding
                    le = LabelEncoder()
                    cleaned_data[col] = le.fit_transform(cleaned_data[col])
                    encoders_dict[col] = {
                        'type': 'binary',
                        'encoder': le,
                        'original_column': col
                    }
                    print(f"[DATA_CLEANING] Binary encoded column '{col}' (2 values: {list(unique_values)} → 0/1)")

                elif unique_count <= 10:
                    # Multi-level encoding: Use one-hot encoding
                    # First apply label encoding to get numeric values
                    le = LabelEncoder()
                    temp_encoded = le.fit_transform(cleaned_data[col])

                    # Create one-hot encoded columns
                    one_hot_df = pd.get_dummies(cleaned_data[col], prefix=col, dtype=int)

                    # Store the original column and add one-hot columns
                    original_col_data = cleaned_data[col].copy()
                    cleaned_data = cleaned_data.drop(columns=[col])
                    cleaned_data = pd.concat([cleaned_data, one_hot_df], axis=1)

                    # Store encoding info for reverse transformation
                    encoders_dict[col] = {
                        'type': 'onehot',
                        'encoder': le,
                        'original_column': col,
                        'onehot_columns': list(one_hot_df.columns),
                        'original_values': list(unique_values)
                    }
                    print(f"[DATA_CLEANING] One-hot encoded column '{col}' ({unique_count} values → {len(one_hot_df.columns)} columns)")

                else:
                    # High-cardinality: Use label encoding (fallback to current approach)
                    le = LabelEncoder()
                    cleaned_data[col] = le.fit_transform(cleaned_data[col])
                    encoders_dict[col] = {
                        'type': 'label',
                        'encoder': le,
                        'original_column': col
                    }
                    print(f"[DATA_CLEANING] Label encoded column '{col}' ({unique_count} unique values)")

            except Exception as e:
                print(f"[DATA_CLEANING] Warning: Failed to encode column '{col}': {e}")
                # Fallback to simple label encoding
                try:
                    le = LabelEncoder()
                    cleaned_data[col] = le.fit_transform(cleaned_data[col].astype(str))
                    encoders_dict[col] = {
                        'type': 'label_fallback',
                        'encoder': le,
                        'original_column': col
                    }
                    print(f"[DATA_CLEANING] Fallback label encoding applied to '{col}'")
                except Exception as fallback_error:
                    print(f"[DATA_CLEANING] Error: Failed to encode column '{col}' even with fallback: {fallback_error}")

    # Step 3: Ensure all numerical columns are proper numeric types
    for col in cleaned_data.columns:
        if col not in categorical_columns:
            try:
                # Try to convert to numeric, forcing errors to NaN
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')

                # Fill any new NaN values created by conversion with 0
                if cleaned_data[col].isnull().any():
                    nan_count = cleaned_data[col].isnull().sum()
                    cleaned_data[col].fillna(0.0, inplace=True)
                    print(f"[DATA_CLEANING] Converted and filled {nan_count} non-numeric values in column '{col}'")
            except Exception as e:
                print(f"[DATA_CLEANING] Warning: Failed to convert column '{col}' to numeric: {e}")

    # Step 4: Final validation
    remaining_nulls = cleaned_data.isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"[DATA_CLEANING] Warning: {remaining_nulls} null values remain after cleaning")

    print(f"[DATA_CLEANING] Data cleaning completed successfully")
    print(f"[DATA_CLEANING] Final shape: {cleaned_data.shape}")
    print(f"[DATA_CLEANING] Data types: {dict(cleaned_data.dtypes)}")

    return cleaned_data, categorical_columns, encoders_dict


def prepare_data_for_any_model(data, categorical_columns=None, model_name="Model"):
    """
    Universal data preparation function that can be called from notebooks.
    Handles all the preprocessing needed for robust model training.

    Args:
        data (pd.DataFrame): Input dataset
        categorical_columns (list, optional): List of categorical column names
        model_name (str): Name of the model for logging purposes

    Returns:
        tuple: (cleaned_data, categorical_columns_used, preprocessing_info)
    """
    print(f"\n[{model_name.upper()}] Preparing data for training...")

    # Apply comprehensive preprocessing
    cleaned_data, categorical_cols_used, encoders_dict = clean_and_preprocess_data(
        data, categorical_columns
    )

    # Create preprocessing info for potential reverse transformation
    preprocessing_info = {
        'encoders': encoders_dict,
        'categorical_columns': categorical_cols_used,
        'original_columns': list(data.columns),
        'cleaned_shape': cleaned_data.shape,
        'original_shape': data.shape
    }

    print(f"[{model_name.upper()}] Data preparation completed:")
    print(f"   • Original shape: {data.shape}")
    print(f"   • Cleaned shape: {cleaned_data.shape}")
    print(f"   • Categorical columns: {categorical_cols_used}")
    print(f"   • Missing values handled: {(data.isnull().sum().sum() - cleaned_data.isnull().sum().sum())}")

    return cleaned_data, categorical_cols_used, preprocessing_info


def prepare_data_for_hyperparameter_optimization(data, categorical_columns=None):
    """
    Prepare data for CTGAN hyperparameter optimization by preprocessing categorical variables.

    This function ensures that categorical data is properly encoded so CTGAN doesn't try
    to treat strings like 'Female'/'Male' as continuous numerical variables.

    Parameters:
    - data: pandas DataFrame with raw data
    - categorical_columns: list of categorical column names (optional, auto-detected if None)

    Returns:
    - processed_data: DataFrame with categorical variables encoded as numeric
    - discrete_columns: list of column names to pass to CTGAN as discrete_columns
    - encoders: dict of LabelEncoders for reverse transformation if needed
    """
    try:
        print(f"[HYPEROPT_PREP] Preparing data for hyperparameter optimization...")
        print(f"[HYPEROPT_PREP] Input data shape: {data.shape}")

        # Make a copy to avoid modifying original data
        processed_data = data.copy()

        # Auto-detect categorical columns if not provided
        if categorical_columns is None:
            categorical_columns = []
            for col in processed_data.columns:
                if processed_data[col].dtype == 'object' or processed_data[col].dtype.name == 'category':
                    categorical_columns.append(col)
            print(f"[HYPEROPT_PREP] Auto-detected categorical columns: {categorical_columns}")
        else:
            print(f"[HYPEROPT_PREP] Using provided categorical columns: {categorical_columns}")

        # Track encoded columns and store encoders
        discrete_columns = []
        encoders = {}

        # Process categorical columns
        for col in categorical_columns:
            if col in processed_data.columns:
                print(f"[HYPEROPT_PREP] Encoding categorical column: {col}")

                # Handle missing values first
                processed_data[col] = processed_data[col].fillna('Unknown')

                # Create and fit label encoder
                encoder = LabelEncoder()
                processed_data[col] = encoder.fit_transform(processed_data[col].astype(str))

                # Store encoder and mark as discrete
                encoders[col] = encoder
                discrete_columns.append(col)

                print(f"[HYPEROPT_PREP] Column '{col}' encoded: {len(encoder.classes_)} unique values")

        # Handle any remaining missing values in numeric columns
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if processed_data[col].isnull().any():
                median_val = processed_data[col].median()
                processed_data[col] = processed_data[col].fillna(median_val)
                print(f"[HYPEROPT_PREP] Filled {processed_data[col].isnull().sum()} missing values in numeric column '{col}' with median: {median_val}")

        # Ensure all data is numeric
        for col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                try:
                    processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                    processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                    print(f"[HYPEROPT_PREP] Converted column '{col}' to numeric")
                except Exception as e:
                    print(f"[WARNING] Could not convert column '{col}' to numeric: {e}")

        print(f"[HYPEROPT_PREP] Final data shape: {processed_data.shape}")
        print(f"[HYPEROPT_PREP] Discrete columns for CTGAN: {discrete_columns}")
        print(f"[HYPEROPT_PREP] Data types: {processed_data.dtypes.value_counts().to_dict()}")
        print(f"[HYPEROPT_PREP] Missing values: {processed_data.isnull().sum().sum()}")

        return processed_data, discrete_columns, encoders

    except Exception as e:
        print(f"[HYPEROPT_PREP] ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Return original data with empty discrete columns as fallback
        return data, [], {}


# ============================================================================
# CONFIG-DRIVEN PREPROCESSING (Phase 3 - January 2026)
# ============================================================================

from typing import Dict, Any, Tuple, List, Optional
import warnings


def _detect_categorical_columns(df: pd.DataFrame, target_column: Optional[str] = None) -> List[str]:
    """
    Auto-detect categorical columns in a DataFrame.

    Args:
        df: Input DataFrame
        target_column: Target column to exclude from detection

    Returns:
        List of categorical column names
    """
    categorical_cols = []

    for col in df.columns:
        if col == target_column:
            continue

        dtype = df[col].dtype

        # Object or category type
        if dtype == 'object' or dtype.name == 'category':
            categorical_cols.append(col)
        # Boolean
        elif dtype == 'bool':
            categorical_cols.append(col)
        # Integer with few unique values (likely categorical)
        elif dtype in ['int64', 'int32'] and df[col].nunique() <= 20:
            categorical_cols.append(col)

    return categorical_cols


def _detect_task_type(df: pd.DataFrame, target_column: str, max_classes: int = 20) -> str:
    """
    Auto-detect whether the task is classification or regression.

    Args:
        df: Input DataFrame
        target_column: Name of target column
        max_classes: Max unique values to consider as classification

    Returns:
        "classification" or "regression"
    """
    if target_column not in df.columns:
        return "classification"  # Default

    target = df[target_column]
    dtype = target.dtype
    n_unique = target.nunique()

    # Categorical types are classification
    if dtype == 'object' or dtype.name == 'category' or dtype == 'bool':
        return "classification"

    # Few unique values suggest classification
    if n_unique <= max_classes:
        return "classification"

    # Many unique values suggest regression
    return "regression"


def _apply_missing_strategy(
    df: pd.DataFrame,
    strategy: str,
    categorical_columns: List[str],
    target_column: Optional[str] = None,
    mice_max_iter: int = 10
) -> pd.DataFrame:
    """
    Apply a missing data handling strategy.

    Args:
        df: Input DataFrame
        strategy: One of "none", "drop", "median", "mode", "mice", "indicator_onehot"
        categorical_columns: List of categorical column names
        target_column: Target column (handled specially)
        mice_max_iter: Max iterations for MICE imputation

    Returns:
        DataFrame with missing values handled
    """
    result = df.copy()

    if strategy == "none":
        return result

    if strategy == "drop":
        rows_before = len(result)
        result = result.dropna()
        rows_dropped = rows_before - len(result)
        if rows_dropped > 0:
            print(f"[PREPROCESS] Dropped {rows_dropped} rows with missing values")
        return result

    if strategy == "median":
        # Fill numeric columns with median
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if result[col].isna().any():
                median_val = result[col].median()
                result[col] = result[col].fillna(median_val)
        return result

    if strategy == "mode":
        # Fill all columns with mode
        for col in result.columns:
            if result[col].isna().any():
                mode_vals = result[col].mode()
                if len(mode_vals) > 0:
                    result[col] = result[col].fillna(mode_vals[0])
        return result

    if strategy == "mice":
        # MICE imputation for numeric, mode for categorical
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            from sklearn.ensemble import RandomForestRegressor

            # Separate numeric and categorical
            numeric_cols = [c for c in result.columns if c not in categorical_columns and pd.api.types.is_numeric_dtype(result[c])]
            cat_cols = [c for c in result.columns if c in categorical_columns or not pd.api.types.is_numeric_dtype(result[c])]

            # MICE for numeric columns
            if numeric_cols and result[numeric_cols].isna().any().any():
                imputer = IterativeImputer(
                    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
                    max_iter=mice_max_iter,
                    random_state=42
                )
                result[numeric_cols] = imputer.fit_transform(result[numeric_cols])
                print(f"[PREPROCESS] Applied MICE imputation to {len(numeric_cols)} numeric columns")

            # Mode for categorical columns
            for col in cat_cols:
                if result[col].isna().any():
                    mode_vals = result[col].mode()
                    if len(mode_vals) > 0:
                        result[col] = result[col].fillna(mode_vals[0])

        except ImportError:
            warnings.warn("sklearn IterativeImputer not available, falling back to median/mode")
            return _apply_missing_strategy(result, "median", categorical_columns, target_column)

        return result

    if strategy == "indicator_onehot":
        # Step 1: Add missingness indicator columns
        cols_with_na = [col for col in result.columns if result[col].isna().any()]
        for col in cols_with_na:
            indicator_col = f"{col}__is_missing"
            result[indicator_col] = result[col].isna().astype(int)
            print(f"[PREPROCESS] Added missingness indicator: {indicator_col}")

        # Step 2: Impute remaining NA (median for numeric, mode for categorical)
        for col in result.columns:
            if col.endswith("__is_missing"):
                continue
            if result[col].isna().any():
                if col in categorical_columns or result[col].dtype == 'object':
                    mode_vals = result[col].mode()
                    fill_val = mode_vals[0] if len(mode_vals) > 0 else "Unknown"
                    result[col] = result[col].fillna(fill_val)
                else:
                    result[col] = result[col].fillna(result[col].median())

        # Step 3: One-hot encode categorical columns (NOT the target)
        cat_cols_to_encode = [c for c in categorical_columns if c != target_column and c in result.columns]
        if cat_cols_to_encode:
            result = pd.get_dummies(result, columns=cat_cols_to_encode, dtype=int)
            print(f"[PREPROCESS] One-hot encoded {len(cat_cols_to_encode)} categorical columns")

        return result

    warnings.warn(f"Unknown missing strategy '{strategy}', using 'none'")
    return result


def preprocess_dataset(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess a dataset according to the notebook configuration.

    This is the main entry point for config-driven preprocessing.

    Args:
        df: Input DataFrame
        config: Notebook configuration dictionary (from validate_config)

    Returns:
        Tuple of (processed_df, metadata)
        - processed_df: Preprocessed DataFrame
        - metadata: Dictionary containing:
            - dataset_identifier: Identifier string
            - target_column: Target column name
            - categorical_columns: List of categorical columns
            - task_type: "classification" or "regression"
            - transform_log: List of transformations applied
    """
    from src.utils.paths import extract_dataset_identifier

    result = df.copy()
    transform_log = []

    # Extract config values
    target_column = config.get("target_column")
    categorical_columns = config.get("categorical_columns", [])
    task_type = config.get("task_type", "auto")
    use_row_subset = config.get("use_row_subset", False)
    sample_n = config.get("sample_n", 500)
    sample_random_state = config.get("sample_random_state", 42)
    missing_strategy = config.get("missing_strategy", "none")
    mice_max_iter = config.get("mice_max_iter", 10)
    dataset_identifier_override = config.get("dataset_identifier_override")
    data_file = config.get("data_file")

    print(f"[PREPROCESS] Starting preprocessing pipeline")
    print(f"[PREPROCESS] Input shape: {result.shape}")

    # Step 1: Validate target column exists
    if target_column and target_column not in result.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame. Available: {list(result.columns)}")

    # Step 2: Standardize column names (remove special chars, lowercase)
    original_columns = list(result.columns)
    result.columns = [str(col).strip().replace(" ", "_").lower() for col in result.columns]

    # Update target_column if it was renamed
    if target_column:
        target_column_clean = str(target_column).strip().replace(" ", "_").lower()
        if target_column_clean != target_column:
            transform_log.append(f"Renamed target column: {target_column} -> {target_column_clean}")
            target_column = target_column_clean

    if original_columns != list(result.columns):
        transform_log.append(f"Standardized {len(result.columns)} column names")

    # Step 3: Auto-detect categorical columns if not provided
    if not categorical_columns:
        categorical_columns = _detect_categorical_columns(result, target_column)
        transform_log.append(f"Auto-detected {len(categorical_columns)} categorical columns")
    else:
        # Clean provided categorical column names
        categorical_columns = [str(c).strip().replace(" ", "_").lower() for c in categorical_columns]

    print(f"[PREPROCESS] Categorical columns: {categorical_columns}")

    # Step 4: Auto-detect task type if needed
    if task_type == "auto" and target_column:
        task_type = _detect_task_type(result, target_column)
        transform_log.append(f"Auto-detected task type: {task_type}")

    print(f"[PREPROCESS] Task type: {task_type}")

    # Step 5: Apply missing data strategy
    if missing_strategy != "none":
        na_before = result.isna().sum().sum()
        result = _apply_missing_strategy(
            result,
            missing_strategy,
            categorical_columns,
            target_column,
            mice_max_iter
        )
        na_after = result.isna().sum().sum()
        transform_log.append(f"Applied missing strategy '{missing_strategy}': {na_before} -> {na_after} NA values")
        print(f"[PREPROCESS] Missing values: {na_before} -> {na_after}")

    # Step 6: Subset rows if requested
    if use_row_subset and len(result) > sample_n:
        result = result.sample(n=sample_n, random_state=sample_random_state)
        transform_log.append(f"Subsampled to {sample_n} rows")
        print(f"[PREPROCESS] Subsampled to {sample_n} rows")

    # Step 7: Determine dataset identifier
    if dataset_identifier_override:
        dataset_identifier = dataset_identifier_override
    elif data_file:
        dataset_identifier = extract_dataset_identifier(data_file)
    else:
        dataset_identifier = "unknown-dataset"

    print(f"[PREPROCESS] Final shape: {result.shape}")
    print(f"[PREPROCESS] Dataset identifier: {dataset_identifier}")

    # Build metadata
    metadata = {
        "dataset_identifier": dataset_identifier,
        "target_column": target_column,
        "categorical_columns": categorical_columns,
        "task_type": task_type,
        "transform_log": transform_log,
        "original_shape": df.shape,
        "processed_shape": result.shape,
        "columns": list(result.columns)
    }

    return result, metadata


def load_and_preprocess_from_config(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, List[str], Dict[str, Any]]:
    """
    Load data from file and preprocess according to config.

    This is a convenience function that combines loading and preprocessing.

    Args:
        config: Validated notebook configuration dictionary

    Returns:
        Tuple of:
            - data: Processed DataFrame
            - original_data: Copy of data before modeling (same as data after preprocessing)
            - target_column: Target column name
            - DATASET_IDENTIFIER: Dataset identifier string
            - categorical_columns: List of categorical column names
            - metadata: Full preprocessing metadata
    """
    from src.config import validate_config

    # Ensure config is validated
    config = validate_config(config)

    data_file = config.get("data_file")
    if not data_file:
        raise ValueError("'data_file' is required in config")

    # Load the data
    print(f"[LOAD] Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"[LOAD] Loaded {len(df)} rows, {len(df.columns)} columns")

    # Preprocess
    data, metadata = preprocess_dataset(df, config)

    # Create outputs matching notebook contract
    original_data = data.copy()
    target_column = metadata["target_column"]
    DATASET_IDENTIFIER = metadata["dataset_identifier"]
    categorical_columns = metadata["categorical_columns"]

    return data, original_data, target_column, DATASET_IDENTIFIER, categorical_columns, metadata


print("[OK] Config-driven preprocessing functions loaded (Phase 3)")
