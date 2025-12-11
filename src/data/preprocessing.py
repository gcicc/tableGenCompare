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
