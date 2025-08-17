#!/usr/bin/env python3
"""
Test script for the Clinical Synthetic Data Generation Framework
Tests all 4 datasets with sections 1, 2, and 3 to ensure compatibility
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import utilities from the framework
import re
from typing import Dict, List, Tuple, Any

# Column Name Standardization and Dataset Analysis Utilities
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names by removing special characters and normalizing formatting.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    
    # Create mapping of old to new column names
    name_mapping = {}
    
    for col in df.columns:
        # Remove special characters and normalize
        new_name = re.sub(r'[^\w\s]', '', str(col))  # Remove special chars
        new_name = re.sub(r'\s+', '_', new_name.strip())  # Replace spaces with underscores
        new_name = new_name.lower()  # Convert to lowercase
        
        # Handle duplicate names
        if new_name in name_mapping.values():
            counter = 1
            while f"{new_name}_{counter}" in name_mapping.values():
                counter += 1
            new_name = f"{new_name}_{counter}"
            
        name_mapping[col] = new_name
    
    # Rename columns
    df = df.rename(columns=name_mapping)
    
    print(f"Column Name Standardization:")
    for old, new in name_mapping.items():
        if old != new:
            print(f"   '{old}' -> '{new}'")
    
    return df, name_mapping

def detect_target_column(df: pd.DataFrame, target_hint: str = None) -> str:
    """
    Detect the target column in the dataset.
    
    Args:
        df: Input dataframe
        target_hint: User-provided hint for target column name
        
    Returns:
        Name of the detected target column
    """
    # Common target column patterns
    target_patterns = [
        'target', 'label', 'class', 'outcome', 'result', 'diagnosis', 
        'response', 'y', 'dependent', 'output', 'prediction'
    ]
    
    # If user provided hint, try to find it first
    if target_hint:
        # Try exact match (case insensitive)
        for col in df.columns:
            if col.lower() == target_hint.lower():
                print(f"✅ Target column found: '{col}' (user specified)")
                return col
        
        # Try partial match
        for col in df.columns:
            if target_hint.lower() in col.lower():
                print(f"✅ Target column found: '{col}' (partial match to '{target_hint}')")
                return col
    
    # Auto-detect based on patterns
    for pattern in target_patterns:
        for col in df.columns:
            if pattern in col.lower():
                print(f"✅ Target column auto-detected: '{col}' (pattern: '{pattern}')")
                return col
    
    # If no pattern match, check for binary columns (likely targets)
    binary_cols = []
    for col in df.columns:
        unique_vals = df[col].dropna().nunique()
        if unique_vals == 2:
            binary_cols.append(col)
    
    if binary_cols:
        target_col = binary_cols[0]  # Take first binary column
        print(f"✅ Target column inferred: '{target_col}' (binary column)")
        return target_col
    
    # Last resort: use last column
    target_col = df.columns[-1]
    print(f"⚠️ Target column defaulted to: '{target_col}' (last column)")
    return target_col

def analyze_column_types(df: pd.DataFrame, categorical_hint: List[str] = None) -> Dict[str, str]:
    """
    Analyze and categorize column types.
    
    Args:
        df: Input dataframe
        categorical_hint: User-provided list of categorical columns
        
    Returns:
        Dictionary mapping column names to types ('categorical', 'continuous', 'binary')
    """
    column_types = {}
    
    for col in df.columns:
        # Skip if user explicitly specified as categorical
        if categorical_hint and col in categorical_hint:
            column_types[col] = 'categorical'
            continue
            
        # Analyze column characteristics
        non_null_data = df[col].dropna()
        unique_count = non_null_data.nunique()
        total_count = len(non_null_data)
        
        # Determine type based on data characteristics
        if unique_count == 2:
            column_types[col] = 'binary'
        elif df[col].dtype == 'object' or unique_count < 10:
            column_types[col] = 'categorical'
        elif df[col].dtype in ['int64', 'float64'] and unique_count > 10:
            column_types[col] = 'continuous'
        else:
            # Default based on uniqueness ratio
            uniqueness_ratio = unique_count / total_count
            if uniqueness_ratio < 0.1:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'continuous'
    
    return column_types

def validate_dataset_config(df: pd.DataFrame, target_col: str, config: Dict[str, Any]) -> bool:
    """
    Validate dataset configuration and provide warnings.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        config: Configuration dictionary
        
    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n🔍 Dataset Validation:")
    
    valid = True
    
    # Check if target column exists
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found in dataset!")
        print(f"   Available columns: {list(df.columns)}")
        valid = False
    else:
        print(f"✅ Target column '{target_col}' found")
    
    # Check dataset size
    if len(df) < 100:
        print(f"⚠️ Small dataset: {len(df)} rows (recommend >1000 for synthetic data)")
    else:
        print(f"✅ Dataset size: {len(df)} rows")
    
    # Check for missing data
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct > 20:
        print(f"⚠️ High missing data: {missing_pct:.1f}% (recommend MICE imputation)")
    elif missing_pct > 0:
        print(f"🔍 Missing data: {missing_pct:.1f}% (manageable)")
    else:
        print(f"✅ No missing data")
    
    return valid

def test_dataset_configuration(config):
    """Test a specific dataset configuration"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {config['dataset_name']}")
    print(f"{'='*60}")
    
    # Configuration summary
    print(f"📊 Configuration Summary:")
    print(f"   Dataset: {config['dataset_name']}")
    print(f"   File: {config['data_file']}")
    print(f"   Target: {config['target_column']}")
    print(f"   Missing Data Strategy: {config['missing_strategy']}")
    
    try:
        # SECTION 1: Data Loading and Configuration
        print(f"\n🧪 SECTION 1: Data Loading and Configuration")
        print(f"📂 Loading dataset: {config['data_file']}")
        
        # Load the dataset
        data = pd.read_csv(config['data_file'])
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Original shape: {data.shape}")
        
        # Standardize column names
        print(f"\n🔄 Standardizing column names...")
        data_standardized, column_mapping = standardize_column_names(data)
        
        # Update target column name if it was changed
        original_target = config['target_column']
        target_column = config['target_column']
        if target_column in column_mapping:
            target_column = column_mapping[target_column]
            print(f"🎯 Target column updated: '{original_target}' → '{target_column}'")
        
        # Detect target column (in case user didn't specify or name changed)
        target_column = detect_target_column(data_standardized, target_column)
        
        # Analyze column types
        print(f"\n🔍 Analyzing column types...")
        column_types = analyze_column_types(data_standardized, config['categorical_columns'])
        
        print(f"\n📋 Column Type Analysis:")
        for col, col_type in column_types.items():
            print(f"   {col}: {col_type}")
        
        # Validate configuration
        config_dict = {
            'data_file': config['data_file'],
            'target_column': target_column,
            'categorical_columns': config['categorical_columns'],
            'missing_strategy': config['missing_strategy']
        }
        
        validation_passed = validate_dataset_config(data_standardized, target_column, config_dict)
        
        if not validation_passed:
            print(f"\n❌ Configuration validation failed. Please review the configuration.")
            return False
        else:
            print(f"\n✅ Configuration validation passed!")
        
        # Update data reference
        data = data_standardized
        
        print(f"\n📊 Final Dataset Summary:")
        print(f"   Shape: {data.shape}")
        print(f"   Target Column: {target_column}")
        print(f"   Missing Values: {data.isnull().sum().sum()}")
        print(f"   Categorical Columns: {[col for col, typ in column_types.items() if typ == 'categorical']}")
        print(f"   Continuous Columns: {[col for col, typ in column_types.items() if typ == 'continuous']}")
        print(f"   Binary Columns: {[col for col, typ in column_types.items() if typ == 'binary']}")
        
        # SECTION 2: Basic EDA
        print(f"\n🧪 SECTION 2: Exploratory Data Analysis")
        
        # Basic statistics
        print(f"\n📈 Basic Statistics:")
        print(f"   Data Types:")
        for dtype, count in data.dtypes.value_counts().items():
            print(f"     {dtype}: {count} columns")
        
        # Missing data analysis
        missing_data = data.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        if len(missing_cols) > 0:
            print(f"\n🔍 Missing Data Analysis:")
            for col, missing_count in missing_cols.items():
                pct = (missing_count / len(data)) * 100
                print(f"     {col}: {missing_count} ({pct:.1f}%)")
        else:
            print(f"\n✅ No missing data detected")
        
        # Target distribution
        print(f"\n🎯 Target Variable Analysis:")
        target_dist = data[target_column].value_counts()
        print(f"   Distribution: {dict(target_dist)}")
        print(f"   Balance: {target_dist.min()/target_dist.max():.2f}")
        
        # SECTION 3: Data Preprocessing Test
        print(f"\n🧪 SECTION 3: Data Preprocessing")
        
        # Simple preprocessing test
        print(f"🔧 Testing basic preprocessing...")
        
        # Separate features and target
        features = data.drop(columns=[target_column])
        target = data[target_column]
        
        print(f"   Features shape: {features.shape}")
        print(f"   Target shape: {target.shape}")
        
        # Test numeric/categorical separation
        numeric_features = features.select_dtypes(include=[np.number])
        categorical_features = features.select_dtypes(include=['object', 'category'])
        
        print(f"   Numeric features: {numeric_features.shape[1]}")
        print(f"   Categorical features: {categorical_features.shape[1]}")
        
        # Test for any obvious issues
        if numeric_features.shape[1] == 0:
            print(f"⚠️ Warning: No numeric features detected")
        
        print(f"\n✅ SECTIONS 1-3 COMPLETED SUCCESSFULLY!")
        
        # Store results
        results = {
            'dataset_name': config['dataset_name'],
            'original_shape': data.shape,
            'target_column': target_column,
            'column_mapping': column_mapping,
            'column_types': column_types,
            'missing_data': dict(missing_cols) if len(missing_cols) > 0 else {},
            'target_distribution': dict(target_dist),
            'numeric_features': numeric_features.shape[1],
            'categorical_features': categorical_features.shape[1],
            'validation_passed': validation_passed,
            'success': True
        }
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return {
            'dataset_name': config['dataset_name'],
            'success': False,
            'error': str(e)
        }

def main():
    """Main testing function"""
    
    # Dataset configurations
    dataset_configs = [
        {
            'dataset_name': 'Breast Cancer Wisconsin',
            'data_file': 'data/Breast_cancer_data.csv',
            'target_column': 'diagnosis',
            'categorical_columns': [],
            'missing_strategy': 'mice'
        },
        {
            'dataset_name': 'Pakistani Diabetes',
            'data_file': 'data/Pakistani_Diabetes_Dataset.csv',
            'target_column': 'Outcome',
            'categorical_columns': ['Gender', 'Rgn'],
            'missing_strategy': 'mice'
        },
        {
            'dataset_name': 'Alzheimer\'s Disease',
            'data_file': 'data/alzheimers_disease_data.csv',
            'target_column': 'Diagnosis',
            'categorical_columns': ['Gender', 'Ethnicity', 'EducationLevel'],
            'missing_strategy': 'mice'
        },
        {
            'dataset_name': 'Liver Disease',
            'data_file': 'data/liver_train.csv',
            'target_column': 'Result',
            'categorical_columns': ['Gender of the patient'],
            'missing_strategy': 'mice'
        }
    ]
    
    print(f"Clinical Synthetic Data Generation Framework - Dataset Testing")
    print(f"Testing {len(dataset_configs)} datasets with sections 1-3")
    
    all_results = []
    
    for config in dataset_configs:
        result = test_dataset_configuration(config)
        all_results.append(result)
    
    # Summary report
    print(f"\n{'='*60}")
    print(f"📋 TESTING SUMMARY REPORT")
    print(f"{'='*60}")
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"\n✅ SUCCESSFUL: {len(successful)}/{len(all_results)} datasets")
    for result in successful:
        print(f"   • {result['dataset_name']}")
        if 'target_column' in result:
            print(f"     Target: {result['target_column']}")
            print(f"     Shape: {result['original_shape']}")
            print(f"     Missing data: {len(result['missing_data'])} columns")
    
    if failed:
        print(f"\n❌ FAILED: {len(failed)} datasets")
        for result in failed:
            print(f"   • {result['dataset_name']}: {result['error']}")
    
    print(f"\n🎯 CONFIGURATION RECOMMENDATIONS:")
    for result in successful:
        if 'column_mapping' in result and result['column_mapping']:
            print(f"\n{result['dataset_name']}:")
            print(f"   Column mappings applied: {len(result['column_mapping'])}")
            if result['missing_data']:
                print(f"   Columns with missing data: {list(result['missing_data'].keys())}")
            print(f"   Recommended target: '{result['target_column']}'")
    
    return all_results

if __name__ == "__main__":
    results = main()