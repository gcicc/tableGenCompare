"""
Clinical Data Preprocessing Pipeline

Specialized preprocessing for clinical trial data including:
- Missing value imputation using MICE
- Categorical encoding
- Data validation
- Column type inference
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import logging

class ClinicalDataPreprocessor:
    """Preprocessing pipeline for clinical trial data."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.imputer = None
        self.label_encoders = {}
        self.column_types = {}
        self.target_column = None
        self.is_fitted = False
        
    def fit_transform(self, data: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Fit preprocessor and transform data.
        
        Args:
            data: Input clinical data
            target_col: Name of target column (if any)
            
        Returns:
            Preprocessed data ready for synthetic data generation
        """
        # For liver dataset: target_col = 'Result'
        self.target_column = target_col
        data_clean = self._clean_column_names(data.copy())
        
        # Remove rows with missing target values
        if target_col and target_col in data_clean.columns:
            data_clean = data_clean.dropna(subset=[target_col])
            logging.info(f"Removed rows with missing target values. Shape: {data_clean.shape}")
        
        # Identify column types
        self.column_types = self._identify_column_types(data_clean, target_col)
        
        # Encode categorical variables
        data_encoded = self._encode_categorical_variables(data_clean)
        
        # Handle missing values with MICE
        data_imputed = self._impute_missing_values(data_encoded)
        
        self.is_fitted = True
        return data_imputed
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        data_clean = self._clean_column_names(data.copy())
        data_encoded = self._encode_categorical_variables(data_clean, fit=False)
        data_imputed = self._impute_missing_values(data_encoded, fit=False)
        
        return data_imputed
    
    def _clean_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names."""
        data.columns = (data.columns
                       .str.strip()
                       .str.replace(r'\xa0', '', regex=True)
                       .str.replace(r'\s+', '_', regex=True)
                       .str.lower())
        return data
    
    def _identify_column_types(self, data: pd.DataFrame, target_col: str = None) -> Dict[str, str]:
        """Identify continuous and categorical columns."""
        column_types = {}
        
        for col in data.columns:
            if col == target_col:
                column_types[col] = 'target'
            elif data[col].dtype in ['int64', 'float64']:
                # Consider as categorical if few unique values
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio < 0.05 and data[col].nunique() < 20:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'continuous'
            else:
                column_types[col] = 'categorical'
        
        logging.info(f"Column types identified: {column_types}")
        return column_types
    
    def _encode_categorical_variables(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        data_encoded = data.copy()
        
        categorical_cols = [col for col, col_type in self.column_types.items() 
                          if col_type == 'categorical' and col != self.target_column]
        
        if categorical_cols:
            # One-hot encoding for non-target categorical columns
            data_encoded = pd.get_dummies(data_encoded, columns=categorical_cols, drop_first=True)
        
        # Label encode target column if categorical
        if (self.target_column and self.target_column in data_encoded.columns 
            and data_encoded[self.target_column].dtype == 'object'):
            
            if fit:
                le = LabelEncoder()
                data_encoded[self.target_column] = le.fit_transform(
                    data_encoded[self.target_column].astype(str)
                )
                self.label_encoders[self.target_column] = le
            else:
                le = self.label_encoders[self.target_column]
                data_encoded[self.target_column] = le.transform(
                    data_encoded[self.target_column].astype(str)
                )
        
        return data_encoded
    
    def _impute_missing_values(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Impute missing values using MICE."""
        if data.isnull().sum().sum() == 0:
            return data
        
        # Separate target column for imputation
        target_data = None
        if self.target_column and self.target_column in data.columns:
            target_data = data[self.target_column].copy()
            data_for_imputation = data.drop(columns=[self.target_column])
        else:
            data_for_imputation = data.copy()
        
        if fit:
            self.imputer = IterativeImputer(max_iter=10, random_state=self.random_state)
            imputed_array = self.imputer.fit_transform(data_for_imputation)
        else:
            imputed_array = self.imputer.transform(data_for_imputation)
        
        data_imputed = pd.DataFrame(imputed_array, columns=data_for_imputation.columns)
        
        # Reattach target column
        if target_data is not None:
            data_imputed[self.target_column] = target_data.values
        
        logging.info(f"Missing value imputation completed. Shape: {data_imputed.shape}")
        return data_imputed
    
    def get_discrete_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify discrete columns for synthetic data models."""
        discrete_cols = []
        for col in data.columns:
            if (col in self.column_types and self.column_types[col] == 'categorical') or \
               col.endswith('_1') or col.endswith('_True'):  # One-hot encoded columns
                discrete_cols.append(col)
        return discrete_cols
    
    def generate_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary for reporting."""
        summary = {
            'shape': data.shape,
            'missing_values': data.isnull().sum().sum(),
            'column_types': self.column_types,
            'discrete_columns': self.get_discrete_columns(data),
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        if self.target_column and self.target_column in data.columns:
            summary['target_distribution'] = data[self.target_column].value_counts().to_dict()
        
        return summary