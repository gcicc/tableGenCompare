"""
Data validation utilities for ensuring dataset quality and consistency.

This module provides comprehensive data validation to catch issues early
and ensure datasets meet quality standards for synthetic data generation.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validator for synthetic data benchmarking datasets.
    
    Validates data quality, schema compliance, and suitability for
    synthetic data generation tasks.
    """
    
    def __init__(self):
        """Initialize data validator."""
        self.validation_rules = {
            "max_missing_percentage": 10.0,
            "min_unique_values": 5,
            "max_duplicate_percentage": 5.0,
            "min_class_balance_ratio": 0.05,
            "max_memory_usage_mb": 1000.0
        }
    
    def validate_dataset(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive dataset validation.
        
        Args:
            data: Dataset to validate
            config: Dataset configuration
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "quality_metrics": {},
            "recommendations": []
        }
        
        # Update validation rules from config if provided
        thresholds = config.get("quality_thresholds", {})
        self.validation_rules.update(thresholds)
        
        # Run validation checks
        validation_results = self._validate_basic_structure(data, validation_results)
        validation_results = self._validate_data_quality(data, validation_results)
        validation_results = self._validate_target_column(data, config, validation_results)
        validation_results = self._validate_feature_distribution(data, config, validation_results)
        validation_results = self._validate_schema_compliance(data, config, validation_results)
        validation_results = self._generate_recommendations(data, config, validation_results)
        
        # Overall validation status
        validation_results["is_valid"] = len(validation_results["issues"]) == 0
        
        if not validation_results["is_valid"]:
            logger.warning(f"Dataset validation failed with {len(validation_results['issues'])} issues")
        elif validation_results["warnings"]:
            logger.info(f"Dataset validation passed with {len(validation_results['warnings'])} warnings")
        else:
            logger.info("Dataset validation passed successfully")
        
        return validation_results
    
    def _validate_basic_structure(
        self, 
        data: pd.DataFrame, 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate basic dataset structure."""
        # Check if dataset is empty
        if data.empty:
            results["issues"].append("Dataset is empty")
            return results
        
        # Check for minimum size
        if len(data) < 100:
            results["warnings"].append(f"Dataset is very small ({len(data)} samples). Consider larger dataset for reliable results.")
        
        # Check for minimum features
        if len(data.columns) < 2:
            results["issues"].append("Dataset must have at least 2 columns (features + target)")
        
        # Check for all-null columns
        null_columns = data.columns[data.isnull().all()].tolist()
        if null_columns:
            results["issues"].append(f"Columns with all missing values: {null_columns}")
        
        # Basic statistics
        results["quality_metrics"]["shape"] = data.shape
        results["quality_metrics"]["memory_usage_mb"] = data.memory_usage(deep=True).sum() / 1024**2
        
        return results
    
    def _validate_data_quality(
        self, 
        data: pd.DataFrame, 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate data quality metrics."""
        # Missing values
        missing_count = data.isnull().sum().sum()
        missing_percentage = (missing_count / data.size) * 100
        
        results["quality_metrics"]["missing_values"] = {
            "count": missing_count,
            "percentage": missing_percentage
        }
        
        if missing_percentage > self.validation_rules["max_missing_percentage"]:
            results["issues"].append(f"Too many missing values: {missing_percentage:.2f}% (max: {self.validation_rules['max_missing_percentage']}%)")
        
        # Duplicate rows
        duplicate_count = data.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(data)) * 100
        
        results["quality_metrics"]["duplicates"] = {
            "count": duplicate_count,
            "percentage": duplicate_percentage
        }
        
        if duplicate_percentage > self.validation_rules["max_duplicate_percentage"]:
            results["warnings"].append(f"High duplicate rate: {duplicate_percentage:.2f}% (max recommended: {self.validation_rules['max_duplicate_percentage']}%)")
        
        # Memory usage
        memory_mb = results["quality_metrics"]["memory_usage_mb"]
        if memory_mb > self.validation_rules["max_memory_usage_mb"]:
            results["warnings"].append(f"Large dataset: {memory_mb:.2f}MB (may affect performance)")
        
        # Data type analysis
        results["quality_metrics"]["data_types"] = {
            "numeric_columns": len(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(data.select_dtypes(include=['object']).columns),
            "datetime_columns": len(data.select_dtypes(include=['datetime64']).columns)
        }
        
        return results
    
    def _validate_target_column(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any], 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate target column properties."""
        target_col = config["dataset"]["target_column"]
        
        # Check if target column exists
        if target_col not in data.columns:
            results["issues"].append(f"Target column '{target_col}' not found in dataset")
            return results
        
        target_data = data[target_col]
        
        # Basic target statistics
        unique_values = target_data.nunique()
        value_counts = target_data.value_counts()
        
        results["quality_metrics"]["target"] = {
            "column": target_col,
            "unique_values": unique_values,
            "value_counts": value_counts.to_dict(),
            "missing_values": target_data.isnull().sum()
        }
        
        # Check for missing values in target
        if target_data.isnull().any():
            results["issues"].append(f"Target column '{target_col}' contains missing values")
        
        # Validate based on task type
        task_type = config["dataset"].get("task_type", "unknown")
        
        if task_type == "binary_classification":
            if unique_values != 2:
                results["issues"].append(f"Binary classification task should have 2 unique target values, found {unique_values}")
            
            # Check class balance
            if len(value_counts) == 2:
                min_class_count = value_counts.min()
                max_class_count = value_counts.max()
                balance_ratio = min_class_count / max_class_count
                
                results["quality_metrics"]["target"]["class_balance_ratio"] = balance_ratio
                
                if balance_ratio < self.validation_rules["min_class_balance_ratio"]:
                    results["warnings"].append(f"Severe class imbalance: ratio {balance_ratio:.3f} (min recommended: {self.validation_rules['min_class_balance_ratio']})")
                elif balance_ratio < 0.3:
                    results["warnings"].append(f"Class imbalance detected: ratio {balance_ratio:.3f}")
        
        elif task_type == "multiclass_classification":
            if unique_values < 3:
                results["issues"].append(f"Multiclass classification task should have 3+ unique target values, found {unique_values}")
        
        elif task_type == "regression":
            if target_data.dtype not in [np.number]:
                results["issues"].append(f"Regression target should be numeric, found {target_data.dtype}")
        
        return results
    
    def _validate_feature_distribution(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any], 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate feature distributions and characteristics."""
        target_col = config["dataset"]["target_column"]
        feature_cols = [col for col in data.columns if col != target_col]
        
        feature_stats = {}
        
        for col in feature_cols:
            col_data = data[col]
            stats = {
                "dtype": str(col_data.dtype),
                "unique_values": col_data.nunique(),
                "missing_values": col_data.isnull().sum(),
                "missing_percentage": (col_data.isnull().sum() / len(col_data)) * 100
            }
            
            # Numeric feature analysis
            if col_data.dtype in [np.number]:
                stats.update({
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "zeros": (col_data == 0).sum(),
                    "infinite_values": np.isinf(col_data).sum()
                })
                
                # Check for constant features
                if col_data.nunique() == 1:
                    results["warnings"].append(f"Constant feature detected: {col}")
                
                # Check for very low variance
                if col_data.std() < 1e-6:
                    results["warnings"].append(f"Very low variance feature: {col}")
                
                # Check for infinite values
                if np.isinf(col_data).any():
                    results["issues"].append(f"Infinite values detected in feature: {col}")
            
            # Categorical feature analysis
            else:
                top_values = col_data.value_counts().head(5).to_dict()
                stats.update({
                    "top_values": top_values,
                    "most_frequent_percentage": (col_data.value_counts().iloc[0] / len(col_data)) * 100
                })
                
                # Check for high cardinality
                if col_data.nunique() > len(col_data) * 0.5:
                    results["warnings"].append(f"High cardinality categorical feature: {col} ({col_data.nunique()} unique values)")
            
            feature_stats[col] = stats
        
        results["quality_metrics"]["features"] = feature_stats
        
        return results
    
    def _validate_schema_compliance(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any], 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate data against expected schema."""
        expected_schema = config.get("expected_schema", {})
        
        if not expected_schema:
            return results
        
        schema_issues = []
        
        for col, expected in expected_schema.items():
            if col not in data.columns:
                schema_issues.append(f"Expected column '{col}' not found")
                continue
            
            col_data = data[col]
            
            # Check data type
            expected_type = expected.get("type")
            if expected_type:
                if expected_type == "float" and col_data.dtype not in ['float32', 'float64']:
                    schema_issues.append(f"Column '{col}' expected {expected_type}, got {col_data.dtype}")
                elif expected_type == "int" and col_data.dtype not in ['int32', 'int64']:
                    schema_issues.append(f"Column '{col}' expected {expected_type}, got {col_data.dtype}")
            
            # Check value ranges
            if "min_value" in expected:
                min_val = col_data.min()
                if min_val < expected["min_value"]:
                    schema_issues.append(f"Column '{col}' has values below expected minimum: {min_val} < {expected['min_value']}")
            
            if "max_value" in expected:
                max_val = col_data.max()
                if max_val > expected["max_value"]:
                    schema_issues.append(f"Column '{col}' has values above expected maximum: {max_val} > {expected['max_value']}")
        
        if schema_issues:
            results["warnings"].extend(schema_issues)
        
        return results
    
    def _generate_recommendations(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any], 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommendations for improving dataset quality."""
        recommendations = []
        
        # Sample size recommendations
        if len(data) < 1000:
            recommendations.append("Consider increasing sample size to 1000+ for more reliable synthetic data generation")
        
        # Feature recommendations
        numeric_features = len(data.select_dtypes(include=[np.number]).columns)
        if numeric_features < 3:
            recommendations.append("Consider adding more numeric features for better synthetic data quality")
        
        # Missing value recommendations
        missing_percentage = results["quality_metrics"]["missing_values"]["percentage"]
        if missing_percentage > 5:
            recommendations.append("Consider imputing or removing features with high missing value rates")
        
        # Class balance recommendations
        target_info = results["quality_metrics"].get("target", {})
        if "class_balance_ratio" in target_info and target_info["class_balance_ratio"] < 0.3:
            recommendations.append("Consider balancing classes using sampling techniques")
        
        results["recommendations"] = recommendations
        
        return results