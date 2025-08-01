"""
Dataset handler for loading and preprocessing datasets with configuration-driven approach.

This module provides a unified interface for loading different datasets
with consistent preprocessing and validation.
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from .data_validator import DataValidator

logger = logging.getLogger(__name__)


class DatasetHandler:
    """
    Unified handler for loading and preprocessing datasets with configuration-driven approach.
    
    This class provides consistent interfaces for different datasets while allowing
    dataset-specific customization through YAML configuration files.
    """
    
    def __init__(self, config_dir: str = "configs/datasets"):
        """
        Initialize dataset handler.
        
        Args:
            config_dir: Directory containing dataset configuration files
        """
        self.config_dir = Path(config_dir)
        self.validator = DataValidator()
        self._loaded_configs = {}
        
    def load_dataset(
        self, 
        dataset_name: str, 
        data_dir: str = "data",
        validate: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a dataset with its configuration.
        
        Args:
            dataset_name: Name of the dataset (config file name without .yaml)
            data_dir: Directory containing data files
            validate: Whether to perform data validation
            
        Returns:
            Tuple of (processed_dataframe, dataset_metadata)
        """
        # Load configuration
        config = self._load_config(dataset_name)
        
        # Construct file path
        data_path = Path(data_dir) / config["dataset"]["file_path"].split("/")[-1]
        
        if not data_path.exists():
            # Try original path from config
            data_path = Path(config["dataset"]["file_path"])
            
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        logger.info(f"Loading dataset: {config['dataset']['name']}")
        
        # Load data
        try:
            if data_path.suffix.lower() == '.csv':
                data = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
                
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
        
        logger.info(f"Loaded {len(data)} samples with {len(data.columns)} features")
        
        # Validate data if requested
        if validate:
            validation_results = self.validator.validate_dataset(data, config)
            if not validation_results["is_valid"]:
                logger.warning(f"Data validation issues: {validation_results['issues']}")
        
        # Apply preprocessing
        processed_data = self._preprocess_data(data, config)
        
        # Create metadata
        metadata = self._create_metadata(processed_data, config)
        
        return processed_data, metadata
    
    def preprocess_data(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply preprocessing steps based on configuration.
        
        Args:
            data: Raw dataset
            config: Dataset configuration
            
        Returns:
            Preprocessed dataset
        """
        return self._preprocess_data(data, config)
    
    def split_data(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        return_indices: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train/test sets based on configuration.
        
        Args:
            data: Dataset to split
            config: Dataset configuration
            return_indices: Whether to return split indices
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) or with indices if requested
        """
        target_col = config["dataset"]["target_column"]
        eval_config = config.get("evaluation", {})
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Get split parameters
        test_size = eval_config.get("test_size", 0.3)
        random_state = eval_config.get("random_state", 42)
        stratify = y if eval_config.get("stratify", True) and y.nunique() > 1 else None
        
        split_result = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        return split_result
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing dataset information
        """
        config = self._load_config(dataset_name)
        
        return {
            "name": config["dataset"]["name"],
            "description": config["dataset"]["description"],
            "task_type": config["dataset"]["task_type"],
            "domain": config["dataset"]["domain"],
            "features": config["features"],
            "metadata": config.get("metadata", {}),
            "preprocessing": config.get("preprocessing", {}),
            "evaluation": config.get("evaluation", {})
        }
    
    def list_available_datasets(self) -> List[str]:
        """
        List all available dataset configurations.
        
        Returns:
            List of dataset names
        """
        if not self.config_dir.exists():
            return []
        
        config_files = list(self.config_dir.glob("*.yaml"))
        return [f.stem for f in config_files]
    
    def validate_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Validate a dataset configuration file.
        
        Args:
            dataset_name: Name of the dataset configuration to validate
            
        Returns:
            Validation results
        """
        try:
            config = self._load_config(dataset_name)
            
            # Check required fields
            required_fields = [
                "dataset.name",
                "dataset.file_path", 
                "dataset.target_column",
                "dataset.task_type"
            ]
            
            issues = []
            for field in required_fields:
                if not self._get_nested_value(config, field):
                    issues.append(f"Missing required field: {field}")
            
            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "config": config
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "issues": [f"Configuration error: {e}"],
                "config": None
            }
    
    def _load_config(self, dataset_name: str) -> Dict[str, Any]:
        """Load dataset configuration from YAML file."""
        if dataset_name in self._loaded_configs:
            return self._loaded_configs[dataset_name]
        
        config_path = self.config_dir / f"{dataset_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self._loaded_configs[dataset_name] = config
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config for {dataset_name}: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply preprocessing steps based on configuration."""
        processed_data = data.copy()
        preprocessing_config = config.get("preprocessing", {})
        
        # Handle missing values
        missing_strategy = preprocessing_config.get("missing_value_strategy", "median")
        processed_data = self._handle_missing_values(processed_data, missing_strategy)
        
        # Handle outliers
        if preprocessing_config.get("outlier_detection", False):
            outlier_method = preprocessing_config.get("outlier_method", "iqr")
            processed_data = self._handle_outliers(processed_data, outlier_method, config)
        
        # Data type optimization
        processed_data = self._optimize_data_types(processed_data)
        
        # Validate processed data
        processed_data = self._validate_processed_data(processed_data, config)
        
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values based on strategy."""
        if data.isnull().sum().sum() == 0:
            return data
        
        processed_data = data.copy()
        
        for col in data.columns:
            if data[col].isnull().any():
                if data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    if strategy == "median":
                        processed_data[col].fillna(data[col].median(), inplace=True)
                    elif strategy == "mean":
                        processed_data[col].fillna(data[col].mean(), inplace=True)
                else:
                    if strategy in ["mode", "most_frequent"]:
                        processed_data[col].fillna(data[col].mode()[0], inplace=True)
        
        return processed_data
    
    def _handle_outliers(self, data: pd.DataFrame, method: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Handle outliers based on method."""
        # For now, just log outlier detection - implement actual handling if needed
        target_col = config["dataset"]["target_column"]
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        outlier_counts = {}
        for col in numeric_cols:
            if method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                outlier_counts[col] = outliers.sum()
        
        total_outliers = sum(outlier_counts.values())
        if total_outliers > 0:
            logger.info(f"Detected {total_outliers} outliers across {len(outlier_counts)} features")
        
        return data  # Return original data for now
    
    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        optimized_data = data.copy()
        
        for col in data.columns:
            if data[col].dtype == 'int64':
                if data[col].min() >= -2147483648 and data[col].max() <= 2147483647:
                    optimized_data[col] = data[col].astype('int32')
            elif data[col].dtype == 'float64':
                optimized_data[col] = pd.to_numeric(data[col], downcast='float')
        
        return optimized_data
    
    def _validate_processed_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Validate processed data against expected schema."""
        expected_schema = config.get("expected_schema", {})
        
        for col, schema in expected_schema.items():
            if col in data.columns:
                # Check data type
                if schema.get("type") == "float" and data[col].dtype not in ['float32', 'float64']:
                    logger.warning(f"Column {col} expected float, got {data[col].dtype}")
                elif schema.get("type") == "int" and data[col].dtype not in ['int32', 'int64']:
                    logger.warning(f"Column {col} expected int, got {data[col].dtype}")
                
                # Check value ranges
                if "min_value" in schema and data[col].min() < schema["min_value"]:
                    logger.warning(f"Column {col} has values below expected minimum {schema['min_value']}")
                if "max_value" in schema and data[col].max() > schema["max_value"]:
                    logger.warning(f"Column {col} has values above expected maximum {schema['max_value']}")
        
        return data
    
    def _create_metadata(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive metadata for the dataset."""
        target_col = config["dataset"]["target_column"]
        
        metadata = {
            "dataset_info": config["dataset"],
            "shape": data.shape,
            "features": {
                "total": len(data.columns),
                "numeric": len(data.select_dtypes(include=[np.number]).columns),
                "categorical": len(data.select_dtypes(include=['object']).columns)
            },
            "target_info": {
                "column": target_col,
                "unique_values": data[target_col].nunique(),
                "value_counts": data[target_col].value_counts().to_dict()
            },
            "data_quality": {
                "missing_values": data.isnull().sum().sum(),
                "duplicate_rows": data.duplicated().sum(),
                "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2
            },
            "preprocessing_applied": config.get("preprocessing", {}),
            "source_metadata": config.get("metadata", {})
        }
        
        return metadata
    
    def _get_nested_value(self, dictionary: Dict[str, Any], key: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key.split('.')
        value = dictionary
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value