"""
Dataset management system for multi-dataset benchmarking.

This module handles loading, validation, and management of multiple datasets
for comprehensive benchmarking across different data characteristics.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """Dataset type classification."""
    CLINICAL = "clinical"
    FINANCIAL = "financial"
    MARKETING = "marketing"
    GENERAL = "general"
    SYNTHETIC = "synthetic"


class DatasetSize(Enum):
    """Dataset size classification."""
    SMALL = "small"      # < 1K rows
    MEDIUM = "medium"    # 1K - 10K rows  
    LARGE = "large"      # 10K - 100K rows
    XLARGE = "xlarge"    # > 100K rows


@dataclass
class DatasetConfig:
    """Configuration for a benchmark dataset."""
    
    name: str
    file_path: Optional[str] = None
    data: Optional[pd.DataFrame] = None
    target_column: Optional[str] = None
    
    # Dataset metadata
    dataset_type: DatasetType = DatasetType.GENERAL
    description: str = ""
    source: str = ""
    
    # Data characteristics
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    n_categorical: Optional[int] = None
    n_numerical: Optional[int] = None
    n_binary: Optional[int] = None
    
    # Preprocessing options
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation options
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.data is not None:
            self._analyze_data_characteristics()
        elif self.file_path is not None:
            self._load_and_analyze_data()
    
    def _load_and_analyze_data(self):
        """Load data from file and analyze characteristics."""
        try:
            file_path = Path(self.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
            
            # Load based on file extension
            if file_path.suffix.lower() == '.csv':
                self.data = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                self.data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            self._analyze_data_characteristics()
            logger.info(f"Loaded dataset '{self.name}' from {self.file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset '{self.name}': {e}")
            raise
    
    def _analyze_data_characteristics(self):
        """Analyze and store data characteristics."""
        if self.data is None:
            return
        
        self.n_samples, self.n_features = self.data.shape
        
        # Analyze column types
        categorical_cols = []
        numerical_cols = []
        binary_cols = []
        
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                categorical_cols.append(col)
            elif self.data[col].dtype in ['int64', 'int32'] and self.data[col].nunique() == 2:
                binary_cols.append(col)
            elif self.data[col].dtype in ['int64', 'int32'] and self.data[col].nunique() <= 10:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        self.n_categorical = len(categorical_cols)
        self.n_numerical = len(numerical_cols)
        self.n_binary = len(binary_cols)
        
        # Store column information
        self.preprocessing_config.update({
            'categorical_columns': categorical_cols,
            'numerical_columns': numerical_cols,
            'binary_columns': binary_cols
        })
    
    @property
    def dataset_size(self) -> DatasetSize:
        """Get dataset size classification."""
        if self.n_samples is None:
            return DatasetSize.SMALL
        
        if self.n_samples < 1000:
            return DatasetSize.SMALL
        elif self.n_samples < 10000:
            return DatasetSize.MEDIUM
        elif self.n_samples < 100000:
            return DatasetSize.LARGE
        else:
            return DatasetSize.XLARGE
    
    @property
    def complexity_score(self) -> float:
        """Calculate dataset complexity score (0-1)."""
        if self.n_features is None or self.n_samples is None:
            return 0.0
        
        # Factors contributing to complexity
        feature_complexity = min(self.n_features / 20, 1.0)  # Normalize to 20 features
        categorical_complexity = (self.n_categorical or 0) / max(self.n_features, 1)
        size_complexity = min(self.n_samples / 10000, 1.0)  # Normalize to 10K samples
        
        return (feature_complexity + categorical_complexity + size_complexity) / 3
    
    def get_summary(self) -> Dict[str, Any]:
        """Get dataset summary information."""
        return {
            'name': self.name,
            'type': self.dataset_type.value,
            'size_category': self.dataset_size.value,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_categorical': self.n_categorical,
            'n_numerical': self.n_numerical,
            'n_binary': self.n_binary,
            'complexity_score': self.complexity_score,
            'target_column': self.target_column,
            'description': self.description
        }


class DatasetManager:
    """
    Manager for handling multiple datasets in benchmarking scenarios.
    
    Provides functionality to load, validate, and organize datasets for
    comprehensive multi-dataset benchmarking.
    """
    
    def __init__(self):
        """Initialize dataset manager."""
        self.datasets: Dict[str, DatasetConfig] = {}
        self.dataset_groups: Dict[str, List[str]] = {}
    
    def add_dataset(self, dataset_config: DatasetConfig) -> None:
        """
        Add a dataset to the manager.
        
        Args:
            dataset_config: Dataset configuration object
        """
        if dataset_config.name in self.datasets:
            logger.warning(f"Dataset '{dataset_config.name}' already exists. Replacing.")
        
        self.datasets[dataset_config.name] = dataset_config
        logger.info(f"Added dataset '{dataset_config.name}' to manager")
    
    def add_dataset_from_file(
        self, 
        name: str, 
        file_path: str, 
        target_column: Optional[str] = None,
        dataset_type: DatasetType = DatasetType.GENERAL,
        description: str = "",
        **kwargs
    ) -> DatasetConfig:
        """
        Add a dataset from file.
        
        Args:
            name: Dataset name
            file_path: Path to dataset file
            target_column: Target column name
            dataset_type: Type of dataset
            description: Dataset description
            **kwargs: Additional configuration options
            
        Returns:
            Created dataset configuration
        """
        config = DatasetConfig(
            name=name,
            file_path=file_path,
            target_column=target_column,
            dataset_type=dataset_type,
            description=description,
            **kwargs
        )
        
        self.add_dataset(config)
        return config
    
    def add_dataset_from_dataframe(
        self,
        name: str,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        dataset_type: DatasetType = DatasetType.GENERAL,
        description: str = "",
        **kwargs
    ) -> DatasetConfig:
        """
        Add a dataset from DataFrame.
        
        Args:
            name: Dataset name
            data: Dataset DataFrame
            target_column: Target column name
            dataset_type: Type of dataset
            description: Dataset description
            **kwargs: Additional configuration options
            
        Returns:
            Created dataset configuration
        """
        config = DatasetConfig(
            name=name,
            data=data.copy(),
            target_column=target_column,
            dataset_type=dataset_type,
            description=description,
            **kwargs
        )
        
        self.add_dataset(config)
        return config
    
    def create_dataset_group(self, group_name: str, dataset_names: List[str]) -> None:
        """
        Create a group of datasets for batch operations.
        
        Args:
            group_name: Name of the group
            dataset_names: List of dataset names to include
        """
        # Validate all datasets exist
        missing_datasets = [name for name in dataset_names if name not in self.datasets]
        if missing_datasets:
            raise ValueError(f"Datasets not found: {missing_datasets}")
        
        self.dataset_groups[group_name] = dataset_names
        logger.info(f"Created dataset group '{group_name}' with {len(dataset_names)} datasets")
    
    def get_dataset(self, name: str) -> DatasetConfig:
        """
        Get a dataset by name.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset configuration
        """
        if name not in self.datasets:
            raise KeyError(f"Dataset '{name}' not found")
        
        return self.datasets[name]
    
    def get_datasets_by_type(self, dataset_type: DatasetType) -> List[DatasetConfig]:
        """
        Get all datasets of a specific type.
        
        Args:
            dataset_type: Type of datasets to retrieve
            
        Returns:
            List of matching dataset configurations
        """
        return [
            config for config in self.datasets.values()
            if config.dataset_type == dataset_type
        ]
    
    def get_datasets_by_size(self, size: DatasetSize) -> List[DatasetConfig]:
        """
        Get all datasets of a specific size category.
        
        Args:
            size: Size category to filter by
            
        Returns:
            List of matching dataset configurations
        """
        return [
            config for config in self.datasets.values()
            if config.dataset_size == size
        ]
    
    def get_dataset_group(self, group_name: str) -> List[DatasetConfig]:
        """
        Get all datasets in a group.
        
        Args:
            group_name: Name of the group
            
        Returns:
            List of dataset configurations in the group
        """
        if group_name not in self.dataset_groups:
            raise KeyError(f"Dataset group '{group_name}' not found")
        
        return [self.datasets[name] for name in self.dataset_groups[group_name]]
    
    def list_datasets(self) -> List[str]:
        """Get list of all dataset names."""
        return list(self.datasets.keys())
    
    def list_groups(self) -> List[str]:
        """Get list of all group names."""
        return list(self.dataset_groups.keys())
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all managed datasets.
        
        Returns:
            Summary report with dataset statistics
        """
        if not self.datasets:
            return {'total_datasets': 0, 'datasets': []}
        
        datasets_summary = []
        total_samples = 0
        total_features = 0
        type_counts = {}
        size_counts = {}
        
        for config in self.datasets.values():
            summary = config.get_summary()
            datasets_summary.append(summary)
            
            if config.n_samples:
                total_samples += config.n_samples
            if config.n_features:
                total_features += config.n_features
            
            # Count by type
            dataset_type = config.dataset_type.value
            type_counts[dataset_type] = type_counts.get(dataset_type, 0) + 1
            
            # Count by size
            size_category = config.dataset_size.value
            size_counts[size_category] = size_counts.get(size_category, 0) + 1
        
        return {
            'total_datasets': len(self.datasets),
            'total_samples': total_samples,
            'average_features': total_features / len(self.datasets) if self.datasets else 0,
            'datasets_by_type': type_counts,
            'datasets_by_size': size_counts,
            'groups': {name: len(datasets) for name, datasets in self.dataset_groups.items()},
            'datasets': datasets_summary
        }
    
    def validate_datasets(self) -> Dict[str, List[str]]:
        """
        Validate all datasets and return validation results.
        
        Returns:
            Dictionary with validation results (valid/invalid dataset names)
        """
        valid_datasets = []
        invalid_datasets = []
        validation_errors = {}
        
        for name, config in self.datasets.items():
            try:
                # Check if data is available
                if config.data is None:
                    raise ValueError("No data available")
                
                # Check for empty dataset
                if config.data.empty:
                    raise ValueError("Dataset is empty")
                
                # Check for missing target column if specified
                if config.target_column and config.target_column not in config.data.columns:
                    raise ValueError(f"Target column '{config.target_column}' not found")
                
                # Check for excessive missing values
                missing_percentage = config.data.isnull().sum().sum() / (config.data.shape[0] * config.data.shape[1])
                if missing_percentage > 0.5:
                    raise ValueError(f"Excessive missing values: {missing_percentage:.1%}")
                
                valid_datasets.append(name)
                
            except Exception as e:
                invalid_datasets.append(name)
                validation_errors[name] = str(e)
                logger.warning(f"Dataset '{name}' validation failed: {e}")
        
        return {
            'valid': valid_datasets,
            'invalid': invalid_datasets,
            'errors': validation_errors
        }
    
    def save_configuration(self, file_path: str) -> None:
        """
        Save dataset manager configuration to file.
        
        Args:
            file_path: Path to save configuration
        """
        config_data = {
            'datasets': {},
            'groups': self.dataset_groups
        }
        
        # Save dataset configurations (excluding actual data)
        for name, dataset_config in self.datasets.items():
            config_data['datasets'][name] = {
                'name': dataset_config.name,
                'file_path': dataset_config.file_path,
                'target_column': dataset_config.target_column,
                'dataset_type': dataset_config.dataset_type.value,
                'description': dataset_config.description,
                'source': dataset_config.source,
                'preprocessing_config': dataset_config.preprocessing_config,
                'evaluation_config': dataset_config.evaluation_config
            }
        
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved dataset manager configuration to {file_path}")
    
    def load_configuration(self, file_path: str) -> None:
        """
        Load dataset manager configuration from file.
        
        Args:
            file_path: Path to configuration file
        """
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        # Clear existing configuration
        self.datasets.clear()
        self.dataset_groups.clear()
        
        # Load dataset configurations
        for name, dataset_info in config_data.get('datasets', {}).items():
            try:
                config = DatasetConfig(
                    name=dataset_info['name'],
                    file_path=dataset_info.get('file_path'),
                    target_column=dataset_info.get('target_column'),
                    dataset_type=DatasetType(dataset_info.get('dataset_type', 'general')),
                    description=dataset_info.get('description', ''),
                    source=dataset_info.get('source', ''),
                    preprocessing_config=dataset_info.get('preprocessing_config', {}),
                    evaluation_config=dataset_info.get('evaluation_config', {})
                )
                self.add_dataset(config)
            except Exception as e:
                logger.error(f"Failed to load dataset '{name}': {e}")
        
        # Load groups
        self.dataset_groups = config_data.get('groups', {})
        
        logger.info(f"Loaded dataset manager configuration from {file_path}")