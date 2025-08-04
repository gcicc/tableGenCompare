"""
Utility functions for the API service.
"""

import os
import re
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that a DataFrame is suitable for synthetic data generation."""
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if len(df) < 10:
        raise ValueError("DataFrame must have at least 10 rows")
    
    if len(df.columns) < 2:
        raise ValueError("DataFrame must have at least 2 columns")
    
    # Check for excessive missing values
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_ratio > 0.5:
        raise ValueError(f"Too many missing values: {missing_ratio:.1%}")
    
    # Check for columns with all same values
    constant_columns = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_columns.append(col)
    
    if constant_columns:
        raise ValueError(f"Columns with constant values: {constant_columns}")
    
    # Check data types
    unsupported_types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype.startswith('datetime') or dtype == 'timedelta64[ns]':
            unsupported_types.append(col)
    
    if unsupported_types:
        raise ValueError(f"Unsupported data types in columns: {unsupported_types}")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace unsafe characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Ensure it doesn't start with dot or dash
    filename = filename.lstrip('.-')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename or "unnamed_file"

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""

def validate_file_upload(file_path: str, max_size_mb: int = 100) -> Dict[str, Any]:
    """Validate uploaded file."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="File not found")
    
    # Check file size
    file_size = os.path.getsize(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB > {max_size_mb}MB"
        )
    
    # Check file extension
    valid_extensions = ['.csv', '.json', '.parquet']
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file_ext}. Allowed: {valid_extensions}"
        )
    
    return {
        "file_size": file_size,
        "file_extension": file_ext,
        "file_hash": calculate_file_hash(file_path)
    }

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def estimate_training_time(
    data_shape: tuple,
    model_name: str,
    epochs: int = None
) -> Dict[str, Union[float, str]]:
    """Estimate training time based on data size and model type."""
    rows, cols = data_shape
    
    # Base time estimates (in seconds per epoch)
    base_times = {
        'ganeraid': 0.1,
        'ctgan': 2.0,
        'tvae': 1.5,
        'copulagan': 1.0
    }
    
    base_time = base_times.get(model_name.lower(), 1.0)
    
    # Scale with data size
    size_factor = (rows / 1000) * (cols / 10)
    size_factor = max(0.1, min(size_factor, 10))  # Clamp between 0.1x and 10x
    
    time_per_epoch = base_time * size_factor
    
    # Default epochs if not provided
    if epochs is None:
        default_epochs = {
            'ganeraid': 100,
            'ctgan': 300,
            'tvae': 300,
            'copulagan': 200
        }
        epochs = default_epochs.get(model_name.lower(), 100)
    
    total_time = time_per_epoch * epochs
    
    return {
        "estimated_seconds": total_time,
        "estimated_minutes": total_time / 60,
        "formatted": format_duration(total_time),
        "epochs": epochs,
        "time_per_epoch": time_per_epoch
    }

def validate_hyperparameters(
    hyperparams: Dict[str, Any],
    model_name: str
) -> Dict[str, Any]:
    """Validate and sanitize hyperparameters for a specific model."""
    
    # Common parameter ranges
    param_ranges = {
        'ganeraid': {
            'epochs': (1, 1000),
            'lr_g': (1e-5, 1e-1),
            'lr_d': (1e-5, 1e-1),
            'hidden_feature_space': (50, 1000),
            'batch_size': (8, 512)
        },
        'ctgan': {
            'epochs': (1, 1000),
            'batch_size': (8, 1000),
            'lr': (1e-5, 1e-1),
            'embedding_dim': (64, 512),
            'generator_dim': (128, 1024),
            'discriminator_dim': (128, 1024)
        },
        'tvae': {
            'epochs': (1, 1000),
            'batch_size': (8, 1000),
            'lr': (1e-5, 1e-1),
            'embedding_dim': (64, 512),
            'compress_dims': (64, 512),
            'decompress_dims': (64, 512)
        }
    }
    
    model_ranges = param_ranges.get(model_name.lower(), {})
    validated = {}
    
    for param, value in hyperparams.items():
        if param in model_ranges:
            min_val, max_val = model_ranges[param]
            
            if isinstance(value, (int, float)):
                validated[param] = max(min_val, min(value, max_val))
                if validated[param] != value:
                    logger.warning(f"Clamped {param} from {value} to {validated[param]}")
            else:
                logger.warning(f"Invalid type for {param}: {type(value)}")
        else:
            # Pass through unknown parameters (with warning)
            validated[param] = value
            logger.warning(f"Unknown hyperparameter for {model_name}: {param}")
    
    return validated

def create_error_response(
    error_type: str,
    detail: str,
    status_code: int = 500,
    request_id: Optional[str] = None
) -> HTTPException:
    """Create standardized error response."""
    
    error_content = {
        "error": error_type,
        "detail": detail,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    if request_id:
        error_content["request_id"] = request_id
    
    return HTTPException(status_code=status_code, detail=error_content)

def validate_csv_file(file_path: str) -> pd.DataFrame:
    """Validate and load CSV file with error handling."""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("Unable to read CSV file with any supported encoding")
        
        # Validate the dataframe
        validate_dataframe(df)
        
        return df
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading CSV file: {str(e)}"
        )

def generate_api_documentation() -> Dict[str, Any]:
    """Generate API documentation structure."""
    return {
        "title": "Synthetic Data Generation API",
        "version": "1.0.0",
        "description": "Production-ready API for training generative models and creating synthetic tabular data",
        "endpoints": {
            "/health": {
                "method": "GET",
                "description": "Health check endpoint",
                "authentication": False
            },
            "/models": {
                "method": "GET",
                "description": "List available models",
                "authentication": True
            },
            "/train": {
                "method": "POST",
                "description": "Train a synthetic data generation model",
                "authentication": True,
                "rate_limit": "10 requests per hour"
            },
            "/generate": {
                "method": "POST",
                "description": "Generate synthetic data",
                "authentication": True,
                "rate_limit": "50 requests per hour"
            },
            "/evaluate": {
                "method": "POST",
                "description": "Evaluate synthetic data quality",
                "authentication": True,
                "rate_limit": "20 requests per hour"
            },
            "/jobs/{job_id}": {
                "method": "GET",
                "description": "Get job status",
                "authentication": True
            },
            "/jobs": {
                "method": "GET",
                "description": "List jobs",
                "authentication": True
            },
            "/download/{job_id}": {
                "method": "GET",
                "description": "Download job results",
                "authentication": True
            }
        },
        "authentication": {
            "type": "Bearer Token",
            "description": "Use API key or JWT token in Authorization header"
        },
        "rate_limits": {
            "default": "100 requests per hour",
            "training": "10 requests per hour",
            "generation": "50 requests per hour"
        }
    }

def cleanup_temp_files(directory: str, max_age_hours: int = 24):
    """Clean up temporary files older than specified age."""
    import time
    
    if not os.path.exists(directory):
        return
    
    cutoff_time = time.time() - (max_age_hours * 3600)
    cleaned_count = 0
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                file_age = os.path.getmtime(file_path)
                
                if file_age < cutoff_time:
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Error removing {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary files from {directory}")
            
    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {e}")

def validate_json_data(data: Any) -> pd.DataFrame:
    """Validate and convert JSON data to DataFrame."""
    try:
        if isinstance(data, str):
            import json
            data = json.loads(data)
        
        if isinstance(data, dict):
            # Single record
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # Multiple records
            df = pd.DataFrame(data)
        else:
            raise ValueError("JSON data must be an object or array of objects")
        
        validate_dataframe(df)
        return df
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing JSON data: {str(e)}"
        )