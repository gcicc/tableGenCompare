"""
Path Utilities for Results Organization

This module provides utilities for managing file paths and results directories
in a standardized way across all notebooks.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union
import pandas as pd

from src.config import SESSION_TIMESTAMP


def extract_dataset_identifier(data_file_path):
    """
    Extract dataset identifier from file path or filename.

    Parameters:
    -----------
    data_file_path : str
        Path to the data file

    Returns:
    --------
    str : Dataset identifier (lowercase, hyphen-separated)

    Examples:
    ---------
    >>> extract_dataset_identifier('/path/to/Alzheimers_Disease_Data.csv')
    'alzheimers-disease-data'
    """
    if isinstance(data_file_path, str):
        filename = os.path.basename(data_file_path)
        dataset_id = os.path.splitext(filename)[0].lower()
        dataset_id = dataset_id.replace('_', '-').replace(' ', '-')
        return dataset_id
    return "unknown-dataset"


def get_results_path(dataset_identifier, section_number, model_name=None):
    """
    Generate standardized results path.

    Parameters:
    -----------
    dataset_identifier : str
        Dataset identifier (e.g., 'alzheimers-disease-data')
    section_number : int
        Section number (1-5)
    model_name : str, optional
        Model name for model-specific subdirectory

    Returns:
    --------
    str : Standardized path in format: results/{dataset_id}/{YYYY-MM-DD}/Section-{N}[/model_name]

    Examples:
    ---------
    >>> get_results_path('alzheimers-disease-data', 2)
    'results/alzheimers-disease-data/2024-12-05/Section-2'
    >>> get_results_path('alzheimers-disease-data', 3, 'ctgan')
    'results/alzheimers-disease-data/2024-12-05/Section-3/ctgan'
    """
    base_path = f"results/{dataset_identifier}/{SESSION_TIMESTAMP}/Section-{section_number}"
    if model_name:
        return f"{base_path}/{model_name}"
    return base_path


def ensure_results_dir(path: str) -> str:
    """
    Ensure a results directory exists, creating it if necessary.

    Parameters:
    -----------
    path : str
        Directory path to ensure exists

    Returns:
    --------
    str : The same path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_model_params(
    params: Dict[str, Any],
    dataset_identifier: str,
    section_number: int,
    model_name: str,
    filename: str = "params.json"
) -> str:
    """
    Save model parameters to a JSON file in the standard results location.

    Parameters:
    -----------
    params : dict
        Model parameters/hyperparameters to save
    dataset_identifier : str
        Dataset identifier
    section_number : int
        Section number (1-5)
    model_name : str
        Model name (e.g., 'ctgan', 'tvae')
    filename : str
        Output filename (default: 'params.json')

    Returns:
    --------
    str : Path to the saved file
    """
    results_dir = get_results_path(dataset_identifier, section_number, model_name)
    ensure_results_dir(results_dir)

    filepath = os.path.join(results_dir, filename)

    # Add metadata
    output = {
        "saved_at": datetime.now().isoformat(),
        "session_timestamp": SESSION_TIMESTAMP,
        "model_name": model_name,
        "section": section_number,
        "dataset": dataset_identifier,
        "parameters": params
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"[RESULTS] Saved parameters to: {filepath}")
    return filepath


def save_evaluation_summary(
    summary: Dict[str, Any],
    dataset_identifier: str,
    section_number: int,
    model_name: Optional[str] = None,
    filename: str = "evaluation_summary.json"
) -> str:
    """
    Save evaluation summary to a JSON file in the standard results location.

    Parameters:
    -----------
    summary : dict
        Evaluation results/metrics to save
    dataset_identifier : str
        Dataset identifier
    section_number : int
        Section number (1-5)
    model_name : str, optional
        Model name for model-specific results
    filename : str
        Output filename (default: 'evaluation_summary.json')

    Returns:
    --------
    str : Path to the saved file
    """
    results_dir = get_results_path(dataset_identifier, section_number, model_name)
    ensure_results_dir(results_dir)

    filepath = os.path.join(results_dir, filename)

    # Add metadata
    output = {
        "saved_at": datetime.now().isoformat(),
        "session_timestamp": SESSION_TIMESTAMP,
        "section": section_number,
        "dataset": dataset_identifier,
        "model": model_name,
        "summary": summary
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"[RESULTS] Saved evaluation summary to: {filepath}")
    return filepath


def save_synthetic_sample(
    synthetic_df: pd.DataFrame,
    dataset_identifier: str,
    section_number: int,
    model_name: str,
    n_rows: Optional[int] = None,
    filename: str = "synthetic_sample.csv"
) -> str:
    """
    Save a sample of synthetic data to CSV in the standard results location.

    Parameters:
    -----------
    synthetic_df : pd.DataFrame
        Generated synthetic data
    dataset_identifier : str
        Dataset identifier
    section_number : int
        Section number (1-5)
    model_name : str
        Model name
    n_rows : int, optional
        Number of rows to save (default: all rows, max 1000)
    filename : str
        Output filename (default: 'synthetic_sample.csv')

    Returns:
    --------
    str : Path to the saved file
    """
    results_dir = get_results_path(dataset_identifier, section_number, model_name)
    ensure_results_dir(results_dir)

    filepath = os.path.join(results_dir, filename)

    # Limit rows if needed
    if n_rows is None:
        n_rows = min(len(synthetic_df), 1000)

    sample = synthetic_df.head(n_rows)
    sample.to_csv(filepath, index=False)

    print(f"[RESULTS] Saved {len(sample)} synthetic rows to: {filepath}")
    return filepath


def save_comparison_table(
    comparison_df: pd.DataFrame,
    dataset_identifier: str,
    section_number: int,
    filename: str = "model_comparison.csv"
) -> str:
    """
    Save a model comparison table to CSV in the standard results location.

    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison table with metrics for each model
    dataset_identifier : str
        Dataset identifier
    section_number : int
        Section number (1-5)
    filename : str
        Output filename (default: 'model_comparison.csv')

    Returns:
    --------
    str : Path to the saved file
    """
    results_dir = get_results_path(dataset_identifier, section_number)
    ensure_results_dir(results_dir)

    filepath = os.path.join(results_dir, filename)
    comparison_df.to_csv(filepath, index=True)

    print(f"[RESULTS] Saved comparison table to: {filepath}")
    return filepath


def get_latest_results_path(dataset_identifier: str, section_number: int = None) -> Optional[str]:
    """
    Get the path to the most recent results for a dataset.

    Parameters:
    -----------
    dataset_identifier : str
        Dataset identifier
    section_number : int, optional
        Specific section to look for

    Returns:
    --------
    str or None : Path to latest results, or None if not found
    """
    base_dir = f"results/{dataset_identifier}"
    if not os.path.exists(base_dir):
        return None

    # Find the most recent date folder
    date_folders = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ], reverse=True)

    if not date_folders:
        return None

    latest_date = date_folders[0]
    path = os.path.join(base_dir, latest_date)

    if section_number:
        path = os.path.join(path, f"Section-{section_number}")

    return path if os.path.exists(path) else None
