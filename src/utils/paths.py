"""
Path Utilities for Results Organization

This module provides utilities for managing file paths and results directories
in a standardized way across all notebooks.
"""

import os
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


def get_results_path(dataset_identifier, section_number):
    """
    Generate standardized results path.

    Parameters:
    -----------
    dataset_identifier : str
        Dataset identifier (e.g., 'alzheimers-disease-data')
    section_number : int
        Section number (1-5)

    Returns:
    --------
    str : Standardized path in format: results/{dataset_id}/{YYYY-MM-DD}/Section-{N}

    Examples:
    ---------
    >>> get_results_path('alzheimers-disease-data', 2)
    'results/alzheimers-disease-data/2024-12-05/Section-2'
    """
    return f"results/{dataset_identifier}/{SESSION_TIMESTAMP}/Section-{section_number}"
