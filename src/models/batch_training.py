"""
Batch Training Module for Synthetic Data Models

This module provides batch training functionality to train multiple synthetic
data models in sequence, driven by NOTEBOOK_CONFIG['models_to_run'].

Phase 5 - January 2026
"""

import time
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import torch

from .model_factory import ModelFactory
from .registry import resolve_models, get_model_display_name

logger = logging.getLogger(__name__)

# Models known to have CUDA issues - force CPU for these
FORCE_CPU_MODELS = {"ganeraid"}


def _get_device_for_model(model_name: str) -> str:
    """
    Determine the appropriate device for a model.

    Most models can use CUDA if available, but some have known issues
    and should be forced to use CPU.
    """
    # Check if model should be forced to CPU
    if model_name.lower() in FORCE_CPU_MODELS:
        logger.info(f"Model {model_name} forced to CPU due to known CUDA issues")
        return "cpu"

    # Use CUDA if available for other models
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def train_models_batch(
    data: pd.DataFrame,
    models_to_run: List[str],
    target_column: str,
    categorical_columns: List[str] = None,
    n_samples: int = None,
    random_state: int = 42,
    verbose: bool = True,
    continue_on_error: bool = True
) -> Dict[str, Dict]:
    """
    Train multiple models and generate synthetic data in batch.

    This function iterates through the specified models, trains each one,
    and generates synthetic data. Results are collected in a dictionary
    for downstream evaluation.

    Parameters:
    -----------
    data : pd.DataFrame
        The real dataset to train models on
    models_to_run : List[str]
        List of model names to train (e.g., ["ctgan", "tvae", "copulagan"])
        Can also be "all" to run all available models
    target_column : str
        Name of the target column in the dataset
    categorical_columns : List[str], optional
        List of categorical column names. If None, auto-detected.
    n_samples : int, optional
        Number of synthetic samples to generate. Defaults to len(data).
    random_state : int
        Random seed for reproducibility (default: 42)
    verbose : bool
        Whether to print progress messages (default: True)
    continue_on_error : bool
        If True, continue with other models when one fails (default: True)

    Returns:
    --------
    Dict[str, Dict] : Results dictionary with structure:
        {
            "ctgan": {
                "synthetic_data": pd.DataFrame,
                "model": SyntheticDataModel instance,
                "training_time": float (seconds),
                "status": "success" | "error",
                "error_message": str (only if status == "error")
            },
            ...
        }

    Example:
    --------
    >>> from src.models.batch_training import train_models_batch
    >>> results = train_models_batch(
    ...     data=data,
    ...     models_to_run=["ctgan", "tvae"],
    ...     target_column="diagnosis",
    ...     n_samples=len(data)
    ... )
    >>> synthetic_data_ctgan = results["ctgan"]["synthetic_data"]
    """
    if n_samples is None:
        n_samples = len(data)

    # Resolve model names (handles "all" and aliases)
    resolved_models = resolve_models(models_to_run)

    if verbose:
        print(f"\n{'='*60}")
        print("BATCH MODEL TRAINING")
        print(f"{'='*60}")
        print(f"Models to train: {len(resolved_models)}")
        print(f"Dataset shape: {data.shape}")
        print(f"Target column: {target_column}")
        print(f"Samples to generate: {n_samples}")
        print(f"{'='*60}\n")

    results = {}
    successful = 0
    failed = 0

    for idx, model_name in enumerate(resolved_models, 1):
        display_name = get_model_display_name(model_name)

        if verbose:
            print(f"\n[{idx}/{len(resolved_models)}] Training {display_name}...")
            print(f"{'-'*50}")

        start_time = time.time()

        try:
            # Determine appropriate device for this model
            device = _get_device_for_model(model_name)

            # Create model instance via factory
            model = ModelFactory.create(
                model_name,
                device=device,
                random_state=random_state
            )

            # Prepare training kwargs based on model type
            train_kwargs = _get_model_train_kwargs(
                model_name,
                data,
                categorical_columns,
                target_column
            )

            # Train the model
            if verbose:
                print(f"  Training {display_name}...")

            model.train(data, **train_kwargs)

            # Generate synthetic data
            if verbose:
                print(f"  Generating {n_samples} synthetic samples...")

            synthetic_data = model.generate(n_samples)

            training_time = time.time() - start_time

            results[model_name] = {
                "synthetic_data": synthetic_data,
                "model": model,
                "training_time": training_time,
                "status": "success"
            }

            successful += 1

            if verbose:
                print(f"  [OK] {display_name} completed in {training_time:.2f}s")
                print(f"  Synthetic data shape: {synthetic_data.shape}")

        except Exception as e:
            training_time = time.time() - start_time
            error_msg = str(e)

            results[model_name] = {
                "synthetic_data": None,
                "model": None,
                "training_time": training_time,
                "status": "error",
                "error_message": error_msg
            }

            failed += 1

            if verbose:
                print(f"  [ERROR] {display_name} failed: {error_msg}")

            logger.error(f"Model {model_name} failed: {e}")

            if not continue_on_error:
                raise

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("BATCH TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total models: {len(resolved_models)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"{'='*60}\n")

        # List successful models
        if successful > 0:
            print("Successful models:")
            for name, result in results.items():
                if result["status"] == "success":
                    print(f"  - {get_model_display_name(name)}: {result['training_time']:.2f}s")

        # List failed models
        if failed > 0:
            print("\nFailed models:")
            for name, result in results.items():
                if result["status"] == "error":
                    print(f"  - {get_model_display_name(name)}: {result['error_message']}")

    return results


def _get_model_train_kwargs(
    model_name: str,
    data: pd.DataFrame,
    categorical_columns: Optional[List[str]],
    target_column: str
) -> Dict[str, Any]:
    """
    Get model-specific training keyword arguments.

    Different models require different parameters for training.
    This function returns the appropriate kwargs for each model type.

    Parameters:
    -----------
    model_name : str
        Name of the model
    data : pd.DataFrame
        Training data
    categorical_columns : List[str], optional
        Categorical columns
    target_column : str
        Target column name

    Returns:
    --------
    Dict[str, Any] : Training keyword arguments
    """
    model_name = model_name.lower()

    # Auto-detect categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = _auto_detect_categorical(data, target_column)

    # Base kwargs for all models
    kwargs = {}

    # Model-specific kwargs
    if model_name in ["ctgan", "copulagan", "tvae"]:
        # SDV-based models use discrete_columns
        kwargs["discrete_columns"] = categorical_columns

    elif model_name in ["ctabgan", "ctabganplus"]:
        # CTABGAN models use categorical_columns and target_col
        kwargs["categorical_columns"] = categorical_columns
        kwargs["target_col"] = target_column

    elif model_name == "ganeraid":
        # GANerAid uses categorical_columns
        kwargs["categorical_columns"] = categorical_columns
        kwargs["epochs"] = 500  # Reduced for demo (default is 5000)

    elif model_name == "pategan":
        # PATE-GAN uses discrete_columns
        kwargs["discrete_columns"] = categorical_columns

    elif model_name == "medgan":
        # MEDGAN uses discrete_columns
        kwargs["discrete_columns"] = categorical_columns

    return kwargs


def _auto_detect_categorical(
    data: pd.DataFrame,
    target_column: str
) -> List[str]:
    """
    Auto-detect categorical columns in the dataset.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    target_column : str
        Target column name (excluded from detection if binary)

    Returns:
    --------
    List[str] : List of categorical column names
    """
    categorical = []

    for col in data.columns:
        # Skip target column for classification tasks
        # (it will be handled specially by models)
        dtype = data[col].dtype

        if dtype == 'object' or dtype.name == 'category':
            categorical.append(col)
        elif dtype in ['int64', 'int32'] and data[col].nunique() <= 20:
            categorical.append(col)
        elif data[col].nunique() == 2:
            categorical.append(col)

    return categorical


def extract_synthetic_data_to_globals(
    results: Dict[str, Dict],
    scope: dict = None,
    prefix: str = "synthetic_data_"
) -> List[str]:
    """
    Extract synthetic data from batch results to global variables.

    This function creates global variables like `synthetic_data_ctgan`,
    `synthetic_data_tvae`, etc. for backward compatibility with existing
    notebook cells in Section 3.2.

    Parameters:
    -----------
    results : Dict[str, Dict]
        Results from train_models_batch()
    scope : dict, optional
        Namespace to inject variables into (e.g., globals())
        If None, returns variable mapping without injection.
    prefix : str
        Prefix for variable names (default: "synthetic_data_")

    Returns:
    --------
    List[str] : List of variable names created

    Example:
    --------
    >>> results = train_models_batch(...)
    >>> created_vars = extract_synthetic_data_to_globals(results, globals())
    >>> print(created_vars)  # ['synthetic_data_ctgan', 'synthetic_data_tvae', ...]
    """
    created_vars = []

    for model_name, result in results.items():
        if result["status"] == "success" and result["synthetic_data"] is not None:
            var_name = f"{prefix}{model_name}"

            if scope is not None:
                scope[var_name] = result["synthetic_data"]

            created_vars.append(var_name)

    return created_vars


print("[OK] Batch training module loaded from src/models/batch_training.py")
