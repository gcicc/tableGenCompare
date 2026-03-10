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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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
    if TORCH_AVAILABLE and torch.cuda.is_available():
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
    continue_on_error: bool = True,
    checkpoint=None
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
        # Show GPU status and per-model device assignments
        if TORCH_AVAILABLE and torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
        else:
            print("GPU available: No (using CPU)")
        print(f"Device assignments:")
        for m in resolved_models:
            print(f"  - {get_model_display_name(m)}: {_get_device_for_model(m)}")
        print(f"{'='*60}\n")

    results = {}
    successful = 0
    failed = 0

    for idx, model_name in enumerate(resolved_models, 1):
        display_name = get_model_display_name(model_name)

        # Check checkpoint for this model
        ckpt_id = f"section_3.1_model_{model_name}"
        if checkpoint is not None and checkpoint.exists(ckpt_id):
            saved = checkpoint.load(ckpt_id)
            results[model_name] = saved
            successful += 1
            if verbose:
                print(f"\n[{idx}/{len(resolved_models)}] [RESUME] {display_name} loaded from checkpoint ({saved['training_time']:.2f}s)")
            continue

        if verbose:
            print(f"\n[{idx}/{len(resolved_models)}] Training {display_name}...")
            print(f"{'-'*50}")

        start_time = time.time()

        try:
            # Determine appropriate device for this model
            device = _get_device_for_model(model_name)
            if verbose:
                print(f"  Device: {device}")

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

            # Save checkpoint for this model (without model object - not picklable)
            if checkpoint is not None:
                checkpoint.save(ckpt_id, {
                    "synthetic_data": synthetic_data,
                    "model": None,
                    "training_time": training_time,
                    "status": "success"
                })

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

    # Filter to only columns that exist in the data (handles one-hot encoding cases)
    existing_columns = set(data.columns)
    categorical_columns = [col for col in categorical_columns if col in existing_columns]

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


def train_models_batch_with_best_params(
    data: pd.DataFrame,
    target_column: str,
    models_to_run: List[str] = None,
    categorical_columns: List[str] = None,
    n_samples: int = None,
    random_state: int = 42,
    section_number: int = 4,
    dataset_identifier: str = None,
    scope: dict = None,
    verbose: bool = True,
    continue_on_error: bool = True,
    checkpoint=None
) -> Dict[str, Dict]:
    """
    Train models with best parameters from Section 4 HPO.

    This function loads best hyperparameters from Section 4 CSV files,
    trains each model with those parameters, and generates synthetic data.
    Creates `synthetic_{model}_final` variables for Section 5.2 compatibility.

    Parameters:
    -----------
    data : pd.DataFrame
        The real dataset to train models on
    target_column : str
        Name of the target column in the dataset
    models_to_run : List[str], optional
        List of model names to train. Defaults to NOTEBOOK_CONFIG['models_to_run'] or all.
    categorical_columns : List[str], optional
        List of categorical column names. If None, auto-detected.
    n_samples : int, optional
        Number of synthetic samples to generate. Defaults to len(data).
    random_state : int
        Random seed for reproducibility (default: 42)
    section_number : int
        Section number to load parameters from (default: 4)
    dataset_identifier : str, optional
        Dataset identifier for loading parameters. Auto-detected if None.
    scope : dict, optional
        Namespace to inject synthetic_*_final variables (e.g., globals())
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
                "params_used": dict,
                "training_time": float (seconds),
                "status": "success" | "error",
                "objective_score": float,
                "similarity_score": float,
                "accuracy_score": float,
                "error_message": str (only if status == "error")
            },
            ...
        }

    Example:
    --------
    >>> from src.models.batch_training import train_models_batch_with_best_params
    >>> results = train_models_batch_with_best_params(
    ...     data=data,
    ...     target_column="diagnosis",
    ...     scope=globals(),  # Creates synthetic_*_final variables
    ...     verbose=True
    ... )
    >>> # Access synthetic data
    >>> synthetic_ctgan_final  # Now available in globals
    """
    from src.utils.parameters import load_best_parameters_from_csv
    from src.config import DATASET_IDENTIFIER as CONFIG_DATASET_IDENTIFIER

    if n_samples is None:
        n_samples = len(data)

    # Auto-detect dataset identifier
    if dataset_identifier is None:
        if scope and 'DATASET_IDENTIFIER' in scope:
            dataset_identifier = scope['DATASET_IDENTIFIER']
        else:
            dataset_identifier = CONFIG_DATASET_IDENTIFIER

    # Resolve models to run
    if models_to_run is None:
        if scope and 'NOTEBOOK_CONFIG' in scope:
            models_to_run = scope['NOTEBOOK_CONFIG'].get('models_to_run', 'all')
        else:
            models_to_run = 'all'

    resolved_models = resolve_models(models_to_run)

    if verbose:
        print(f"\n{'='*60}")
        print("SECTION 5.1: BATCH TRAINING WITH BEST PARAMETERS")
        print(f"{'='*60}")
        print(f"Models to train: {len(resolved_models)}")
        print(f"Dataset shape: {data.shape}")
        print(f"Target column: {target_column}")
        print(f"Samples to generate: {n_samples}")
        print(f"Loading parameters from: Section {section_number}")
        # Show GPU status and per-model device assignments
        if TORCH_AVAILABLE and torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
        else:
            print("GPU available: No (using CPU)")
        print(f"Device assignments:")
        for m in resolved_models:
            print(f"  - {get_model_display_name(m)}: {_get_device_for_model(m)}")
        print(f"{'='*60}\n")

    # Load best parameters from Section 4
    if verbose:
        print("[1/3] Loading best parameters from Section 4...")

    param_data = load_best_parameters_from_csv(
        section_number=section_number,
        dataset_identifier=dataset_identifier,
        fallback_to_memory=True,
        scope=scope
    )

    if verbose:
        print(f"   Parameter source: {param_data['source']}")
        print(f"   Models with parameters: {param_data['models_count']}")

    # Try to import evaluation function for scoring
    try:
        from src.objective.functions import enhanced_objective_function_v2
        has_eval_func = True
    except ImportError:
        has_eval_func = False
        if verbose:
            print("   Note: Enhanced objective function not available for scoring")

    results = {}
    successful = 0
    failed = 0

    if verbose:
        print(f"\n[2/3] Training models with optimized parameters...")

    for idx, model_name in enumerate(resolved_models, 1):
        display_name = get_model_display_name(model_name)

        # Check checkpoint for this model
        ckpt_id = f"section_5.1_model_{model_name}"
        if checkpoint is not None and checkpoint.exists(ckpt_id):
            saved = checkpoint.load(ckpt_id)
            results[model_name] = saved
            successful += 1
            if verbose:
                print(f"\n[{idx}/{len(resolved_models)}] [RESUME] {display_name} loaded from checkpoint ({saved['training_time']:.2f}s)")
            continue

        if verbose:
            print(f"\n[{idx}/{len(resolved_models)}] Training {display_name}...")
            print(f"{'-'*50}")

        start_time = time.time()

        try:
            # Get best parameters for this model
            model_key = model_name.lower().replace('-', '').replace('+', 'plus')
            best_params = param_data['parameters'].get(model_key, {})

            if not best_params:
                if verbose:
                    print(f"  WARNING: No parameters found for {display_name}, using defaults")
                best_params = {}

            # Determine appropriate device for this model
            device = _get_device_for_model(model_name)
            if verbose:
                print(f"  Device: {device}")

            # Create model instance via factory
            model = ModelFactory.create(
                model_name,
                device=device,
                random_state=random_state
            )

            # Prepare training kwargs with best params
            train_kwargs = _prepare_training_kwargs_with_best_params(
                model_name,
                data,
                categorical_columns,
                target_column,
                best_params
            )

            if verbose:
                print(f"  Parameters loaded: {len(best_params)}")
                for param, value in list(train_kwargs.items())[:5]:  # Show first 5
                    if isinstance(value, float):
                        print(f"    - {param}: {value:.4f}")
                    else:
                        print(f"    - {param}: {value}")
                if len(train_kwargs) > 5:
                    print(f"    ... and {len(train_kwargs) - 5} more")

            # Train the model
            if verbose:
                print(f"  Training {display_name}...")

            model.train(data, **train_kwargs)

            # Generate synthetic data
            if verbose:
                print(f"  Generating {n_samples} synthetic samples...")

            synthetic_data = model.generate(n_samples)

            training_time = time.time() - start_time

            # Evaluate if function available
            objective_score = 0.0
            similarity_score = 0.0
            accuracy_score = 0.0

            if has_eval_func:
                try:
                    objective_score, similarity_score, accuracy_score = enhanced_objective_function_v2(
                        real_data=data,
                        synthetic_data=synthetic_data,
                        target_column=target_column
                    )
                except Exception as eval_error:
                    if verbose:
                        print(f"  Note: Evaluation failed: {eval_error}")

            results[model_name] = {
                "synthetic_data": synthetic_data,
                "model": model,
                "params_used": best_params,
                "training_time": training_time,
                "status": "success",
                "objective_score": objective_score,
                "similarity_score": similarity_score,
                "accuracy_score": accuracy_score,
                "param_source": param_data['source']
            }

            # Save checkpoint for this model (without model object)
            if checkpoint is not None:
                checkpoint.save(ckpt_id, {
                    "synthetic_data": synthetic_data,
                    "model": None,
                    "params_used": best_params,
                    "training_time": training_time,
                    "status": "success",
                    "objective_score": objective_score,
                    "similarity_score": similarity_score,
                    "accuracy_score": accuracy_score,
                    "param_source": param_data['source']
                })

            successful += 1

            if verbose:
                print(f"  [OK] {display_name} completed in {training_time:.2f}s")
                print(f"  Synthetic data shape: {synthetic_data.shape}")
                if has_eval_func:
                    print(f"  Objective score: {objective_score:.4f}")

        except Exception as e:
            training_time = time.time() - start_time
            error_msg = str(e)

            results[model_name] = {
                "synthetic_data": None,
                "model": None,
                "params_used": {},
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

    # Extract synthetic data to globals with _final suffix
    if verbose:
        print(f"\n[3/3] Creating synthetic_*_final variables...")

    created_vars = extract_final_synthetic_to_globals(results, scope)

    if verbose:
        print(f"   Created: {created_vars}")

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("BATCH TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total models: {len(resolved_models)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Parameter source: {param_data['source']}")
        print(f"{'='*60}\n")

        # List successful models
        if successful > 0:
            print("Successful models:")
            for name, result in results.items():
                if result["status"] == "success":
                    score_str = f", score: {result['objective_score']:.4f}" if result.get('objective_score') else ""
                    print(f"  - {get_model_display_name(name)}: {result['training_time']:.2f}s{score_str}")

        # List failed models
        if failed > 0:
            print("\nFailed models:")
            for name, result in results.items():
                if result["status"] == "error":
                    print(f"  - {get_model_display_name(name)}: {result['error_message']}")

    return results


def _prepare_training_kwargs_with_best_params(
    model_name: str,
    data: pd.DataFrame,
    categorical_columns: Optional[List[str]],
    target_column: str,
    best_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare training kwargs for a model with best parameters from HPO.

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
    best_params : Dict[str, Any]
        Best parameters from HPO

    Returns:
    --------
    Dict[str, Any] : Training keyword arguments
    """
    model_name = model_name.lower()

    # Auto-detect categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = _auto_detect_categorical(data, target_column)

    # Filter to only columns that exist in the data (handles one-hot encoding cases)
    existing_columns = set(data.columns)
    categorical_columns = [col for col in categorical_columns if col in existing_columns]

    # Start with best params
    kwargs = dict(best_params)

    # Model-specific kwargs additions
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
        if "epochs" not in kwargs:
            kwargs["epochs"] = 500  # Default if not in best_params

    elif model_name == "pategan":
        # PATE-GAN uses discrete_columns
        kwargs["discrete_columns"] = categorical_columns

    elif model_name == "medgan":
        # MEDGAN uses discrete_columns
        kwargs["discrete_columns"] = categorical_columns

    return kwargs


def extract_final_synthetic_to_globals(
    results: Dict[str, Dict],
    scope: dict = None
) -> List[str]:
    """
    Extract synthetic data from batch results to global variables with _final suffix.

    This function creates global variables like `synthetic_ctgan_final`,
    `synthetic_tvae_final`, etc. for Section 5.2 compatibility.

    Parameters:
    -----------
    results : Dict[str, Dict]
        Results from train_models_batch_with_best_params()
    scope : dict, optional
        Namespace to inject variables into (e.g., globals())
        If None, returns variable mapping without injection.

    Returns:
    --------
    List[str] : List of variable names created

    Example:
    --------
    >>> results = train_models_batch_with_best_params(...)
    >>> created_vars = extract_final_synthetic_to_globals(results, globals())
    >>> print(created_vars)  # ['synthetic_ctgan_final', 'synthetic_tvae_final', ...]
    """
    created_vars = []

    for model_name, result in results.items():
        if result["status"] == "success" and result["synthetic_data"] is not None:
            # Use _final suffix for Section 5 pattern
            var_name = f"synthetic_{model_name}_final"

            if scope is not None:
                scope[var_name] = result["synthetic_data"]

                # Also create the *_final_results variable for compatibility
                results_var_name = f"{model_name}_final_results"
                scope[results_var_name] = {
                    'model_name': get_model_display_name(model_name),
                    'objective_score': result.get('objective_score', 0.0),
                    'similarity_score': result.get('similarity_score', 0.0),
                    'accuracy_score': result.get('accuracy_score', 0.0),
                    'best_params': result.get('params_used', {}),
                    'parameter_source': result.get('param_source', 'unknown'),
                    'synthetic_data': result["synthetic_data"]
                }

            created_vars.append(var_name)

    return created_vars


print("[OK] Batch training module loaded from src/models/batch_training.py")
