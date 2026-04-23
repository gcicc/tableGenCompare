"""
Batch Hyperparameter Optimization Module for Synthetic Data Models

This module provides batch HPO functionality to optimize multiple synthetic
data models using Optuna, driven by NOTEBOOK_CONFIG['models_to_run'].

Phase 5 - January 2026
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np

try:
    import optuna
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .model_factory import ModelFactory
from .registry import resolve_models, get_model_display_name
from .search_spaces import get_search_space, get_pruner_config

logger = logging.getLogger(__name__)


def optimize_models_batch(
    data: pd.DataFrame,
    models_to_run: List[str],
    target_column: str,
    categorical_columns: List[str] = None,
    n_trials: int = 50,
    run_mode: str = "full",
    random_state: int = 42,
    verbose: bool = True,
    continue_on_error: bool = True,
    custom_objective_fn: Callable = None
) -> Dict[str, Dict]:
    """
    Run Optuna HPO for multiple models in batch.

    This function iterates through the specified models, runs hyperparameter
    optimization for each, and collects results including the best parameters
    and Optuna study objects.

    Parameters:
    -----------
    data : pd.DataFrame
        The real dataset for training and evaluation
    models_to_run : List[str]
        List of model names to optimize (e.g., ["ctgan", "tvae"])
        Can also be "all" to run all available models
    target_column : str
        Name of the target column in the dataset
    categorical_columns : List[str], optional
        List of categorical column names. If None, auto-detected.
    n_trials : int
        Number of Optuna trials per model (default: 50)
    run_mode : str
        Either "debug" or "full" for search space selection (default: "full")
    random_state : int
        Random seed for reproducibility (default: 42)
    verbose : bool
        Whether to print progress messages (default: True)
    continue_on_error : bool
        If True, continue with other models when one fails (default: True)
    custom_objective_fn : Callable, optional
        Custom objective function factory. If None, uses enhanced_objective_function_v2.

    Returns:
    --------
    Dict[str, Dict] : Results dictionary with structure:
        {
            "ctgan": {
                "study": optuna.Study,
                "best_params": dict,
                "best_score": float,
                "optimization_time": float (seconds),
                "status": "success" | "error",
                "error_message": str (only if status == "error")
            },
            ...
        }

    Example:
    --------
    >>> from src.models.batch_optimization import optimize_models_batch
    >>> results = optimize_models_batch(
    ...     data=data,
    ...     models_to_run=["ctgan", "tvae"],
    ...     target_column="diagnosis",
    ...     n_trials=50,
    ...     run_mode="full"
    ... )
    >>> ctgan_study = results["ctgan"]["study"]
    >>> best_params = results["ctgan"]["best_params"]
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for hyperparameter optimization")

    # Resolve model names
    resolved_models = resolve_models(models_to_run)

    # Filter by dataset size (e.g., remove GReaT for small datasets)
    from .registry import filter_models_by_dataset_size
    resolved_models = filter_models_by_dataset_size(
        resolved_models,
        data_size=len(data),
        verbose=verbose,
    )

    if verbose:
        print(f"\n{'='*60}")
        print("BATCH HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Models to optimize: {len(resolved_models)}")
        print(f"Trials per model: {n_trials}")
        print(f"Run mode: {run_mode}")
        print(f"Target column: {target_column}")
        print(f"{'='*60}\n")

    # Auto-detect categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = _auto_detect_categorical(data, target_column)

    results = {}
    successful = 0
    failed = 0

    for idx, model_name in enumerate(resolved_models, 1):
        display_name = get_model_display_name(model_name)

        if verbose:
            print(f"\n[{idx}/{len(resolved_models)}] Optimizing {display_name}...")
            print(f"{'-'*50}")

        start_time = time.time()

        try:
            # Create objective function for this model
            objective = _create_model_objective(
                model_name=model_name,
                data=data,
                target_column=target_column,
                categorical_columns=categorical_columns,
                run_mode=run_mode,
                random_state=random_state,
                custom_objective_fn=custom_objective_fn,
                verbose=verbose
            )

            # Get pruner configuration
            pruner_config = get_pruner_config(model_name)
            if pruner_config:
                pruner = MedianPruner(**pruner_config)
            else:
                pruner = None

            # Create Optuna study
            study = optuna.create_study(
                direction="maximize",
                pruner=pruner,
                study_name=f"{model_name}_hpo"
            )

            # Suppress Optuna logging if not verbose
            if not verbose:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            # Run optimization
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=verbose
            )

            optimization_time = time.time() - start_time

            results[model_name] = {
                "study": study,
                "best_params": study.best_params,
                "best_score": study.best_value,
                "n_trials_completed": len(study.trials),
                "optimization_time": optimization_time,
                "status": "success"
            }

            successful += 1

            if verbose:
                print(f"  [OK] {display_name} optimization complete")
                print(f"  Best score: {study.best_value:.4f}")
                print(f"  Trials: {len(study.trials)}/{n_trials}")
                print(f"  Time: {optimization_time:.2f}s")

        except Exception as e:
            optimization_time = time.time() - start_time
            error_msg = str(e)

            results[model_name] = {
                "study": None,
                "best_params": None,
                "best_score": None,
                "optimization_time": optimization_time,
                "status": "error",
                "error_message": error_msg
            }

            failed += 1

            if verbose:
                print(f"  [ERROR] {display_name} failed: {error_msg}")

            logger.error(f"Model {model_name} optimization failed: {e}")

            if not continue_on_error:
                raise

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("BATCH OPTIMIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total models: {len(resolved_models)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"{'='*60}\n")

        if successful > 0:
            print("Best scores by model:")
            for name, result in results.items():
                if result["status"] == "success":
                    print(f"  - {get_model_display_name(name)}: {result['best_score']:.4f}")

    return results


def _create_model_objective(
    model_name: str,
    data: pd.DataFrame,
    target_column: str,
    categorical_columns: List[str],
    run_mode: str,
    random_state: int,
    custom_objective_fn: Callable = None,
    verbose: bool = False
) -> Callable:
    """
    Create an Optuna objective function for a specific model.

    Parameters:
    -----------
    model_name : str
        Name of the model
    data : pd.DataFrame
        Training data
    target_column : str
        Target column name
    categorical_columns : List[str]
        Categorical columns
    run_mode : str
        "debug" or "full"
    random_state : int
        Random seed
    custom_objective_fn : Callable, optional
        Custom evaluation function
    verbose : bool
        Whether to print trial progress

    Returns:
    --------
    Callable : Optuna objective function
    """
    from src.objective.functions import enhanced_objective_function_v2

    def objective(trial: 'optuna.Trial') -> float:
        try:
            # Get hyperparameters from search space
            params = get_search_space(
                model_name,
                trial,
                run_mode,
                data_size=len(data),
                n_cols=data.shape[1],
            )

            # Create and configure model
            model = ModelFactory.create(
                model_name,
                device="cpu",
                random_state=random_state
            )

            # Get model-specific training kwargs
            train_kwargs = _get_train_kwargs(
                model_name,
                params,
                categorical_columns,
                target_column,
                data
            )

            # Train model
            model.train(data, **train_kwargs)

            # Generate synthetic data
            synthetic_data = model.generate(len(data))

            # Coerce target dtype to match real data
            synthetic_data = _coerce_target_dtype(data, synthetic_data, target_column)

            # Evaluate using custom or default objective function
            if custom_objective_fn is not None:
                score = custom_objective_fn(data, synthetic_data, target_column, trial)
            else:
                score, _, _ = enhanced_objective_function_v2(
                    data,
                    synthetic_data,
                    target_column,
                    trial=trial
                )

            return float(score)

        except optuna.TrialPruned:
            raise
        except Exception as e:
            if verbose:
                print(f"  Trial {trial.number} failed: {e}")
            logger.warning(f"Trial {trial.number} for {model_name} failed: {e}")
            return 0.0

    return objective


def _get_train_kwargs(
    model_name: str,
    params: Dict[str, Any],
    categorical_columns: List[str],
    target_column: str,
    data: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Convert search space parameters to model training kwargs.

    Different models expect different parameter names and structures.
    """
    model_name = model_name.lower()

    # Filter to only columns that exist in the data (handles one-hot encoding cases)
    if data is not None:
        existing_columns = set(data.columns)
        categorical_columns = [col for col in categorical_columns if col in existing_columns]

    kwargs = {}

    # Common parameters that most models accept
    if "epochs" in params:
        kwargs["epochs"] = params["epochs"]
    if "batch_size" in params:
        kwargs["batch_size"] = params["batch_size"]

    # Model-specific parameter mapping
    if model_name == "ctgan":
        kwargs.update({
            "discrete_columns": categorical_columns,
            "generator_lr": params.get("generator_lr"),
            "discriminator_lr": params.get("discriminator_lr"),
            "generator_dim": params.get("generator_dim"),
            "discriminator_dim": params.get("discriminator_dim"),
            "pac": params.get("pac"),
            "discriminator_steps": params.get("discriminator_steps"),
            "generator_decay": params.get("generator_decay"),
            "discriminator_decay": params.get("discriminator_decay"),
        })
        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

    elif model_name in ["ctabgan", "ctabganplus"]:
        kwargs.update({
            "categorical_columns": categorical_columns,
            "target_col": target_column,
            "test_ratio": params.get("test_ratio", 0.2),
        })

    elif model_name == "ganeraid":
        kwargs.update({
            "categorical_columns": categorical_columns,
            "nr_of_rows": params.get("nr_of_rows"),
            "hidden_feature_space": params.get("hidden_feature_space"),
            "lr_d": params.get("lr_d"),
            "lr_g": params.get("lr_g"),
            "binary_noise": params.get("binary_noise"),
            "generator_decay": params.get("generator_decay"),
            "discriminator_decay": params.get("discriminator_decay"),
            "dropout_generator": params.get("dropout_generator"),
            "dropout_discriminator": params.get("dropout_discriminator"),
        })
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

    elif model_name == "copulagan":
        kwargs.update({
            "discrete_columns": categorical_columns,
            "pac": params.get("pac"),
            "generator_lr": params.get("generator_lr"),
            "discriminator_lr": params.get("discriminator_lr"),
            "generator_decay": params.get("generator_decay"),
            "discriminator_decay": params.get("discriminator_decay"),
        })
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

    elif model_name == "tvae":
        kwargs.update({
            "discrete_columns": categorical_columns,
            "learning_rate": params.get("learning_rate"),
            "embedding_dim": params.get("embedding_dim"),
            "l2scale": params.get("l2scale"),
            "compress_dims": params.get("compress_dims"),
            "decompress_dims": params.get("decompress_dims"),
        })
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

    elif model_name == "pategan":
        kwargs.update({
            "discrete_columns": categorical_columns,
            "generator_lr": params.get("generator_lr"),
            "discriminator_lr": params.get("discriminator_lr"),
            "generator_dim": params.get("generator_dim"),
            "discriminator_dim": params.get("discriminator_dim"),
            "generator_decay": params.get("generator_decay"),
            "discriminator_decay": params.get("discriminator_decay"),
            "num_teachers": params.get("num_teachers"),
            "noise_multiplier": params.get("noise_multiplier"),
            "target_epsilon": params.get("target_epsilon"),
            "lap_scale": params.get("lap_scale"),
        })
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

    elif model_name == "medgan":
        kwargs.update({
            "discrete_columns": categorical_columns,
            "pretrain_epochs": params.get("pretrain_epochs"),
            "latent_dim": params.get("latent_dim"),
            "autoencoder_lr": params.get("autoencoder_lr"),
            "generator_lr": params.get("generator_lr"),
            "discriminator_lr": params.get("discriminator_lr"),
            "l2_reg": params.get("l2_reg"),
            "dropout": params.get("dropout"),
            "autoencoder_dim": params.get("autoencoder_dim"),
            "generator_dim": params.get("generator_dim"),
            "discriminator_dim": params.get("discriminator_dim"),
        })
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

    return kwargs


def _coerce_target_dtype(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: str
) -> pd.DataFrame:
    """
    Coerce synthetic target column dtype to match real data.

    This ensures evaluation functions work correctly by aligning dtypes.
    """
    if target_column not in real_data.columns or target_column not in synthetic_data.columns:
        return synthetic_data

    result = synthetic_data.copy()
    real_dtype = real_data[target_column].dtype

    try:
        if pd.api.types.is_numeric_dtype(real_dtype):
            if pd.api.types.is_integer_dtype(real_dtype):
                result[target_column] = pd.to_numeric(
                    result[target_column], errors='coerce'
                ).fillna(0).astype(int)
            else:
                result[target_column] = pd.to_numeric(
                    result[target_column], errors='coerce'
                )
        else:
            result[target_column] = result[target_column].astype(str)
    except Exception as e:
        logger.warning(f"Could not coerce target dtype: {e}")

    return result


def _auto_detect_categorical(
    data: pd.DataFrame,
    target_column: str
) -> List[str]:
    """Auto-detect categorical columns."""
    categorical = []

    for col in data.columns:
        dtype = data[col].dtype

        if dtype == 'object' or dtype.name == 'category':
            categorical.append(col)
        elif dtype in ['int64', 'int32'] and data[col].nunique() <= 20:
            categorical.append(col)
        elif data[col].nunique() == 2:
            categorical.append(col)

    return categorical


def extract_studies_to_globals(
    results: Dict[str, Dict],
    scope: dict = None,
    suffix: str = "_study"
) -> List[str]:
    """
    Extract Optuna studies from batch results to global variables.

    Creates global variables like `ctgan_study`, `tvae_study`, etc.
    for backward compatibility with Section 4.2 notebook cells.

    Parameters:
    -----------
    results : Dict[str, Dict]
        Results from optimize_models_batch()
    scope : dict, optional
        Namespace to inject variables into (e.g., globals())
    suffix : str
        Suffix for variable names (default: "_study")

    Returns:
    --------
    List[str] : List of variable names created

    Example:
    --------
    >>> results = optimize_models_batch(...)
    >>> created_vars = extract_studies_to_globals(results, globals())
    >>> print(created_vars)  # ['ctgan_study', 'tvae_study', ...]
    """
    created_vars = []

    for model_name, result in results.items():
        if result["status"] == "success" and result["study"] is not None:
            var_name = f"{model_name}{suffix}"

            if scope is not None:
                scope[var_name] = result["study"]

            created_vars.append(var_name)

    return created_vars


print("[OK] Batch optimization module loaded from src/models/batch_optimization.py")
