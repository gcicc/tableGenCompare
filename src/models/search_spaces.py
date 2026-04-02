"""
Hyperparameter Search Spaces for Synthetic Data Models

This module provides centralized search space definitions for Optuna-based
hyperparameter optimization. Each model has debug and full mode search spaces
that match the original notebook cells.

Phase 5 - January 2026
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_search_space(
    model_name: str,
    trial: 'optuna.Trial',
    run_mode: str = "full",
    data_size: int = None
) -> Dict[str, Any]:
    """
    Get hyperparameter search space for Optuna trial.

    This function returns sampled hyperparameters for the specified model,
    preserving the debug/full mode distinction from the original notebook cells.

    Parameters:
    -----------
    model_name : str
        Model name (e.g., "ctgan", "tvae", "copulagan")
    trial : optuna.Trial
        Optuna trial object for hyperparameter sampling
    run_mode : str
        Either "debug" or "full" (default: "full")
    data_size : int, optional
        Dataset size (needed for GANerAid feasible triples)

    Returns:
    --------
    Dict[str, Any] : Sampled hyperparameters for the model

    Raises:
    -------
    optuna.TrialPruned : If constraints are violated (e.g., batch_size % pac != 0)
    ValueError : If model_name is not supported

    Example:
    --------
    >>> def objective(trial):
    ...     params = get_search_space("ctgan", trial, "debug")
    ...     # Use params to configure and train model
    ...     return score
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for hyperparameter search spaces")

    model_name = model_name.lower()

    if model_name == "ctgan":
        return _get_ctgan_search_space(trial, run_mode)
    elif model_name == "ctabgan":
        return _get_ctabgan_search_space(trial, run_mode)
    elif model_name == "ctabganplus":
        return _get_ctabganplus_search_space(trial, run_mode)
    elif model_name == "ganeraid":
        return _get_ganeraid_search_space(trial, run_mode, data_size)
    elif model_name == "copulagan":
        return _get_copulagan_search_space(trial, run_mode)
    elif model_name == "tvae":
        return _get_tvae_search_space(trial, run_mode)
    elif model_name == "pategan":
        return _get_pategan_search_space(trial, run_mode)
    elif model_name == "medgan":
        return _get_medgan_search_space(trial, run_mode)
    elif model_name == "tabddpm":
        return _get_tabddpm_search_space(trial, run_mode)
    elif model_name == "great":
        return _get_great_search_space(trial, run_mode)
    else:
        raise ValueError(f"No search space defined for model: {model_name}")


def get_pruner_config(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get pruner configuration for a model.

    Parameters:
    -----------
    model_name : str
        Model name

    Returns:
    --------
    Optional[Dict[str, Any]] : Pruner configuration or None if no pruner

    Example:
    --------
    >>> config = get_pruner_config("ctgan")
    >>> if config:
    ...     pruner = optuna.pruners.MedianPruner(**config)
    """
    model_name = model_name.lower()

    # Models with MedianPruner
    if model_name in ["ctgan", "ctabganplus"]:
        return {
            "n_startup_trials": 5,
            "n_warmup_steps": 2
        }

    # Models without pruner (no intermediate reporting)
    return None


def _get_ctgan_search_space(trial: 'optuna.Trial', run_mode: str) -> Dict[str, Any]:
    """
    CTGAN search space from Cell 54.

    Constraint: batch_size % pac == 0
    """
    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", 200, 400, step=50)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        pac = trial.suggest_categorical("pac", [1, 2, 4, 8])
        generator_lr = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
        discriminator_lr = trial.suggest_float("discriminator_lr", 1e-5, 1e-3, log=True)
        discriminator_steps = trial.suggest_int("discriminator_steps", 1, 3)
        generator_decay = trial.suggest_float("generator_decay", 1e-8, 1e-4, log=True)
        discriminator_decay = trial.suggest_float("discriminator_decay", 1e-8, 1e-4, log=True)
        generator_dim = trial.suggest_categorical(
            "generator_dim", [(128, 128), (256, 256), (256, 512, 256)]
        )
        discriminator_dim = trial.suggest_categorical(
            "discriminator_dim", [(128, 128), (256, 256), (256, 512, 256)]
        )
    else:  # full mode
        epochs = trial.suggest_int("epochs", 100, 1000, step=50)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        pac = trial.suggest_categorical("pac", [1, 2, 4, 8, 10])
        generator_lr = trial.suggest_float("generator_lr", 5e-6, 5e-3, log=True)
        discriminator_lr = trial.suggest_float("discriminator_lr", 5e-6, 5e-3, log=True)
        discriminator_steps = trial.suggest_int("discriminator_steps", 1, 5)
        generator_decay = trial.suggest_float("generator_decay", 1e-8, 1e-4, log=True)
        discriminator_decay = trial.suggest_float("discriminator_decay", 1e-8, 1e-4, log=True)
        generator_dim = trial.suggest_categorical(
            "generator_dim",
            [(128, 128), (256, 256), (512, 256), (256, 512), (512, 512),
             (128, 256, 128), (256, 128, 64), (256, 512, 256)]
        )
        discriminator_dim = trial.suggest_categorical(
            "discriminator_dim",
            [(128, 128), (256, 256), (256, 512), (512, 256),
             (128, 256, 128), (256, 512, 256)]
        )

    # CTGAN constraint: batch_size must be divisible by pac
    if batch_size % pac != 0:
        raise optuna.TrialPruned()

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "pac": pac,
        "generator_lr": generator_lr,
        "discriminator_lr": discriminator_lr,
        "discriminator_steps": discriminator_steps,
        "generator_decay": generator_decay,
        "discriminator_decay": discriminator_decay,
        "generator_dim": generator_dim,
        "discriminator_dim": discriminator_dim
    }


def _get_ctabgan_search_space(trial: 'optuna.Trial', run_mode: str) -> Dict[str, Any]:
    """
    CTAB-GAN search space from Cell 56.

    Note: test_ratio is fixed at 0.2 (not a synthesis-quality hyperparameter)
    """
    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", 200, 400, step=50)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    else:  # full mode
        epochs = trial.suggest_int("epochs", 200, 800, step=50)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "test_ratio": 0.2  # Fixed, not tuned
    }


def _get_ctabganplus_search_space(trial: 'optuna.Trial', run_mode: str) -> Dict[str, Any]:
    """
    CTAB-GAN+ search space from Cell 58.

    Note: test_ratio is fixed at 0.2
    """
    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", 200, 400, step=50)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    else:  # full mode
        epochs = trial.suggest_int("epochs", 150, 1000, step=50)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "test_ratio": 0.2  # Fixed, not tuned
    }


def _get_ganeraid_search_space(
    trial: 'optuna.Trial',
    run_mode: str,
    data_size: int = None
) -> Dict[str, Any]:
    """
    GANerAid search space from Cell 60.

    Uses precomputed feasible triples for (nr_of_rows, batch_size, hidden_feature_space)
    with divisibility constraints.
    """
    if data_size is None:
        data_size = 1000  # Default assumption

    # Compute feasible triples
    feasible_triples = _compute_ganeraid_feasible_triples(run_mode, data_size)

    if not feasible_triples:
        raise optuna.TrialPruned("No feasible parameter triples for GANerAid")

    # Sample from feasible triples
    triple_idx = trial.suggest_categorical(
        "feasible_triple_idx",
        list(range(len(feasible_triples)))
    )
    nr_of_rows, batch_size, hidden_feature_space = feasible_triples[triple_idx]

    # Sample other parameters
    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", 100, 300, step=50)
    else:
        epochs = trial.suggest_int("epochs", 100, 500, step=50)

    lr_d = trial.suggest_float("lr_d", 1e-6, 5e-3, log=True)
    lr_g = trial.suggest_float("lr_g", 1e-6, 5e-3, log=True)
    binary_noise = trial.suggest_float("binary_noise", 0.05, 0.6)
    generator_decay = trial.suggest_float("generator_decay", 1e-8, 1e-3, log=True)
    discriminator_decay = trial.suggest_float("discriminator_decay", 1e-8, 1e-3, log=True)
    dropout_generator = trial.suggest_float("dropout_generator", 0.0, 0.5)
    dropout_discriminator = trial.suggest_float("dropout_discriminator", 0.0, 0.5)

    return {
        "epochs": epochs,
        "nr_of_rows": nr_of_rows,
        "batch_size": batch_size,
        "hidden_feature_space": hidden_feature_space,
        "lr_d": lr_d,
        "lr_g": lr_g,
        "binary_noise": binary_noise,
        "generator_decay": generator_decay,
        "discriminator_decay": discriminator_decay,
        "dropout_generator": dropout_generator,
        "dropout_discriminator": dropout_discriminator
    }


def _compute_ganeraid_feasible_triples(
    run_mode: str,
    data_size: int
) -> List[Tuple[int, int, int]]:
    """
    Compute feasible (nr_of_rows, batch_size, hidden_feature_space) triples for GANerAid.

    Constraints:
    - batch_size % nr_of_rows == 0
    - hidden_feature_space % nr_of_rows == 0
    - nr_of_rows >= 4 and nr_of_rows < data_size
    """
    if run_mode == "debug":
        row_candidates = [4, 5, 8, 10, 16, 20, 25]
        batch_choices = [64, 100, 128, 200, 256]
        hidden_choices = [100, 150, 200, 300, 400]
    else:
        row_candidates = [4, 5, 8, 10, 16, 20, 25, 32, 40]
        batch_choices = [32, 64, 100, 128, 200, 256, 400, 500]
        hidden_choices = [100, 150, 200, 300, 400, 500, 600]

    feasible = []
    for nr in row_candidates:
        if nr < 4 or nr >= data_size:
            continue
        for bs in batch_choices:
            if bs % nr != 0:
                continue
            for hfs in hidden_choices:
                if hfs % nr != 0:
                    continue
                feasible.append((nr, bs, hfs))

    return feasible


def _get_copulagan_search_space(trial: 'optuna.Trial', run_mode: str) -> Dict[str, Any]:
    """
    CopulaGAN search space from Cell 62.
    """
    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", 100, 300, step=50)
        batch_size = trial.suggest_categorical("batch_size", [100, 200, 500])
        generator_lr = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
        discriminator_lr = trial.suggest_float("discriminator_lr", 1e-5, 1e-3, log=True)
        generator_decay = trial.suggest_float("generator_decay", 1e-8, 1e-4, log=True)
        discriminator_decay = trial.suggest_float("discriminator_decay", 1e-8, 1e-4, log=True)
    else:  # full mode
        epochs = trial.suggest_int("epochs", 100, 800, step=50)
        batch_size = trial.suggest_categorical("batch_size", [100, 200, 500, 1000])
        generator_lr = trial.suggest_float("generator_lr", 5e-6, 5e-3, log=True)
        discriminator_lr = trial.suggest_float("discriminator_lr", 5e-6, 5e-3, log=True)
        generator_decay = trial.suggest_float("generator_decay", 1e-8, 1e-3, log=True)
        discriminator_decay = trial.suggest_float("discriminator_decay", 1e-8, 1e-3, log=True)

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "generator_lr": generator_lr,
        "discriminator_lr": discriminator_lr,
        "generator_decay": generator_decay,
        "discriminator_decay": discriminator_decay
    }


def _get_tvae_search_space(trial: 'optuna.Trial', run_mode: str) -> Dict[str, Any]:
    """
    TVAE search space from Cell 64.
    """
    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", 100, 300, step=50)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        embedding_dim = trial.suggest_int("embedding_dim", 32, 128, step=32)
        l2scale = trial.suggest_float("l2scale", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
    else:  # full mode
        epochs = trial.suggest_int("epochs", 100, 800, step=50)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        embedding_dim = trial.suggest_int("embedding_dim", 32, 256, step=32)
        l2scale = trial.suggest_float("l2scale", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

    compress_dims = trial.suggest_categorical(
        "compress_dims", [(128, 128), (256, 128), (256, 128, 64)]
    )
    decompress_dims = trial.suggest_categorical(
        "decompress_dims", [(128, 128), (64, 128), (64, 128, 256)]
    )

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "embedding_dim": embedding_dim,
        "l2scale": l2scale,
        "dropout": dropout,
        "compress_dims": compress_dims,
        "decompress_dims": decompress_dims
    }


def _get_pategan_search_space(trial: 'optuna.Trial', run_mode: str) -> Dict[str, Any]:
    """
    PATE-GAN search space (Phase 5 - January 2026).
    """
    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", 100, 300, step=50)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        num_teachers = trial.suggest_int("num_teachers", 5, 20, step=5)
    else:
        epochs = trial.suggest_int("epochs", 100, 500, step=50)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        num_teachers = trial.suggest_int("num_teachers", 5, 50, step=5)

    generator_lr = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
    discriminator_lr = trial.suggest_float("discriminator_lr", 1e-5, 1e-3, log=True)
    noise_multiplier = trial.suggest_float("noise_multiplier", 0.5, 2.0)
    target_epsilon = trial.suggest_float("target_epsilon", 0.1, 10.0, log=True)
    lap_scale = trial.suggest_float("lap_scale", 0.01, 1.0, log=True)

    generator_dim = trial.suggest_categorical(
        "generator_dim", [(128, 128), (256, 256), (256, 128)]
    )
    discriminator_dim = trial.suggest_categorical(
        "discriminator_dim", [(128, 128), (256, 256), (256, 128)]
    )

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "generator_lr": generator_lr,
        "discriminator_lr": discriminator_lr,
        "generator_dim": generator_dim,
        "discriminator_dim": discriminator_dim,
        "num_teachers": num_teachers,
        "noise_multiplier": noise_multiplier,
        "target_epsilon": target_epsilon,
        "lap_scale": lap_scale
    }


def _get_medgan_search_space(trial: 'optuna.Trial', run_mode: str) -> Dict[str, Any]:
    """
    MEDGAN search space (Phase 5 - January 2026).
    """
    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", 100, 300, step=50)
        pretrain_epochs = trial.suggest_int("pretrain_epochs", 50, 100, step=25)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    else:
        epochs = trial.suggest_int("epochs", 100, 500, step=50)
        pretrain_epochs = trial.suggest_int("pretrain_epochs", 50, 200, step=25)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    latent_dim = trial.suggest_int("latent_dim", 64, 256, step=32)
    autoencoder_lr = trial.suggest_float("autoencoder_lr", 1e-4, 1e-2, log=True)
    generator_lr = trial.suggest_float("generator_lr", 1e-4, 1e-2, log=True)
    discriminator_lr = trial.suggest_float("discriminator_lr", 1e-4, 1e-2, log=True)
    l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)

    autoencoder_dim = trial.suggest_categorical(
        "autoencoder_dim", [(64, 64), (128, 128), (256, 128)]
    )
    generator_dim = trial.suggest_categorical(
        "generator_dim", [(64, 64), (128, 128), (256, 128)]
    )
    discriminator_dim = trial.suggest_categorical(
        "discriminator_dim", [(128, 64), (256, 128), (256, 256)]
    )

    return {
        "epochs": epochs,
        "pretrain_epochs": pretrain_epochs,
        "batch_size": batch_size,
        "latent_dim": latent_dim,
        "autoencoder_lr": autoencoder_lr,
        "generator_lr": generator_lr,
        "discriminator_lr": discriminator_lr,
        "l2_reg": l2_reg,
        "autoencoder_dim": autoencoder_dim,
        "generator_dim": generator_dim,
        "discriminator_dim": discriminator_dim
    }


def _get_tabddpm_search_space(trial: 'optuna.Trial', run_mode: str) -> Dict[str, Any]:
    """
    TabDDPM search space.

    Diffusion model search space with focus on training iterations,
    learning rates, and model architecture dimensions.
    """
    if run_mode == "debug":
        n_iter = trial.suggest_int("n_iter", 50, 100, step=10)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-5, log=True)
        num_timesteps = trial.suggest_categorical("num_timesteps", [500, 1000])
        dim_embed = trial.suggest_categorical("dim_embed", [64, 128])
        n_layers_hidden = trial.suggest_int("n_layers_hidden", 1, 2)
        n_units_hidden = trial.suggest_categorical("n_units_hidden", [64, 128])
    else:  # full mode
        n_iter = trial.suggest_int("n_iter", 50, 200, step=10)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        num_timesteps = trial.suggest_categorical("num_timesteps", [500, 1000, 1500])
        dim_embed = trial.suggest_categorical("dim_embed", [64, 128, 256])
        n_layers_hidden = trial.suggest_int("n_layers_hidden", 1, 4, step=1)
        n_units_hidden = trial.suggest_categorical("n_units_hidden", [64, 128, 256, 512])

    return {
        "n_iter": n_iter,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "num_timesteps": num_timesteps,
        "dim_embed": dim_embed,
        "n_layers_hidden": n_layers_hidden,
        "n_units_hidden": n_units_hidden,
    }


def _get_great_search_space(trial: 'optuna.Trial', run_mode: str) -> Dict[str, Any]:
    """
    GReaT search space.

    LLM-based tabular generator with tunable LLM choice, batch size, epochs,
    learning rate, and warmup steps.
    """
    if run_mode == "debug":
        llm = trial.suggest_categorical("llm", ["distilgpt2"])
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        epochs = trial.suggest_int("epochs", 10, 20, step=5)
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        warmup_steps = trial.suggest_categorical("warmup_steps", [0, 50])
    else:  # full mode
        llm = trial.suggest_categorical("llm", ["distilgpt2", "gpt2"])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        epochs = trial.suggest_int("epochs", 10, 50, step=5)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        warmup_steps = trial.suggest_int("warmup_steps", 0, 500, step=50)

    return {
        "llm": llm,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "warmup_steps": warmup_steps,
    }


# Registry of supported models
SUPPORTED_MODELS = [
    "ctgan",
    "ctabgan",
    "ctabganplus",
    "ganeraid",
    "copulagan",
    "tvae",
    "pategan",
    "medgan",
    "tabddpm",     # Phase 5 - April 2026
    "great"        # Phase 5 - April 2026
]


def list_supported_models() -> List[str]:
    """Return list of models with defined search spaces."""
    return SUPPORTED_MODELS.copy()


print("[OK] Search spaces module loaded from src/models/search_spaces.py")
