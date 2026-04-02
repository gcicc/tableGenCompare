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


def _compute_data_aware_bounds(data_size: int, n_cols: int) -> Dict[str, Any]:
    """
    Compute dataset-adaptive hyperparameter limits.

    Dynamically sets bounds for batch size, epochs, and network dimensions
    based on dataset size (rows) and dimensionality (columns).

    Parameters
    ----------
    data_size : int
        Number of rows in the dataset
    n_cols : int
        Number of columns (features) in the dataset

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - batch_choices: list of valid batch sizes (all <= data_size // 10)
        - max_epochs: upper epoch bound (inversely scales with data_size)
        - min_epochs: lower epoch bound (always 100)
        - dim_floor: minimum network width (power-of-2 >= n_cols, capped at 256)
        - dim_options: list of tuples for generator/discriminator dimensions
        - embedding_cap: max embedding_dim for VAE models
        - latent_cap: max latent_dim for autoencoders
        - pretrain_ratio: fraction for MEDGAN pretrain_epochs vs. epochs
        - max_teachers: max num_teachers for PATEGAN (prevents teacher starvation)
    """
    # Batch size: standard choices <= data_size // 10, minimum 32
    base_choices = [32, 64, 128, 256, 512]
    threshold = max(32, data_size // 10)
    batch_choices = [b for b in base_choices if b <= threshold]
    if not batch_choices:
        batch_choices = [32]

    # Epochs: inversely scale with data size
    if data_size < 500:
        max_epochs = 300
    elif data_size < 1000:
        max_epochs = 500
    elif data_size < 3000:
        max_epochs = 700
    else:
        max_epochs = 1000
    min_epochs = 100

    # Network dims: floor is nearest power-of-2 >= n_cols, capped at 256
    dim_floor = min(256, max(32, 1 << (n_cols - 1).bit_length()))

    # Build graduated dim list anchored to dim_floor
    d = dim_floor
    if d <= 64:
        dim_options = [
            (d, d),
            (d * 2, d * 2),
            (d * 2, d),
            (d, d * 2, d),
            (d * 2, d * 4, d * 2),
        ]
    elif d <= 128:
        dim_options = [
            (128, 128),
            (256, 256),
            (256, 128),
            (128, 256, 128),
            (256, 512, 256),
        ]
    else:  # d == 256 (n_cols >= 129)
        dim_options = [
            (256, 256),
            (512, 256),
            (256, 512),
            (256, 512, 256),
        ]

    # VAE / autoencoder dimension caps
    embedding_cap = min(256, max(32, dim_floor * 2))
    latent_cap = min(256, max(64, dim_floor))

    # MEDGAN pretrain ratio
    pretrain_ratio = 0.3

    # PATEGAN: max_teachers ensures ~30 rows per teacher (prevents starvation)
    max_teachers = max(5, min(50, data_size // 30))

    return {
        "batch_choices": batch_choices,
        "max_epochs": max_epochs,
        "min_epochs": min_epochs,
        "dim_floor": dim_floor,
        "dim_options": dim_options,
        "embedding_cap": embedding_cap,
        "latent_cap": latent_cap,
        "pretrain_ratio": pretrain_ratio,
        "max_teachers": max_teachers,
    }


def get_search_space(
    model_name: str,
    trial: 'optuna.Trial',
    run_mode: str = "full",
    data_size: int = None,
    n_cols: int = None
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
        Dataset size (number of rows) for data-aware bounds
    n_cols : int, optional
        Number of columns (features) for network dimension scaling

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
    ...     params = get_search_space("ctgan", trial, "debug", data_size=1000, n_cols=15)
    ...     # Use params to configure and train model
    ...     return score
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for hyperparameter search spaces")

    model_name = model_name.lower()

    if model_name == "ctgan":
        return _get_ctgan_search_space(trial, run_mode, data_size, n_cols)
    elif model_name == "ctabgan":
        return _get_ctabgan_search_space(trial, run_mode, data_size, n_cols)
    elif model_name == "ctabganplus":
        return _get_ctabganplus_search_space(trial, run_mode, data_size, n_cols)
    elif model_name == "ganeraid":
        return _get_ganeraid_search_space(trial, run_mode, data_size, n_cols)
    elif model_name == "copulagan":
        return _get_copulagan_search_space(trial, run_mode, data_size, n_cols)
    elif model_name == "tvae":
        return _get_tvae_search_space(trial, run_mode, data_size, n_cols)
    elif model_name == "pategan":
        return _get_pategan_search_space(trial, run_mode, data_size, n_cols)
    elif model_name == "medgan":
        return _get_medgan_search_space(trial, run_mode, data_size, n_cols)
    elif model_name == "tabdiffusion":
        return _get_tabdiffusion_search_space(trial, run_mode, data_size, n_cols)
    elif model_name == "great":
        return _get_great_search_space(trial, run_mode, data_size, n_cols)
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


def _get_ctgan_search_space(
    trial: 'optuna.Trial', run_mode: str, data_size: int = None, n_cols: int = None
) -> Dict[str, Any]:
    """
    CTGAN search space from Cell 54.

    Constraint: batch_size % pac == 0
    """
    data_size = data_size or 1000
    n_cols = n_cols or 10
    bounds = _compute_data_aware_bounds(data_size, n_cols)

    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", 100, min(300, bounds["max_epochs"]), step=50)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])
        pac = trial.suggest_categorical("pac", [1, 2, 4, 8])
        generator_lr = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
        discriminator_lr = trial.suggest_float("discriminator_lr", 1e-5, 1e-3, log=True)
        discriminator_steps = trial.suggest_int("discriminator_steps", 1, 3)
        generator_decay = trial.suggest_float("generator_decay", 1e-8, 1e-4, log=True)
        discriminator_decay = trial.suggest_float("discriminator_decay", 1e-8, 1e-4, log=True)
        generator_dim = trial.suggest_categorical("generator_dim", bounds["dim_options"][:3])
        discriminator_dim = trial.suggest_categorical("discriminator_dim", bounds["dim_options"][:3])
    else:  # full mode
        epochs = trial.suggest_int("epochs", 100, bounds["max_epochs"], step=50)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])
        pac = trial.suggest_categorical("pac", [1, 2, 4, 8, 10])
        generator_lr = trial.suggest_float("generator_lr", 5e-6, 5e-3, log=True)
        discriminator_lr = trial.suggest_float("discriminator_lr", 5e-6, 5e-3, log=True)
        discriminator_steps = trial.suggest_int("discriminator_steps", 1, 5)
        generator_decay = trial.suggest_float("generator_decay", 1e-8, 1e-5, log=True)
        discriminator_decay = trial.suggest_float("discriminator_decay", 1e-8, 1e-5, log=True)
        generator_dim = trial.suggest_categorical("generator_dim", bounds["dim_options"])
        discriminator_dim = trial.suggest_categorical("discriminator_dim", bounds["dim_options"])

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


def _get_ctabgan_search_space(
    trial: 'optuna.Trial', run_mode: str, data_size: int = None, n_cols: int = None
) -> Dict[str, Any]:
    """
    CTAB-GAN search space from Cell 56.

    Note: test_ratio is fixed at 0.2 (not a synthesis-quality hyperparameter)
    """
    data_size = data_size or 1000
    n_cols = n_cols or 10
    bounds = _compute_data_aware_bounds(data_size, n_cols)

    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", bounds["min_epochs"], min(300, bounds["max_epochs"]), step=50)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])
    else:  # full mode
        epochs = trial.suggest_int("epochs", bounds["min_epochs"], bounds["max_epochs"], step=50)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "test_ratio": 0.2  # Fixed, not tuned
    }


def _get_ctabganplus_search_space(
    trial: 'optuna.Trial', run_mode: str, data_size: int = None, n_cols: int = None
) -> Dict[str, Any]:
    """
    CTAB-GAN+ search space from Cell 58.

    Note: test_ratio is fixed at 0.2
    """
    data_size = data_size or 1000
    n_cols = n_cols or 10
    bounds = _compute_data_aware_bounds(data_size, n_cols)

    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", bounds["min_epochs"], min(300, bounds["max_epochs"]), step=50)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])
    else:  # full mode
        epochs = trial.suggest_int("epochs", bounds["min_epochs"], bounds["max_epochs"], step=50)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "test_ratio": 0.2  # Fixed, not tuned
    }


def _get_ganeraid_search_space(
    trial: 'optuna.Trial',
    run_mode: str,
    data_size: int = None,
    n_cols: int = None
) -> Dict[str, Any]:
    """
    GANerAid search space from Cell 60.

    Uses precomputed feasible triples for (nr_of_rows, batch_size, hidden_feature_space)
    with divisibility constraints. Data-aware to prevent batch sizes > data_size // 10.
    """
    data_size = data_size or 1000
    n_cols = n_cols or 10
    bounds = _compute_data_aware_bounds(data_size, n_cols)

    # Compute feasible triples
    feasible_triples = _compute_ganeraid_feasible_triples(run_mode, data_size, bounds["batch_choices"])

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
    data_size: int,
    batch_choices: List[int] = None
) -> List[Tuple[int, int, int]]:
    """
    Compute feasible (nr_of_rows, batch_size, hidden_feature_space) triples for GANerAid.

    Constraints:
    - batch_size % nr_of_rows == 0
    - hidden_feature_space % nr_of_rows == 0
    - nr_of_rows >= 4 and nr_of_rows < data_size
    - batch_size respects data_size // 10 constraint if batch_choices provided

    Parameters
    ----------
    run_mode : str
        "debug" or "full"
    data_size : int
        Number of rows in dataset
    batch_choices : List[int], optional
        Pre-filtered batch sizes from _compute_data_aware_bounds
    """
    if run_mode == "debug":
        row_candidates = [4, 5, 8, 10, 16, 20, 25]
        if batch_choices is None:
            batch_choices = [64, 100, 128, 200, 256]
        hidden_choices = [100, 150, 200, 300, 400]
    else:
        row_candidates = [4, 5, 8, 10, 16, 20, 25, 32, 40]
        if batch_choices is None:
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


def _get_copulagan_search_space(
    trial: 'optuna.Trial', run_mode: str, data_size: int = None, n_cols: int = None
) -> Dict[str, Any]:
    """
    CopulaGAN search space from Cell 62.

    Data-aware batch sizes and network dimensions to prevent overfitting.
    """
    data_size = data_size or 1000
    n_cols = n_cols or 10
    bounds = _compute_data_aware_bounds(data_size, n_cols)

    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", bounds["min_epochs"], min(300, bounds["max_epochs"]), step=50)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])
        generator_lr = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
        discriminator_lr = trial.suggest_float("discriminator_lr", 1e-5, 1e-3, log=True)
    else:  # full mode
        epochs = trial.suggest_int("epochs", bounds["min_epochs"], bounds["max_epochs"], step=50)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])
        generator_lr = trial.suggest_float("generator_lr", 5e-6, 5e-3, log=True)
        discriminator_lr = trial.suggest_float("discriminator_lr", 5e-6, 5e-3, log=True)

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "generator_lr": generator_lr,
        "discriminator_lr": discriminator_lr,
    }


def _get_tvae_search_space(
    trial: 'optuna.Trial', run_mode: str, data_size: int = None, n_cols: int = None
) -> Dict[str, Any]:
    """
    TVAE search space from Cell 64.

    Data-aware batch sizes, epochs, and embedding dimensions.
    Note: dropout parameter is sampled but SDV's TVAESynthesizer doesn't accept it;
    it's included for consistency with VAE best practices.
    """
    data_size = data_size or 1000
    n_cols = n_cols or 10
    bounds = _compute_data_aware_bounds(data_size, n_cols)

    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", bounds["min_epochs"], min(300, bounds["max_epochs"]), step=50)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        embedding_dim = trial.suggest_int("embedding_dim", 32, min(128, bounds["embedding_cap"]), step=32)
        l2scale = trial.suggest_float("l2scale", 1e-6, 1e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.2)
    else:  # full mode
        epochs = trial.suggest_int("epochs", bounds["min_epochs"], bounds["max_epochs"], step=50)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        embedding_dim = trial.suggest_int("embedding_dim", 32, bounds["embedding_cap"], step=32)
        l2scale = trial.suggest_float("l2scale", 1e-6, 1e-4, log=True)  # Reduced high end from 1e-2
        dropout = trial.suggest_float("dropout", 0.0, 0.3)

    compress_dims = trial.suggest_categorical("compress_dims", bounds["dim_options"][:3])
    decompress_dims = trial.suggest_categorical("decompress_dims", bounds["dim_options"][:3])

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


def _get_pategan_search_space(
    trial: 'optuna.Trial', run_mode: str, data_size: int = None, n_cols: int = None
) -> Dict[str, Any]:
    """
    PATE-GAN search space (Phase 5 - January 2026, updated April 2026).

    Data-aware batch sizes, epochs, num_teachers, and network dimensions.
    Added weight decay parameters for regularization.
    """
    data_size = data_size or 1000
    n_cols = n_cols or 10
    bounds = _compute_data_aware_bounds(data_size, n_cols)

    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", bounds["min_epochs"], min(300, bounds["max_epochs"]), step=50)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])
        num_teachers = trial.suggest_int("num_teachers", 5, min(20, bounds["max_teachers"]), step=5)
    else:
        epochs = trial.suggest_int("epochs", bounds["min_epochs"], bounds["max_epochs"], step=50)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])
        num_teachers = trial.suggest_int("num_teachers", 5, bounds["max_teachers"], step=5)

    generator_lr = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
    discriminator_lr = trial.suggest_float("discriminator_lr", 1e-5, 1e-3, log=True)
    generator_decay = trial.suggest_float("generator_decay", 1e-8, 1e-5, log=True)  # NEW: weight decay
    discriminator_decay = trial.suggest_float("discriminator_decay", 1e-8, 1e-5, log=True)  # NEW: weight decay
    noise_multiplier = trial.suggest_float("noise_multiplier", 0.5, 2.0)
    target_epsilon = trial.suggest_float("target_epsilon", 0.1, 10.0, log=True)
    lap_scale = trial.suggest_float("lap_scale", 0.01, 1.0, log=True)

    generator_dim = trial.suggest_categorical("generator_dim", bounds["dim_options"][:3])
    discriminator_dim = trial.suggest_categorical("discriminator_dim", bounds["dim_options"][:3])

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "generator_lr": generator_lr,
        "discriminator_lr": discriminator_lr,
        "generator_decay": generator_decay,
        "discriminator_decay": discriminator_decay,
        "generator_dim": generator_dim,
        "discriminator_dim": discriminator_dim,
        "num_teachers": num_teachers,
        "noise_multiplier": noise_multiplier,
        "target_epsilon": target_epsilon,
        "lap_scale": lap_scale
    }


def _get_medgan_search_space(
    trial: 'optuna.Trial', run_mode: str, data_size: int = None, n_cols: int = None
) -> Dict[str, Any]:
    """
    MEDGAN search space (Phase 5 - January 2026, updated April 2026).

    Data-aware batch sizes, epochs, latent dims, and network dimensions.
    Added dropout parameter for generator regularization.
    """
    data_size = data_size or 1000
    n_cols = n_cols or 10
    bounds = _compute_data_aware_bounds(data_size, n_cols)

    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", bounds["min_epochs"], min(300, bounds["max_epochs"]), step=50)
        max_pretrain = max(50, int(bounds["max_epochs"] * bounds["pretrain_ratio"]))
        pretrain_epochs = trial.suggest_int("pretrain_epochs", 25, max_pretrain, step=25)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])
    else:
        epochs = trial.suggest_int("epochs", bounds["min_epochs"], bounds["max_epochs"], step=50)
        max_pretrain = max(50, int(bounds["max_epochs"] * bounds["pretrain_ratio"]))
        pretrain_epochs = trial.suggest_int("pretrain_epochs", 25, max_pretrain, step=25)
        batch_size = trial.suggest_categorical("batch_size", bounds["batch_choices"])

    latent_dim = trial.suggest_int("latent_dim", 64, bounds["latent_cap"], step=32)
    autoencoder_lr = trial.suggest_float("autoencoder_lr", 1e-4, 1e-2, log=True)
    generator_lr = trial.suggest_float("generator_lr", 1e-4, 1e-2, log=True)
    discriminator_lr = trial.suggest_float("discriminator_lr", 1e-4, 1e-2, log=True)
    l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-3, log=True)  # Tightened high end from 1e-2
    dropout = trial.suggest_float("dropout", 0.0, 0.3)  # NEW: Generator dropout

    autoencoder_dim = trial.suggest_categorical("autoencoder_dim", bounds["dim_options"][:3])
    generator_dim = trial.suggest_categorical("generator_dim", bounds["dim_options"][:3])
    discriminator_dim = trial.suggest_categorical("discriminator_dim", bounds["dim_options"][:3])

    return {
        "epochs": epochs,
        "pretrain_epochs": pretrain_epochs,
        "batch_size": batch_size,
        "latent_dim": latent_dim,
        "autoencoder_lr": autoencoder_lr,
        "generator_lr": generator_lr,
        "discriminator_lr": discriminator_lr,
        "l2_reg": l2_reg,
        "dropout": dropout,
        "autoencoder_dim": autoencoder_dim,
        "generator_dim": generator_dim,
        "discriminator_dim": discriminator_dim
    }


def _get_tabdiffusion_search_space(
    trial: 'optuna.Trial', run_mode: str, data_size: int = None, n_cols: int = None
) -> Dict[str, Any]:
    """
    TabDiffusion search space.

    Diffusion model search space with focus on training epochs, learning rates,
    diffusion steps, and denoising network architecture.

    Updated Phase 5 (April 2026): Optimized for small-to-medium datasets (500-5000 rows)
    - Removed batch_size=256 (overfitting risk for <1000 rows)
    - Added dropout and weight_decay regularization (critical for small data)
    - Reduced hidden_dim max to 128 (256 causes overfitting)
    - Reduced num_layers max to 4 (5 layers too deep)
    - Narrowed learning_rate to [1e-4, 5e-3] (avoid diffusion instability)
    """
    if run_mode == "debug":
        epochs = trial.suggest_int("epochs", 50, 100, step=10)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        num_diffusion_steps = trial.suggest_categorical("num_diffusion_steps", [500, 1000])
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128])
        num_layers = trial.suggest_int("num_layers", 2, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.2)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)
    else:  # full mode - optimized for 500-5000 row datasets
        epochs = trial.suggest_int("epochs", 50, 150, step=10)  # Reduced from 200 (was overfitting)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])  # Removed 256 (unstable for small data)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)  # Narrowed from 1e-2 (diffusion stability)
        num_diffusion_steps = trial.suggest_categorical("num_diffusion_steps", [500, 1000, 1500])
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128])  # Removed 256 (overfitting with <1000 rows)
        num_layers = trial.suggest_int("num_layers", 2, 4, step=1)  # Reduced from 5 (too deep for small data)
        dropout = trial.suggest_float("dropout", 0.1, 0.3)  # NEW: Critical regularization
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)  # NEW: L2 regularization

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_diffusion_steps": num_diffusion_steps,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "weight_decay": weight_decay,
    }


def _get_great_search_space(
    trial: 'optuna.Trial', run_mode: str, data_size: int = None, n_cols: int = None
) -> Dict[str, Any]:
    """
    GReaT search space.

    LLM-based tabular generator with tunable LLM choice, batch size, epochs,
    learning rate, and warmup steps.

    Updated Phase 5 (April 2026): Optimized for small-to-medium datasets (500-5000 rows)
    - Made LLM choice data-aware: distilgpt2 for <2000 rows, both options for ≥2000 rows
    - Removed batch_size=16 (too small for LLM training)
    - Reduced epochs max to 40 (LLMs need fewer epochs)
    - Made warmup_steps data-aware: 5-10% of total training steps
    - Added dropout and weight_decay regularization when using gpt2
    """
    if data_size is None:
        data_size = 2000  # Default assumption

    if run_mode == "debug":
        # Conservative for small data
        llm = trial.suggest_categorical("llm", ["distilgpt2"])
        batch_size = trial.suggest_categorical("batch_size", [32])
        epochs = trial.suggest_int("epochs", 10, 20, step=5)
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        warmup_steps = trial.suggest_categorical("warmup_steps", [0, 50, 100])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)
    else:  # full mode - data-aware
        # LLM choice depends on dataset size
        if data_size < 2000:
            # Use lightweight model for small datasets
            llm = "distilgpt2"
        else:
            # Allow both models for larger datasets
            llm = trial.suggest_categorical("llm", ["distilgpt2", "gpt2"])

        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])  # Removed 16 (too small)
        epochs = trial.suggest_int("epochs", 10, 40, step=5)  # Reduced from 50 (LLMs need fewer epochs)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

        # Calculate data-aware warmup_steps (5-10% of total training steps)
        # Total steps ≈ (data_size / batch_size) * epochs
        # For typical: (2000 / 64) * 30 ≈ 937 steps → warmup = 47-94 steps
        max_warmup = min(300, max(50, data_size // 20))  # 5% of dataset, capped at 300
        warmup_steps = trial.suggest_int("warmup_steps", 0, max_warmup, step=25)

        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)  # NEW: L2 regularization

        # Add dropout if using larger model
        if isinstance(llm, str) and llm == "gpt2":
            dropout = trial.suggest_float("dropout", 0.1, 0.3)  # NEW: For gpt2 only
        else:
            dropout = 0.0  # distilgpt2 doesn't need as much dropout

    return {
        "llm": llm,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "dropout": dropout,
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
    "tabdiffusion", # Phase 5 - April 2026
    "great"         # Phase 5 - April 2026
]


def list_supported_models() -> List[str]:
    """Return list of models with defined search spaces."""
    return SUPPORTED_MODELS.copy()


print("[OK] Search spaces module loaded from src/models/search_spaces.py")
