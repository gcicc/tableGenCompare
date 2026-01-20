"""
Model Registry for Synthetic Data Generation Framework

This module provides a centralized registry of available models and helper
functions for model selection based on notebook configuration.

Phase 4 - January 2026
"""

from typing import Dict, List, Any, Optional, Union
import warnings
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# AVAILABLE MODELS REGISTRY
# ============================================================================

# Canonical model names and their display names
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "ctgan": {
        "display_name": "CTGAN",
        "description": "Conditional Tabular GAN - state-of-the-art for tabular data",
        "module": "src.models.implementations.ctgan_model",
        "class_name": "CTGANModel",
        "requires": ["ctgan"],
    },
    "tvae": {
        "display_name": "TVAE",
        "description": "Tabular Variational Autoencoder - VAE-based approach",
        "module": "src.models.implementations.tvae_model",
        "class_name": "TVAEModel",
        "requires": ["sdv"],
    },
    "ctabgan": {
        "display_name": "CTABGAN",
        "description": "Conditional Tabular GAN with advanced preprocessing",
        "module": "src.models.implementations.ctabgan_model",
        "class_name": "CTABGANModel",
        "requires": ["torch"],
    },
    "ctabganplus": {
        "display_name": "CTABGAN+",
        "description": "Enhanced CTABGAN with WGAN-GP losses",
        "module": "src.models.implementations.ctabganplus_model",
        "class_name": "CTABGANPlusModel",
        "requires": ["torch"],
    },
    "copulagan": {
        "display_name": "CopulaGAN",
        "description": "Copula-based GAN using statistical copula functions",
        "module": "src.models.implementations.copulagan_model",
        "class_name": "CopulaGANModel",
        "requires": ["sdv"],
    },
    "ganeraid": {
        "display_name": "GANerAid",
        "description": "Purpose-built clinical data generator",
        "module": "src.models.implementations.ganeraid_model",
        "class_name": "GANerAidModel",
        "requires": ["ganeraid"],
    },
    # New models (Phase 5 - January 2026)
    "pategan": {
        "display_name": "PATE-GAN",
        "description": "Differentially private GAN using PATE framework",
        "module": "src.models.implementations.pategan_model",
        "class_name": "PATEGANModel",
        "requires": ["torch"],
    },
    "medgan": {
        "display_name": "MEDGAN",
        "description": "Medical GAN for discrete/binary health records",
        "module": "src.models.implementations.medgan_model",
        "class_name": "MEDGANModel",
        "requires": ["torch"],
    },
}

# Aliases for common variations
MODEL_ALIASES: Dict[str, str] = {
    "ctab-gan": "ctabgan",
    "ctab_gan": "ctabgan",
    "ctabgan+": "ctabganplus",
    "ctab-gan+": "ctabganplus",
    "ctab_gan_plus": "ctabganplus",
    "ctabgan-plus": "ctabganplus",
    "copula-gan": "copulagan",
    "copula_gan": "copulagan",
    "ganer-aid": "ganeraid",
    "ganer_aid": "ganeraid",
    # New model aliases (Phase 5)
    "pate-gan": "pategan",
    "pate_gan": "pategan",
    "med-gan": "medgan",
    "med_gan": "medgan",
}


def _normalize_model_name(name: str) -> str:
    """Normalize a model name to its canonical form."""
    normalized = name.lower().strip()
    return MODEL_ALIASES.get(normalized, normalized)


def get_available_model_names() -> List[str]:
    """
    Get list of all available model names.

    Returns:
        List of canonical model names
    """
    return list(AVAILABLE_MODELS.keys())


def is_model_available(model_name: str) -> bool:
    """
    Check if a model is registered and its dependencies are available.

    Args:
        model_name: Model name to check

    Returns:
        True if model is available for use
    """
    from .model_factory import ModelFactory

    normalized = _normalize_model_name(model_name)
    if normalized not in AVAILABLE_MODELS:
        return False

    # Check via ModelFactory which handles dependency checking
    availability = ModelFactory.list_available_models()
    return availability.get(normalized, False)


def resolve_models(models_to_run: Union[str, List[str]]) -> List[str]:
    """
    Resolve model selection to a list of canonical model names.

    Args:
        models_to_run: Either "all" or a list of model names

    Returns:
        List of canonical model names that are available

    Raises:
        ValueError: If no valid models are found
    """
    from .model_factory import ModelFactory

    # Get currently available models
    availability = ModelFactory.list_available_models()
    available = [name for name, is_avail in availability.items() if is_avail]

    if models_to_run == "all":
        if not available:
            raise ValueError("No models are available. Check your dependencies.")
        logger.info(f"Resolved 'all' to {len(available)} available models: {available}")
        return available

    if isinstance(models_to_run, str):
        # Single model name
        models_to_run = [models_to_run]

    # Normalize and validate each model name
    resolved = []
    unknown = []
    unavailable = []

    for name in models_to_run:
        normalized = _normalize_model_name(name)

        if normalized not in AVAILABLE_MODELS:
            unknown.append(name)
            continue

        if normalized not in available:
            unavailable.append(name)
            continue

        if normalized not in resolved:
            resolved.append(normalized)

    # Warn about issues
    if unknown:
        warnings.warn(
            f"Unknown model names ignored: {unknown}. "
            f"Valid models: {list(AVAILABLE_MODELS.keys())}"
        )

    if unavailable:
        warnings.warn(
            f"Unavailable models ignored (missing dependencies): {unavailable}. "
            f"Install required packages or check imports."
        )

    if not resolved:
        raise ValueError(
            f"No valid models found. Requested: {models_to_run}, "
            f"Available: {available}"
        )

    logger.info(f"Resolved {len(resolved)} models: {resolved}")
    return resolved


def get_models_to_run(config: Dict[str, Any]) -> List[str]:
    """
    Get list of models to run based on notebook configuration.

    This is the main entry point for config-driven model selection.

    Args:
        config: Notebook configuration dictionary

    Returns:
        List of canonical model names to run
    """
    models_to_run = config.get("models_to_run", "all")
    return resolve_models(models_to_run)


def get_model_display_name(model_name: str) -> str:
    """
    Get the display name for a model.

    Args:
        model_name: Canonical or alias model name

    Returns:
        Human-readable display name
    """
    normalized = _normalize_model_name(model_name)
    if normalized in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[normalized]["display_name"]
    return model_name.upper()


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get registry information for a model.

    Args:
        model_name: Model name

    Returns:
        Model info dict or None if not found
    """
    normalized = _normalize_model_name(model_name)
    return AVAILABLE_MODELS.get(normalized)


def print_model_summary():
    """Print a summary of all registered models and their availability."""
    from .model_factory import ModelFactory

    availability = ModelFactory.list_available_models()

    print("\n" + "=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)
    print(f"{'Model':<15} {'Display':<12} {'Available':<10} {'Description'}")
    print("-" * 60)

    for name, info in AVAILABLE_MODELS.items():
        is_avail = availability.get(name, False)
        status = "Yes" if is_avail else "No"
        print(f"{name:<15} {info['display_name']:<12} {status:<10} {info['description'][:30]}...")

    print("=" * 60)
    available_count = sum(1 for v in availability.values() if v)
    print(f"Total: {len(AVAILABLE_MODELS)} registered, {available_count} available\n")


# ============================================================================
# TUNING CONFIGURATION HELPERS
# ============================================================================

def get_tuning_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract tuning configuration from notebook config.

    Args:
        config: Notebook configuration dictionary

    Returns:
        Dictionary with tuning parameters:
            - n_trials: Number of optimization trials
            - timeout_seconds: Timeout per study (or None)
            - tuning_mode: "smoke" or "full"
    """
    tuning_mode = config.get("tuning_mode", "smoke")

    if tuning_mode == "full":
        n_trials = config.get("n_trials_full", 50)
    else:
        n_trials = config.get("n_trials_smoke", 5)

    timeout_seconds = config.get("timeout_seconds", None)

    return {
        "n_trials": n_trials,
        "timeout_seconds": timeout_seconds,
        "tuning_mode": tuning_mode
    }


def get_n_trials(config: Dict[str, Any]) -> int:
    """
    Get number of trials based on tuning mode.

    Args:
        config: Notebook configuration dictionary

    Returns:
        Number of trials for optimization
    """
    return get_tuning_config(config)["n_trials"]


print("[OK] Model registry loaded from src/models/registry.py")
