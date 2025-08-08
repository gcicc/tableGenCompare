"""
Factory for creating synthetic data model instances.

This module provides a centralized way to instantiate different synthetic data
models with consistent configuration and error handling.
"""

from typing import Dict, Any, Optional, Type
import logging
from .base_model import SyntheticDataModel

# Import model implementations
try:
    from .implementations.ganeraid_model import GANerAidModel
    GANERAID_AVAILABLE = True
except ImportError:
    GANERAID_AVAILABLE = False

try:
    from .implementations.ctgan_model import CTGANModel
    CTGAN_AVAILABLE = True
except ImportError:
    CTGAN_AVAILABLE = False

try:
    from .implementations.tvae_model import TVAEModel
    TVAE_AVAILABLE = True
except ImportError:
    TVAE_AVAILABLE = False

try:
    from .implementations.copulagan_model import CopulaGANModel
    COPULAGAN_AVAILABLE = True
except ImportError:
    COPULAGAN_AVAILABLE = False

try:
    from .implementations.ctabgan_model import CTABGANModel
    CTABGAN_AVAILABLE = True
except ImportError:
    CTABGAN_AVAILABLE = False

try:
    from .implementations.ctabganplus_model import CTABGANPlusModel
    CTABGANPLUS_AVAILABLE = True
except ImportError:
    CTABGANPLUS_AVAILABLE = False

# TableGAN removed
TABLEGAN_AVAILABLE = False


logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating synthetic data model instances.
    
    Provides a unified interface for instantiating different synthetic data
    models with proper configuration and dependency checking.
    """
    
    # Registry of available models
    _model_registry: Dict[str, Type[SyntheticDataModel]] = {}
    _model_availability: Dict[str, bool] = {}
    
    @classmethod
    def register_models(cls):
        """Register all available model implementations."""
        if GANERAID_AVAILABLE:
            cls._model_registry["ganeraid"] = GANerAidModel
            cls._model_availability["ganeraid"] = True
        else:
            cls._model_availability["ganeraid"] = False
            
        if CTGAN_AVAILABLE:
            cls._model_registry["ctgan"] = CTGANModel
            cls._model_availability["ctgan"] = True
        else:
            cls._model_availability["ctgan"] = False
            
        if TVAE_AVAILABLE:
            cls._model_registry["tvae"] = TVAEModel
            cls._model_availability["tvae"] = True
        else:
            cls._model_availability["tvae"] = False
            
        if COPULAGAN_AVAILABLE:
            cls._model_registry["copulagan"] = CopulaGANModel
            cls._model_availability["copulagan"] = True
        else:
            cls._model_availability["copulagan"] = False
            
        if CTABGAN_AVAILABLE:
            cls._model_registry["ctabgan"] = CTABGANModel
            cls._model_availability["ctabgan"] = True
        else:
            cls._model_availability["ctabgan"] = False
            
        if CTABGANPLUS_AVAILABLE:
            cls._model_registry["ctabganplus"] = CTABGANPlusModel
            cls._model_availability["ctabganplus"] = True
        else:
            cls._model_availability["ctabganplus"] = False
            
        # TableGAN removed
        cls._model_availability["tablegan"] = False
    
    @classmethod
    def create(
        cls, 
        model_name: str, 
        device: str = "cpu",
        random_state: int = 42,
        **kwargs
    ) -> SyntheticDataModel:
        """
        Create a synthetic data model instance.
        
        Args:
            model_name: Name of the model ("ganeraid", "ctgan", "tvae", "copulagan", "ctabgan", "ctabganplus", etc.)
            device: Computing device ("cpu", "cuda", "mps")
            random_state: Random seed for reproducibility
            **kwargs: Additional model-specific configuration
            
        Returns:
            Configured model instance
            
        Raises:
            ValueError: If model_name is not supported or not available
            ImportError: If required dependencies are not installed
        """
        # Register models if not already done
        if not cls._model_registry:
            cls.register_models()
        
        model_name = model_name.lower()
        
        # Check if model is supported
        if model_name not in cls._model_availability:
            available_models = [name for name, available in cls._model_availability.items() if available]
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Available models: {available_models}"
            )
        
        # Check if model dependencies are available
        if not cls._model_availability[model_name]:
            raise ImportError(
                f"Model '{model_name}' is not available. "
                f"Please install the required dependencies. "
                f"Run: pip install -e .[{model_name}]"
            )
        
        # Create model instance
        try:
            model_class = cls._model_registry[model_name]
            model = model_class(device=device, random_state=random_state)
            
            # Apply additional configuration
            if kwargs:
                model.set_config(kwargs)
            
            logger.info(f"Created {model_name} model on device {device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create {model_name} model: {e}")
            raise
    
    @classmethod
    def list_available_models(cls) -> Dict[str, bool]:
        """
        List all models and their availability status.
        
        Returns:
            Dictionary mapping model names to availability status
        """
        if not cls._model_registry:
            cls.register_models()
        
        return cls._model_availability.copy()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model information
        """
        if not cls._model_registry:
            cls.register_models()
        
        model_name = model_name.lower()
        
        if model_name not in cls._model_availability:
            raise ValueError(f"Unknown model: {model_name}")
        
        info = {
            "name": model_name,
            "available": cls._model_availability[model_name],
            "class": cls._model_registry.get(model_name).__name__ if cls._model_availability[model_name] else None
        }
        
        # Get hyperparameter space if model is available
        if cls._model_availability[model_name]:
            try:
                temp_instance = cls._model_registry[model_name]()
                info["hyperparameter_space"] = temp_instance.get_hyperparameter_space()
            except Exception as e:
                info["hyperparameter_space"] = f"Error retrieving: {e}"
        
        return info
    
    @classmethod
    def validate_model_config(cls, model_name: str, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for a specific model.
        
        Args:
            model_name: Name of the model
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        if not cls._model_registry:
            cls.register_models()
        
        model_name = model_name.lower()
        
        if not cls._model_availability.get(model_name, False):
            return False
        
        try:
            # Create temporary instance to validate config
            temp_instance = cls._model_registry[model_name]()
            temp_instance.set_config(config)
            return True
        except Exception as e:
            logger.warning(f"Invalid config for {model_name}: {e}")
            return False


# Initialize model registry on import
ModelFactory.register_models()