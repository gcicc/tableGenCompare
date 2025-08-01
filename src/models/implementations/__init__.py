"""
Model implementations for synthetic tabular data generation.

This package contains concrete implementations of the SyntheticDataModel
interface for various synthetic data generation models.
"""

# Import all available model implementations
try:
    from .ganeraid_model import GANerAidModel
except ImportError:
    pass

try:
    from .ctgan_model import CTGANModel
except ImportError:
    pass

try:
    from .tvae_model import TVAEModel
except ImportError:
    pass

try:
    from .copulagan_model import CopulaGANModel
except ImportError:
    pass

try:
    from .tablegan_model import TableGANModel
except ImportError:
    pass