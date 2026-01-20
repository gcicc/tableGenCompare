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
    from .ctabgan_model import CTABGANModel
except ImportError:
    pass

try:
    from .ctabganplus_model import CTABGANPlusModel
except ImportError:
    pass

# New models (Phase 5 - January 2026)
try:
    from .pategan_model import PATEGANModel
except ImportError:
    pass

try:
    from .medgan_model import MEDGANModel
except ImportError:
    pass

# TableGAN removed