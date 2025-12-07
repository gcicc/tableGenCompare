"""
Model Imports and Availability Checks

This module handles imports for all synthetic data generation models
with robust fallback and compatibility checking.

Code migrated from setup.py CHUNK_001, CHUNK_001B, CHUNK_001C
"""

import sys
import warnings

# ============================================================================
# CHUNK_001: CTAB-GAN Import and Compatibility
# ============================================================================

# First, apply sklearn compatibility patch
try:
    import sklearn
    print(f"Detected sklearn {sklearn.__version__} - applying compatibility patch...")

    from sklearn.mixture import GaussianMixture
    if not hasattr(GaussianMixture, 'n_components'):
        print(f"INFO: Applying sklearn compatibility patches for version {sklearn.__version__}")

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print("Global sklearn compatibility patch applied successfully")

except Exception as e:
    print(f"Warning: Could not apply sklearn compatibility patch: {e}")

# Now try to import CTAB-GAN
try:
    # Try importing from various possible locations
    import_successful = False

    try:
        from model.ctabgan import CTABGANSynthesizer
        print("CTAB-GAN imported successfully")
        import_successful = True
    except ImportError:
        pass

    if not import_successful:
        try:
            sys.path.append('./CTAB-GAN')
            from model.ctabgan import CTABGANSynthesizer
            print("CTAB-GAN imported successfully from ./CTAB-GAN")
            import_successful = True
        except ImportError:
            pass

    if not import_successful:
        try:
            sys.path.append('.')
            from model.ctabgan import CTABGANSynthesizer
            print("CTAB-GAN imported successfully from current directory")
            import_successful = True
        except ImportError:
            pass

    if not import_successful:
        print("WARNING: Could not import CTAB-GAN. Please ensure it's properly installed.")
        # Dummy class fallback
        class CTABGANSynthesizer:
            def __init__(self, *args, **kwargs):
                raise ImportError("CTAB-GAN not available")
        CTABGAN_AVAILABLE = False
    else:
        CTABGAN_AVAILABLE = True

except Exception as e:
    print(f"ERROR importing CTAB-GAN: {e}")
    # Create a dummy class to prevent import errors
    class CTABGANSynthesizer:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CTAB-GAN import failed: {e}")
    CTABGAN_AVAILABLE = False

# ============================================================================
# CHUNK_001B: CTAB-GAN+ Availability Check
# ============================================================================

try:
    # Try to import CTAB-GAN+ - it's in CTAB-GAN-Plus directory with model/ctabgan.py
    sys.path.append('./CTAB-GAN-Plus')
    from model.ctabgan import CTABGAN
    CTABGANPLUS_AVAILABLE = True
    print("[OK] CTAB-GAN+ detected and available")
except ImportError:
    CTABGANPLUS_AVAILABLE = False
    print("[WARNING] CTAB-GAN+ not available - falling back to regular CTAB-GAN")

# ============================================================================
# CHUNK_001C: GANerAid Import and Availability Check
# ============================================================================

try:
    # Try to import GANerAid from various possible locations
    ganeraid_import_successful = False

    # Method 1: Try from src.models.implementations
    try:
        from src.models.implementations.ganeraid_model import GANerAidModel
        print("[OK] GANerAidModel imported successfully from src.models.implementations")
        ganeraid_import_successful = True
    except ImportError:
        pass

    # Method 2: Try direct import (if available in path)
    if not ganeraid_import_successful:
        try:
            from ganeraid_model import GANerAidModel
            print("[OK] GANerAidModel imported successfully (direct import)")
            ganeraid_import_successful = True
        except ImportError:
            pass

    if not ganeraid_import_successful:
        try:
            from GANerAid import GANerAid
            print("[OK] GANerAidModel imported successfully (direct import)")
            ganeraid_import_successful = True
        except ImportError:
            pass

    if not ganeraid_import_successful:
        print("[WARNING] GANerAidModel not available - creating placeholder")
        # Dummy class fallback
        class GANerAidModel:
            def __init__(self, *args, **kwargs):
                raise ImportError("GANerAid not available. Please install it using: pip install GANerAid")
        GANERAID_AVAILABLE = False
    else:
        GANERAID_AVAILABLE = True

except Exception as e:
    print(f"[ERROR] Importing GANerAid failed: {e}")
    # Create a dummy class to prevent import errors
    class GANerAidModel:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"GANerAid import failed: {e}")
    GANERAID_AVAILABLE = False
