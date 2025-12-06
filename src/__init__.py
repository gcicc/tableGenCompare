"""
Clinical Synthetic Data Generation Framework

A comprehensive framework for benchmarking synthetic tabular data generation
models across multiple healthcare datasets with research-grade evaluation metrics.
"""

__version__ = "0.2.0"
__author__ = "Clinical Research Team"

# ============================================================================
# ESSENTIAL THIRD-PARTY IMPORTS
# Available globally when using 'from setup import *' or 'from src import *'
# ============================================================================

# Core data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# Core ML/preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Utility libraries
import warnings
warnings.filterwarnings('ignore')

print("[OK] Essential data science libraries imported successfully!")

# ============================================================================
# FRAMEWORK MODULE IMPORTS
# Core functionality re-exported for backward compatibility
# ============================================================================

# Attempt to import existing modules (may not all exist yet)
try:
    from .models import ModelFactory
except ImportError:
    ModelFactory = None

try:
    from .datasets import DatasetHandler
except ImportError:
    DatasetHandler = None

try:
    from .evaluation import UnifiedEvaluator
except ImportError:
    UnifiedEvaluator = None

# ExperimentRunner not yet implemented
ExperimentRunner = None

__all__ = [
    # External libraries (for 'from src import *')
    "pd", "np", "plt", "sns", "stats",
    "jensenshannon", "wasserstein_distance",
    "train_test_split", "StandardScaler", "LabelEncoder",
    "accuracy_score", "classification_report", "confusion_matrix",
    "RandomForestClassifier", "LogisticRegression", "PCA",

    # Framework modules
    "ModelFactory",
    "DatasetHandler",
    "UnifiedEvaluator",
    "ExperimentRunner",
]
