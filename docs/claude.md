# Clinical Synthetic Data Generation Framework - Implementation Plan

**Document Purpose**: Structured execution plan for implementing improvements from dev-plan.md
**Organization**: 18 tasks across 5 phases (0-4)
**Strategy**: Incremental refactoring - create modular architecture while delivering features
**Status**: Ready for execution

---

## OVERVIEW: INCREMENTAL REFACTOR STRATEGY

This plan implements **Option C: Incremental Refactor** - we'll create a clean modular architecture while delivering features incrementally, rather than big-bang refactoring or growing setup.py to 4000+ lines.

**Key Principles:**
- Create `src/` modular structure upfront (Phase 0)
- Move code to proper modules AS we enhance it (Phases 1-3)
- New code goes directly into correct modules from day 1
- setup.py becomes thin re-export layer for backward compatibility
- Notebooks continue using `from setup import *` - no changes needed

**Benefits:**
✅ Immediate value delivery (don't wait for full refactor)
✅ New code organized properly from start
✅ setup.py shrinks over time instead of growing
✅ Lower risk - only touch code being modified
✅ Progressive improvement with continuous testing

---

## 1. PROJECT CONTEXT & ARCHITECTURE

### 1.1 Project Overview

Clinical Synthetic Data Generation Framework benchmarking 6 generative models (CTGAN, CTAB-GAN, CTAB-GAN+, GANerAid, CopulaGAN, TVAE) across 4 healthcare datasets (Alzheimer's, Breast Cancer, Liver, Pakistani Liver).

### 1.2 Notebook Architecture

- **Main notebooks**: `SynthethicTableGenerator-{Dataset}.ipynb` (4 datasets)
- **V2 notebooks**: `STG-{Dataset}V2.ipynb` (3 datasets: BreastCancer, Liver, Pakistani)
- **Shared code**: Currently `setup.py` (3814 lines) → Will become modular `src/` structure

### 1.3 5-Section Pipeline Structure

| Section | Purpose | Key Outputs |
|---------|---------|-------------|
| 1 | Setup & Data Loading | Data loaded, identifiers set |
| 2 | EDA & Preprocessing | correlation_heatmap.png, feature_distributions.png, README.md |
| 3 | Model Training & Evaluation | Per-model evaluation files, README.md |
| 4 | Hyperparameter Optimization | Best parameters, optimization plots |
| 5 | Final Comparison | trts_comprehensive_analysis.png, privacy_metrics.csv |

### 1.4 Target Architecture (After Phase 0)

```
tableGenCompare/
├── setup.py                    # Thin re-export layer (backward compatible)
├── src/
│   ├── __init__.py
│   ├── config.py               # Global config, SESSION_TIMESTAMP
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # MICE, encoding, cleaning
│   │   └── loading.py          # Dataset loading utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── imports.py          # CHUNK_001, 001B, 001C (model imports)
│   │   └── wrappers.py         # CHUNK_002, 003 (CTABGANModel classes)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── quality.py          # evaluate_synthetic_data_quality()
│   │   ├── trts.py             # comprehensive_trts_analysis()
│   │   ├── privacy.py          # NEW: Privacy metrics
│   │   └── mode_collapse.py    # NEW: Mode collapse detection
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── section2.py         # NEW: correlation heatmap, feature dist
│   │   ├── section3.py         # correlation/distribution comparison
│   │   └── section5.py         # create_trts_visualizations()
│   │
│   ├── objective/
│   │   ├── __init__.py
│   │   └── functions.py        # enhanced_objective_function_v2()
│   │
│   └── utils/
│       ├── __init__.py
│       ├── paths.py            # get_results_path()
│       └── session.py          # SESSION_TIMESTAMP management
```

### 1.5 Current Code Locations (Pre-Refactor)

| Functionality | Current Location | Target Module |
|---------------|------------------|---------------|
| Correlation Heatmap | Notebooks (inline) | src/visualization/section2.py |
| Feature Distributions | Notebooks (inline) | src/visualization/section2.py |
| Section 3 Visualizations | setup.py:1127-1560 | src/visualization/section3.py |
| Section 5 TRTS Viz | setup.py:2986-3176 | src/visualization/section5.py |
| TRTS Analysis | setup.py:2768-2985 | src/evaluation/trts.py |
| Quality Evaluation | setup.py:1127-1560 | src/evaluation/quality.py |
| Objective Functions | setup.py:1584-1889 | src/objective/functions.py |

---

## 2. DEVELOPMENT PRINCIPLES & CONSTRAINTS

### 2.1 Code Consistency Requirements

- **ALL changes must apply to ALL 7 notebooks** (4 main + 3 V2)
- Notebooks continue using `from setup import *` (backward compatible)
- Test with small dataset (Alzheimer ~10 cols) AND large dataset (Pakistani ~15+ cols)

### 2.2 Backward Compatibility

- setup.py remains import point: `from setup import *` still works
- New parameters must have sensible defaults
- Function signatures: add optional parameters only
- No breaking changes to file naming conventions

### 2.3 Testing Strategy

- Visual inspection of plots (before/after comparison)
- CSV file validation (column presence, data types)
- Execute full notebook run on at least one dataset per phase
- Verify with high-column (>15) and low-column (<10) datasets

### 2.4 SOLID Principles

- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Add new metrics without modifying existing calculations
- **Dependency Inversion**: Use function parameters for configuration

---

## 3. IMPLEMENTATION PHASES

---

## PHASE 0: FOUNDATION - MODULAR ARCHITECTURE

**Goal**: Create src/ directory structure and thin setup.py re-export layer
**Duration**: 1 day
**Dependencies**: None (foundational)

---

### TASK 0.1: Create Modular src/ Structure

**Priority**: P0 (Must do first)

**Problem**: setup.py is 3814 lines, will grow to 4000+ with improvements. Need clean architecture before adding features.

**Approach**: Create src/ modules, migrate essential code, establish setup.py as re-export layer.

**Implementation Steps**:

**Step 1: Create Directory Structure**
```bash
cd tableGenCompare
mkdir -p src/data src/models src/evaluation src/visualization src/objective src/utils
touch src/__init__.py
touch src/data/__init__.py src/models/__init__.py src/evaluation/__init__.py
touch src/visualization/__init__.py src/objective/__init__.py src/utils/__init__.py
```

**Step 2: Create Core Modules**

**src/config.py**:
```python
"""Global configuration and session management"""
from datetime import datetime

# Session timestamp (captured at import time)
SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d")
DATASET_IDENTIFIER = None
CURRENT_DATA_FILE = None

def refresh_session_timestamp():
    global SESSION_TIMESTAMP
    SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d")
    return SESSION_TIMESTAMP

print(f"Session timestamp: {SESSION_TIMESTAMP}")
```

**src/utils/paths.py**:
```python
"""Path utilities for results organization"""
import os
from src.config import SESSION_TIMESTAMP

def extract_dataset_identifier(data_file_path):
    """Extract dataset identifier from file path"""
    if isinstance(data_file_path, str):
        filename = os.path.basename(data_file_path)
        dataset_id = os.path.splitext(filename)[0].lower()
        dataset_id = dataset_id.replace('_', '-').replace(' ', '-')
        return dataset_id
    return "unknown-dataset"

def get_results_path(dataset_identifier, section_number):
    """Generate standardized results path"""
    return f"results/{dataset_identifier}/{SESSION_TIMESTAMP}/Section-{section_number}"
```

**Step 3: Create Essential Imports Module**

**src/__init__.py**:
```python
"""Essential third-party imports for notebook convenience"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# Core ML/preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

print("[OK] Essential libraries imported successfully!")
```

**Step 4: Create Thin setup.py Re-Export Layer**

**NEW setup.py**:
```python
"""
Backward-compatible import layer for notebooks.
Notebooks can still use: from setup import *

This file re-exports all functionality from src/ modules.
"""

# Essential imports (for notebook convenience)
from src import *

# Re-export all modules (will be populated as we migrate)
from src.config import *
from src.utils.paths import *

# Model imports (to be migrated from CHUNK_001, 001B, 001C)
# from src.models.imports import *
# from src.models.wrappers import *

# Data utilities (to be migrated)
# from src.data.preprocessing import *
# from src.data.loading import *

# Evaluation (to be migrated/created)
# from src.evaluation.quality import *
# from src.evaluation.trts import *
# from src.evaluation.privacy import *
# from src.evaluation.mode_collapse import *

# Visualization (to be created in Phase 2)
# from src.visualization.section2 import *
# from src.visualization.section3 import *
# from src.visualization.section5 import *

# Objective functions (to be migrated)
# from src.objective.functions import *

# Backward compatibility notes:
# CHUNK_001 → src.models.imports
# CHUNK_002 → src.models.wrappers.CTABGANModel
# CHUNK_003 → src.models.wrappers.CTABGANPlusModel
# CHUNK_004 → src (essential imports)
# CHUNK_017 → src.evaluation.quality
# CHUNK_037 → src.objective.functions
# CHUNK_039 → src.evaluation (optimization analysis)

print("[OK] Clinical Synthetic Data Generation Framework loaded successfully!")
print("[INFO] Using modular src/ architecture")
```

**Step 5: Create Placeholder Modules** (to be populated in later phases)

**src/visualization/section2.py**:
```python
"""Section 2 Visualization Functions (to be implemented in Phase 2)"""
# create_correlation_heatmap()
# create_feature_distributions()
pass
```

**src/visualization/section3.py**, **src/visualization/section5.py**: Similar placeholders

**src/evaluation/privacy.py**, **src/evaluation/mode_collapse.py**: Placeholders for Phase 3

**Step 6: Migrate Critical Unchanged Functions**

Move these from setup.py to src/ immediately (they won't change):
- Model imports (CHUNK_001, 001B, 001C) → `src/models/imports.py`
- Model wrappers (CHUNK_002, 003) → `src/models/wrappers.py`
- Path utilities (already done in Step 2)

**Step 7: Test Backward Compatibility**

Test in ONE notebook:
```python
# Should still work!
from setup import *

# Verify imports
print(f"SESSION_TIMESTAMP: {SESSION_TIMESTAMP}")
print(f"get_results_path: {get_results_path('test', 2)}")

# Check pandas, numpy, etc still imported
df = pd.DataFrame({'a': [1,2,3]})
print(df.head())
```

**Success Criteria**:
- [ ] src/ directory structure created with all subdirectories
- [ ] Core modules created (config.py, utils/paths.py, __init__.py)
- [ ] New setup.py successfully re-exports from src/
- [ ] At least ONE notebook imports successfully with `from setup import *`
- [ ] Essential functions work (get_results_path, SESSION_TIMESTAMP)
- [ ] All 7 notebooks tested and working with new setup.py

**Files Modified**:
- NEW: All files in `src/` directory
- MODIFIED: `setup.py` (replaced with thin re-export layer)

**Dependencies**: None

**Estimated Time**: 1 day (includes testing all 7 notebooks)

---

## PHASE 1: DOCUMENTATION

**Goal**: Create README files describing outputs (establishes targets for features)
**Duration**: 1 day
**Dependencies**: Phase 0 complete

---

### TASK 1.1: Create Section 2 README.md

**Priority**: P2 (Documentation)

**Problem**: Section 2 output folders lack documentation explaining file purposes.

**Affected Files**:
- All 7 notebooks: Add code to Section 2 to generate README.md
- NEW file: `results/{dataset_id}/{date}/Section-2/README.md`

**Implementation**:

**Create function in src/utils/documentation.py**:
```python
"""Documentation generation utilities"""

def create_section2_readme(results_path, dataset_id, timestamp):
    """Create README.md explaining Section 2 output files"""
    readme_content = f"""# Section 2: Exploratory Data Analysis - Output Files

This folder contains EDA results performed before model training.

## Visualizations

- **correlation_heatmap.png**: Pairwise correlations between numeric features. Identifies multicollinearity and relationships. Font size adjusts dynamically based on number of features.

- **feature_distributions.png** (or **feature_distributions_part1.png**, **part2.png**, ...): Histograms of each numeric feature. Multiple files generated for datasets with many features (6 features per file in 3x2 grid for readability).

## Data Analysis CSV Files

- **correlation_matrix.csv**: Raw correlation coefficients between all numeric features (-1 to +1 scale).

- **target_correlations.csv**: Correlations between each feature and target variable, sorted by absolute strength. Identifies most predictive features.

- **column_analysis.csv**: Comprehensive statistics per column including data type, missing values, unique values, and descriptive statistics (mean, std, min, max).

- **target_balance_metrics.csv**: Class distribution analysis for target variable, including counts, proportions, and balance metrics. Important for understanding class imbalance.

- **target_analysis.csv**: Detailed target variable characteristics and distribution.

## Purpose

These files provide baseline understanding of:
- Data quality (missing values, distributions)
- Feature relationships (correlations)
- Target characteristics (class balance)
- Potential preprocessing needs

## Next Steps

After reviewing these files, proceed to Section 3 for synthetic data generation and model evaluation.

---
*Generated by Clinical Synthetic Data Generation Framework*
*Dataset: {dataset_id}*
*Date: {timestamp}*
"""

    readme_path = f"{results_path}/README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"[DOCS] Created: README.md")
    return readme_path
```

**Add to setup.py re-exports**:
```python
from src.utils.documentation import *
```

**Add to ALL 7 notebooks at end of Section 2**:
```python
# Generate README for Section 2 outputs
from src.utils.documentation import create_section2_readme
results_path = get_results_path(DATASET_IDENTIFIER, 2)
create_section2_readme(results_path, DATASET_IDENTIFIER, SESSION_TIMESTAMP)
```

**Success Criteria**:
- [ ] README.md created in all Section-2 output folders
- [ ] Content accurately describes current AND future files (forward-compatible)
- [ ] References multi-part feature distribution files
- [ ] All 7 notebooks generate README.md
- [ ] README includes dataset ID and timestamp

**Files Modified**:
- NEW: `src/utils/documentation.py`
- MODIFIED: `setup.py` (add documentation import)
- MODIFIED: All 7 notebooks (add README generation to Section 2)

**Dependencies**: Task 0.1 (needs src/ structure)

---

### TASK 1.2: Create Section 3 README.md with Column Definitions

**Priority**: P2 (Documentation)

**Problem**: Section 3 CSV files need column definitions and interpretation guidance.

**Affected Files**:
- `src/utils/documentation.py`: Add new functions
- All 7 notebooks: Add code to Section 3

**Implementation**:

**Add to src/utils/documentation.py**:
```python
def create_section3_main_readme(results_path, dataset_id, timestamp):
    """Create main README for Section 3 root folder"""
    readme_content = f"""# Section 3: Model Evaluation - Output Files

This folder contains evaluation results for all trained synthetic data generation models.

## Structure

Each model has its own subfolder:
- **CTGAN/** - Conditional Tabular GAN results
- **CTABGAN/** - CTAB-GAN results
- **CTABGANPLUS/** - CTAB-GAN+ results
- **GANERAID/** - GANerAid results
- **COPULAGAN/** - CopulaGAN results
- **TVAE/** - Tabular Variational Autoencoder results

## Common Files Per Model

### CSV Files
- **evaluation_summary.csv** - High-level quality scores
- **statistical_similarity.csv** - Feature-level similarity metrics
- **mode_collapse_analysis.csv** - Categorical variable diversity check (if applicable)

### Visualizations
- **correlation_comparison.png** - Side-by-side heatmaps (real vs synthetic)
- **distribution_comparison_partN.png** - Overlaid histograms per feature
- **pca_comparison_with_outcome.png** - Principal component analysis
- **mode_collapse_summary.png** - Category coverage visualization (if mode collapse detected)
- **mutual_information_comparison.png** - MI preservation analysis (if computed)

## Column Definitions

### evaluation_summary.csv

| Column | Description |
|--------|-------------|
| model | Model name identifier |
| avg_statistical_similarity | Mean/std matching across features (0-1, higher better) |
| avg_js_similarity | Jensen-Shannon divergence similarity (0-1, 1=identical distributions) |
| correlation_preservation | Pearson correlation between real and synthetic correlation matrices (0-1) |
| overall_pca_similarity | Average similarity across principal components (0-1) |
| ml_utility | Cross-training accuracy average (0-1) |
| overall_quality_score | Composite quality score (0-1) |
| quality_assessment | Categorical: EXCELLENT (≥0.8), GOOD (≥0.6), FAIR (≥0.4), POOR (<0.4) |
| mode_collapse_detected | Boolean: True if any categorical variable lost diversity |
| mode_collapse_count | Number of features with mode collapse |
| mutual_information_preservation | MI score preservation (0-1, if computed) |

### statistical_similarity.csv

| Column | Description |
|--------|-------------|
| column | Feature name |
| real_mean | Mean value in real dataset |
| synthetic_mean | Mean value in synthetic dataset |
| mean_similarity | How closely means match (0-1, normalized by std) |
| std_similarity | Ratio of standard deviations (0-1, 1=perfect match) |
| overall_similarity | Average of mean_similarity and std_similarity |

### mode_collapse_analysis.csv

| Column | Description |
|--------|-------------|
| column | Categorical feature name |
| real_unique_count | Number of unique categories in real data |
| synthetic_unique_count | Number of unique categories in synthetic data |
| category_coverage | Proportion of real categories present in synthetic (0-1) |
| distribution_similarity | 1 - Total Variation Distance for category frequencies (0-1) |
| mode_collapse_flag | Boolean: True if collapse detected |
| collapse_severity | Severe / Moderate / Mild / None |
| missing_categories | List of real categories absent in synthetic |
| extra_categories | List of synthetic categories not in real |

## Interpretation Guide

### pca_comparison_with_outcome.png (4-panel visualization)

**Top-Left (Real Data PC1 vs PC2)**: Scatter plot of real data points in principal component space. Colors indicate target values. Shows natural clustering.

**Top-Right (Synthetic Data PC1 vs PC2)**: Scatter plot of synthetic data. Should exhibit similar clustering patterns and color distribution if synthetic data captures relationships well.

**Bottom-Left (Explained Variance)**: Bar chart showing variance captured by each principal component. Real and synthetic should have similar distributions if structure preserved.

**Bottom-Right (Component Similarity)**: Correlation between real and synthetic for each PC (0-1 scale). Higher scores (>0.7) indicate better structural preservation.

**What to look for**:
- Similar scatter patterns (top-left vs top-right)
- Comparable color distributions (target relationships preserved)
- Similar explained variance ratios (bottom-left)
- High component similarity scores (bottom-right, ideally >0.7)

## Usage

Compare evaluation_summary.csv across all models to identify best performers. Review model-specific visualizations to understand strengths and weaknesses.

---
*Generated by Clinical Synthetic Data Generation Framework*
*Dataset: {dataset_id}*
*Date: {timestamp}*
"""

    readme_path = f"{results_path}/README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"[DOCS] Created: Section-3/README.md")
    return readme_path


def create_section3_model_readme(results_dir, model_name, quality_score, quality_rating,
                                 dataset_id, timestamp):
    """Create model-specific README"""
    readme_content = f"""# {model_name.upper()} - Synthetic Data Quality Evaluation

## Model: {model_name.upper()}

## Generated Files

- **evaluation_summary.csv**: Overall quality metrics
- **statistical_similarity.csv**: Per-feature similarity analysis
- **correlation_comparison.png**: Correlation matrix comparison
- **distribution_comparison_partN.png**: Feature distribution comparisons
- **pca_comparison_with_outcome.png**: Principal component analysis

## Quick Assessment

- **Overall Quality Score**: {quality_score:.3f}
- **Quality Rating**: {quality_rating}

Refer to parent folder README.md for detailed column definitions and interpretation guidance.

---
*Dataset: {dataset_id}*
*Generated: {timestamp}*
"""

    readme_path = f"{results_dir}/README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    return readme_path
```

**Integrate into evaluation code**:

When `evaluate_synthetic_data_quality()` is called (will be migrated to src/evaluation/quality.py in Phase 2), add README generation at the end.

**Success Criteria**:
- [ ] Main README.md in Section-3 root folder
- [ ] Per-model README.md in each MODEL subfolder
- [ ] Column definitions match actual/planned CSV structure
- [ ] Forward-compatible: includes mode_collapse and MI columns
- [ ] PCA interpretation guide clear and accurate
- [ ] All 7 notebooks generate READMEs

**Files Modified**:
- MODIFIED: `src/utils/documentation.py` (add Section 3 functions)
- MODIFIED: `src/evaluation/quality.py` (add README generation call - will do in Phase 2)
- MODIFIED: All 7 notebooks (add main Section 3 README generation)

**Dependencies**: Task 0.1

---

## PHASE 2: VISUALIZATION IMPROVEMENTS

**Goal**: Fix graphics rendering, move code to src/visualization/
**Duration**: 3-4 days
**Dependencies**: Phase 0 complete

**Strategy**: Create functions in src/visualization/ modules, update notebooks to call functions

---

### TASK 2.1: Correlation Heatmap - Dynamic Font Sizing

**Priority**: P1

**Problem**: Heatmap annotations overlap when >15 columns.

**Approach**: Create `create_correlation_heatmap()` in src/visualization/section2.py

**Implementation**:

**NEW: src/visualization/section2.py**:
```python
"""
Section 2 Visualization Functions
EDA plots: correlation heatmaps, feature distributions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def create_correlation_heatmap(correlation_matrix, results_path,
                               filename='correlation_heatmap.png',
                               verbose=True):
    """
    Create correlation heatmap with dynamic font sizing.

    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Correlation matrix to visualize
    results_path : str or Path
        Directory to save output
    filename : str
        Output filename (default: 'correlation_heatmap.png')
    verbose : bool
        Print progress messages

    Returns:
    --------
    str : Path to saved file
    """
    n_cols = len(correlation_matrix.columns)

    # Dynamic annotation control based on column count
    if n_cols <= 10:
        show_annot, font_size, fmt = True, 10, '.3f'
    elif n_cols <= 15:
        show_annot, font_size, fmt = True, 8, '.2f'
    elif n_cols <= 20:
        show_annot, font_size, fmt = True, 6, '.2f'
    else:
        show_annot, font_size, fmt = False, None, '.2f'

    # Dynamic figure size
    figsize = (max(10, n_cols * 0.6), max(8, n_cols * 0.5))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(correlation_matrix,
                annot=show_annot,
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                fmt=fmt,
                annot_kws={'size': font_size} if show_annot else {},
                ax=ax)

    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = Path(results_path) / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: {filename}")

    return str(output_path)
```

**Add to setup.py**:
```python
from src.visualization.section2 import *
```

**Update ALL 7 notebooks Section 2**:

REPLACE inline heatmap code with:
```python
# Create correlation heatmap with dynamic font sizing
from src.visualization.section2 import create_correlation_heatmap
correlation_matrix = data_processed[numeric_cols].corr()
create_correlation_heatmap(correlation_matrix, results_path)
```

**Success Criteria**:
- [ ] Heatmap with ≤10 columns: Annotations visible, font 10
- [ ] Heatmap with 11-15: Annotations visible, font 8
- [ ] Heatmap with >20: No annotations, color only
- [ ] Figure size scales with column count
- [ ] All 7 notebooks updated and tested

**Files Modified**:
- NEW: `src/visualization/section2.py`
- MODIFIED: `setup.py` (add import)
- MODIFIED: All 7 notebooks Section 2 (replace inline code)

**Dependencies**: Task 0.1

---

### TASK 2.2: Feature Distributions - Multi-File Grid Splitting

**Priority**: P1

**Problem**: Datasets with >18 columns create excessively tall plots.

**Approach**: Add `create_feature_distributions()` to src/visualization/section2.py

**Implementation**:

**Add to src/visualization/section2.py**:
```python
def create_feature_distributions(data, target_column, results_path,
                                 plots_per_file=6, verbose=True):
    """
    Create feature distribution plots with multi-file splitting.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with features to plot
    target_column : str
        Target column to exclude from plots
    results_path : str or Path
        Directory to save outputs
    plots_per_file : int
        Number of plots per file (default 6 for 3x2 grid)
    verbose : bool
        Print progress messages

    Returns:
    --------
    list : Paths to saved files
    """
    GRID_COLS, GRID_ROWS = 3, 2

    # Get numeric columns excluding target
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_no_target = [col for col in numeric_cols if col != target_column]

    # Split into chunks
    column_chunks = [numeric_cols_no_target[i:i+plots_per_file]
                     for i in range(0, len(numeric_cols_no_target), plots_per_file)]

    saved_files = []

    if verbose:
        print(f"[VIZ] Creating {len(column_chunks)} feature distribution file(s)...")

    for file_idx, cols_subset in enumerate(column_chunks, 1):
        fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(15, 8))
        fig.suptitle(f'Feature Distributions (Part {file_idx}/{len(column_chunks)})',
                     fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i, col in enumerate(cols_subset):
            axes[i].hist(data[col].dropna(), bins=20, edgecolor='black', alpha=0.7)
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel('Value', fontsize=8)
            axes[i].set_ylabel('Frequency', fontsize=8)
            axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(len(cols_subset), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        # File naming (backward compatible)
        if len(column_chunks) > 1:
            filename = f'feature_distributions_part{file_idx}.png'
        else:
            filename = 'feature_distributions.png'

        output_path = Path(results_path) / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        saved_files.append(str(output_path))

        if verbose:
            print(f"[VIZ] Saved: {filename}")

    return saved_files
```

**Update ALL 7 notebooks Section 2**:

REPLACE inline distribution code with:
```python
# Create feature distribution plots (multi-file for large datasets)
from src.visualization.section2 import create_feature_distributions
create_feature_distributions(data_processed, TARGET_COLUMN, results_path)
```

**Success Criteria**:
- [ ] ≤6 columns: Single file `feature_distributions.png`
- [ ] >6 columns: Multiple part files
- [ ] Each file 3x2 grid (6 plots max)
- [ ] Original filename when only 1 file
- [ ] All 7 notebooks updated

**Files Modified**:
- MODIFIED: `src/visualization/section2.py` (add function)
- MODIFIED: All 7 notebooks Section 2

**Dependencies**: Task 2.1 (same module)

---

### TASK 2.3: TRTS Y-Axis Adjustment

**Priority**: P1

**Problem**: Bar labels overlap title when accuracy near 1.0.

**Approach**: Migrate `create_trts_visualizations()` from setup.py to src/visualization/section5.py and add dynamic y-limits.

**Implementation**:

**Step 1: Create src/visualization/section5.py**:

Copy `create_trts_visualizations()` from setup.py (lines 2986-3176) to new file.

**Step 2: Modify y-limit logic**:

Find "Utility Metrics Comparison" subplot (line ~3121), replace:
```python
# OLD
ax3.set_ylim(0, 1.0)

# NEW
max_accuracy = max(model_df[scenarios].max())
if max_accuracy > 0.95:
    y_max = 1.1
elif max_accuracy > 0.85:
    y_max = 1.05
else:
    y_max = 1.0
ax3.set_ylim(0, y_max)
```

Apply same logic to subplot 1 (Overall Performance, line ~3082).

**Step 3: Remove from setup.py, add import**:
```python
# In setup.py
from src.visualization.section5 import *
```

**Success Criteria**:
- [ ] When max accuracy >0.95, y-axis extends to 1.1
- [ ] Labels don't overlap title
- [ ] Function moved from setup.py to src/
- [ ] All notebooks work without changes

**Files Modified**:
- NEW: `src/visualization/section5.py` (migrate function from setup.py)
- MODIFIED: `setup.py` (remove function, add import)

**Dependencies**: Task 0.1

---

### TASK 2.4: Correlation Comparison - Dual Heatmap Fonts

**Priority**: P2

**Problem**: Section 3 side-by-side correlation heatmaps have font overlap issues.

**Approach**: Migrate relevant code from `evaluate_synthetic_data_quality()` to src/visualization/section3.py helper function.

**Implementation**:

**Step 1: Create src/visualization/section3.py**:
```python
"""Section 3 Visualization Functions"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def create_correlation_comparison(real_corr, synth_corr, model_name, results_dir,
                                  verbose=True):
    """
    Create side-by-side correlation heatmap comparison.

    Parameters:
    -----------
    real_corr : pd.DataFrame
        Real data correlation matrix
    synth_corr : pd.DataFrame
        Synthetic data correlation matrix
    model_name : str
        Model name for title
    results_dir : Path
        Directory to save output
    verbose : bool
        Print messages

    Returns:
    --------
    str : Path to saved file
    """
    n_cols = len(real_corr.columns)

    # Dynamic settings (more conservative for dual display)
    if n_cols <= 8:
        show_annot, font_size, fmt, figsize = True, 9, '.2f', (16, 6)
    elif n_cols <= 12:
        show_annot, font_size, fmt, figsize = True, 7, '.2f', (18, 8)
    elif n_cols <= 18:
        show_annot, font_size, fmt, figsize = True, 5, '.2f', (20, 10)
    else:
        show_annot, font_size, fmt = False, None, '.2f'
        figsize = (max(20, n_cols * 0.8), max(10, n_cols * 0.5))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'{model_name.upper()} - Correlation Structure Comparison',
                 fontsize=16, fontweight='bold')

    # Real data
    sns.heatmap(real_corr, annot=show_annot, cmap='RdBu_r', center=0,
                square=True, ax=axes[0], fmt=fmt,
                annot_kws={'size': font_size} if show_annot else {},
                cbar_kws={'shrink': 0.8})
    axes[0].set_title('Real Data', fontsize=12)

    # Synthetic data
    sns.heatmap(synth_corr, annot=show_annot, cmap='RdBu_r', center=0,
                square=True, ax=axes[1], fmt=fmt,
                annot_kws={'size': font_size} if show_annot else {},
                cbar_kws={'shrink': 0.8})
    axes[1].set_title('Synthetic Data', fontsize=12)

    plt.tight_layout()

    output_file = results_dir / 'correlation_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: correlation_comparison.png")

    return str(output_file)
```

**Step 2: Migrate evaluate_synthetic_data_quality() to src/evaluation/quality.py**:

Copy function from setup.py, replace correlation heatmap inline code with call to helper:
```python
# In evaluate_synthetic_data_quality(), correlation section
from src.visualization.section3 import create_correlation_comparison

# Replace inline heatmap code with:
corr_plot_file = create_correlation_comparison(
    real_corr, synth_corr, model_name, results_dir, verbose
)
results['files_generated'].append(corr_plot_file)
```

**Step 3: Update setup.py imports**:
```python
from src.visualization.section3 import *
from src.evaluation.quality import *
```

**Success Criteria**:
- [ ] Dual heatmaps ≤8 cols: Annotations, font 9
- [ ] Dual heatmaps >18 cols: No annotations
- [ ] Figure width scales with column count
- [ ] Notebooks work without changes

**Files Modified**:
- NEW: `src/visualization/section3.py`
- NEW: `src/evaluation/quality.py` (migrate from setup.py)
- MODIFIED: `setup.py` (remove function, add imports)

**Dependencies**: Task 0.1

---

### TASK 2.5: Distribution Comparison - Multi-File Splitting

**Priority**: P2

**Problem**: Similar to Task 2.2 but for Section 3 overlaid histograms.

**Approach**: Add helper to src/visualization/section3.py, integrate into quality.py

**Implementation**:

**Add to src/visualization/section3.py**:
```python
def create_distribution_comparison(real_data, synthetic_data, numeric_cols_no_target,
                                   model_name, results_dir, plots_per_file=6,
                                   display_plots=False, verbose=True):
    """
    Create distribution comparison plots with multi-file splitting.

    Returns:
    --------
    tuple : (list of file paths, avg_js_similarity)
    """
    from scipy.spatial.distance import jensenshannon

    GRID_COLS, GRID_ROWS = 3, 2

    column_chunks = [numeric_cols_no_target[i:i+plots_per_file]
                     for i in range(0, len(numeric_cols_no_target), plots_per_file)]

    saved_files = []
    js_scores = []

    for file_idx, cols_subset in enumerate(column_chunks, 1):
        fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(15, 8))
        fig.suptitle(f'{model_name.upper()} - Distribution Comparison (Part {file_idx}/{len(column_chunks)})',
                     fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i, col in enumerate(cols_subset):
            ax = axes[i]

            # Calculate JS divergence
            real_hist, bins = np.histogram(real_data[col].dropna(), bins=20, density=True)
            synth_hist, _ = np.histogram(synthetic_data[col].dropna(), bins=bins, density=True)

            real_hist = real_hist / real_hist.sum() if real_hist.sum() > 0 else real_hist
            synth_hist = synth_hist / synth_hist.sum() if synth_hist.sum() > 0 else synth_hist

            js_div = jensenshannon(real_hist, synth_hist)
            js_similarity = 1 - js_div
            js_scores.append(js_similarity)

            # Plot overlaid histograms
            ax.hist(real_data[col].dropna(), bins=20, alpha=0.7, label='Real',
                   density=True, color='blue', edgecolor='black')
            ax.hist(synthetic_data[col].dropna(), bins=20, alpha=0.7, label='Synthetic',
                   density=True, color='orange', edgecolor='black')
            ax.set_title(f'{col}\nJS Sim: {js_similarity:.3f}', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(len(cols_subset), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        # File naming
        if len(column_chunks) > 1:
            filename = f'distribution_comparison_part{file_idx}.png'
        else:
            filename = 'distribution_comparison.png'

        output_file = results_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        saved_files.append(str(output_file))

        if display_plots:
            plt.show()
        else:
            plt.close()

    avg_js_similarity = np.mean(js_scores) if js_scores else 0

    if verbose:
        print(f"[VIZ] Generated {len(column_chunks)} distribution comparison file(s)")
        print(f"[VIZ] Average JS Similarity: {avg_js_similarity:.3f}")

    return saved_files, avg_js_similarity
```

**Integrate into src/evaluation/quality.py**:

Replace inline distribution code with call to helper.

**Success Criteria**:
- [ ] ≤6 columns: Single file
- [ ] >6 columns: Multiple part files (3x2 each)
- [ ] JS scores averaged correctly across files
- [ ] Notebooks work without changes

**Files Modified**:
- MODIFIED: `src/visualization/section3.py` (add function)
- MODIFIED: `src/evaluation/quality.py` (use helper)

**Dependencies**: Task 2.4

---

## PHASE 3: ADVANCED METRICS & VISUALIZATIONS

**Goal**: Add mode collapse detection, privacy metrics, MI metrics, visualizations
**Duration**: 5-6 days
**Dependencies**: Phase 0 and 2 complete

**Strategy**: Add new code directly to proper src/ modules from start

---

### TASK 3.1: Mode Collapse Detection + CSV

**Priority**: P1

**Problem**: GANs can suffer mode collapse. Need automatic detection.

**Approach**: Create NEW module src/evaluation/mode_collapse.py

**Implementation**:

**NEW: src/evaluation/mode_collapse.py**:
```python
"""
Mode Collapse Detection for Categorical Variables
Detects when synthetic data loses diversity in categorical features
"""

import pandas as pd
import numpy as np
from scipy import stats


def detect_mode_collapse(real_data, synthetic_data, target_column=None, verbose=True):
    """
    Detect mode collapse in categorical variables.

    Parameters:
    -----------
    real_data : pd.DataFrame
        Real dataset
    synthetic_data : pd.DataFrame
        Synthetic dataset
    target_column : str
        Target column to exclude
    verbose : bool
        Print warnings

    Returns:
    --------
    dict : {
        'mode_collapse_detected': bool,
        'mode_collapse_count': int,
        'mode_collapse_df': pd.DataFrame,
        'summary': str
    }
    """
    mode_collapse_results = []

    # Get categorical columns
    categorical_cols = [col for col in real_data.columns
                       if real_data[col].dtype in ['object', 'category']]

    # Add integer columns with ≤10 unique values
    for col in real_data.select_dtypes(include=['int64', 'int32']).columns:
        if col != target_column and real_data[col].nunique() <= 10:
            categorical_cols.append(col)

    # Remove target if present
    if target_column and target_column in categorical_cols:
        categorical_cols.remove(target_column)

    for col in categorical_cols:
        if col not in synthetic_data.columns:
            continue

        real_unique = set(real_data[col].dropna().unique())
        synth_unique = set(synthetic_data[col].dropna().unique())

        # Coverage: % of real categories in synthetic
        coverage = len(synth_unique & real_unique) / len(real_unique) if len(real_unique) > 0 else 0

        # Flag mode collapse
        mode_collapse_flag = False
        collapse_severity = "None"

        if len(real_unique) > 1 and len(synth_unique) == 1:
            mode_collapse_flag = True
            collapse_severity = "Severe"  # Total collapse to single mode
        elif len(real_unique) > 2 and coverage < 0.5:
            mode_collapse_flag = True
            collapse_severity = "Moderate"  # Lost >50% of modes
        elif coverage < 0.8:
            mode_collapse_flag = True
            collapse_severity = "Mild"  # Lost 20-50% of modes

        # Calculate distribution divergence (Total Variation Distance)
        real_freq = real_data[col].value_counts(normalize=True).to_dict()
        synth_freq = synthetic_data[col].value_counts(normalize=True).to_dict()
        all_categories = set(real_freq.keys()) | set(synth_freq.keys())

        real_vec = np.array([real_freq.get(cat, 0) for cat in all_categories])
        synth_vec = np.array([synth_freq.get(cat, 0) for cat in all_categories])
        tv_distance = 0.5 * np.sum(np.abs(real_vec - synth_vec))
        distribution_similarity = 1 - tv_distance

        mode_collapse_results.append({
            'column': col,
            'real_unique_count': len(real_unique),
            'synthetic_unique_count': len(synth_unique),
            'category_coverage': coverage,
            'distribution_similarity': distribution_similarity,
            'mode_collapse_flag': mode_collapse_flag,
            'collapse_severity': collapse_severity,
            'missing_categories': list(real_unique - synth_unique),
            'extra_categories': list(synth_unique - real_unique)
        })

        if verbose and mode_collapse_flag:
            print(f"   [WARNING] {col}: {collapse_severity} mode collapse detected")
            print(f"      Real: {len(real_unique)} categories, Synthetic: {len(synth_unique)}")

    # Create results dictionary
    if mode_collapse_results:
        mode_collapse_df = pd.DataFrame(mode_collapse_results)
        overall_collapse = mode_collapse_df['mode_collapse_flag'].any()
        collapse_count = mode_collapse_df['mode_collapse_flag'].sum()

        summary = f"Mode collapse detected in {collapse_count}/{len(mode_collapse_results)} categorical features"

        return {
            'mode_collapse_detected': overall_collapse,
            'mode_collapse_count': collapse_count,
            'mode_collapse_df': mode_collapse_df,
            'summary': summary
        }
    else:
        return {
            'mode_collapse_detected': False,
            'mode_collapse_count': 0,
            'mode_collapse_df': pd.DataFrame(),
            'summary': "No categorical variables to analyze"
        }
```

**Integrate into src/evaluation/quality.py**:

Add to `evaluate_synthetic_data_quality()` after distribution similarity section:
```python
# Mode Collapse Detection
from src.evaluation.mode_collapse import detect_mode_collapse

mode_collapse_results = detect_mode_collapse(
    real_data, synthetic_data, target_column, verbose
)

results['mode_collapse_detected'] = mode_collapse_results['mode_collapse_detected']
results['mode_collapse_count'] = mode_collapse_results['mode_collapse_count']

# Save CSV
if save_files and not mode_collapse_results['mode_collapse_df'].empty:
    mc_file = results_dir / 'mode_collapse_analysis.csv'
    mode_collapse_results['mode_collapse_df'].to_csv(mc_file, index=False)
    results['files_generated'].append(str(mc_file))
```

**Update setup.py**:
```python
from src.evaluation.mode_collapse import *
```

**Success Criteria**:
- [ ] Detects severe collapse (1 value when real >1)
- [ ] Detects moderate collapse (<50% coverage)
- [ ] Saves `mode_collapse_analysis.csv`
- [ ] Adds flags to `evaluation_summary.csv`
- [ ] No false positives on good data

**Files Modified**:
- NEW: `src/evaluation/mode_collapse.py`
- MODIFIED: `src/evaluation/quality.py` (integrate detection)
- MODIFIED: `setup.py` (add import)

**Dependencies**: Task 0.1, Task 2.4 (needs quality.py migrated)

---

### TASK 3.2: Mode Collapse Visualization

**Priority**: P2

**Problem**: CSV alone doesn't convey severity visually.

**Approach**: Add visualization function to src/visualization/section3.py

**Implementation**:

**Add to src/visualization/section3.py**:
```python
def create_mode_collapse_visualization(mode_collapse_df, model_name, results_dir, verbose=True):
    """
    Create mode collapse summary visualization.

    Parameters:
    -----------
    mode_collapse_df : pd.DataFrame
        Mode collapse analysis results
    model_name : str
        Model name for title
    results_dir : Path
        Directory to save output

    Returns:
    --------
    str : Path to saved file (or None if no data)
    """
    if mode_collapse_df.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name.upper()} - Mode Collapse Analysis',
                 fontsize=16, fontweight='bold')

    # Left: Category coverage per column
    ax1.barh(mode_collapse_df['column'], mode_collapse_df['category_coverage'],
             color=['red' if x < 0.5 else 'orange' if x < 0.8 else 'green'
                    for x in mode_collapse_df['category_coverage']])
    ax1.axvline(x=0.8, color='orange', linestyle='--', linewidth=2, label='Mild threshold')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Moderate threshold')
    ax1.set_xlabel('Category Coverage (higher is better)')
    ax1.set_title('Category Coverage by Feature')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')

    # Right: Distribution similarity
    ax2.barh(mode_collapse_df['column'], mode_collapse_df['distribution_similarity'],
             color=['red' if x < 0.5 else 'orange' if x < 0.7 else 'green'
                    for x in mode_collapse_df['distribution_similarity']])
    ax2.set_xlabel('Distribution Similarity (higher is better)')
    ax2.set_title('Categorical Distribution Similarity')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    output_file = results_dir / 'mode_collapse_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: mode_collapse_summary.png")

    return str(output_file)
```

**Integrate into src/evaluation/quality.py**:

After mode collapse detection:
```python
# Visualize mode collapse if detected
if mode_collapse_results['mode_collapse_detected']:
    from src.visualization.section3 import create_mode_collapse_visualization
    mc_viz_file = create_mode_collapse_visualization(
        mode_collapse_results['mode_collapse_df'],
        model_name,
        results_dir,
        verbose
    )
    if mc_viz_file:
        results['files_generated'].append(mc_viz_file)
```

**Success Criteria**:
- [ ] Visualization created when mode collapse detected
- [ ] Bar colors indicate severity (red/orange/green)
- [ ] Thresholds clearly marked
- [ ] File saved to Section-3 model folder

**Files Modified**:
- MODIFIED: `src/visualization/section3.py` (add function)
- MODIFIED: `src/evaluation/quality.py` (integrate viz)

**Dependencies**: Task 3.1

---

### TASK 3.3: Higher-Order Metrics (Mutual Information)

**Priority**: P3

**Problem**: Current metrics are univariate. Need MI to capture feature interactions.

**Approach**: Add MI calculation to src/evaluation/quality.py

**Implementation**:

**Add to src/evaluation/quality.py** in `evaluate_synthetic_data_quality()`:

After correlation preservation section:
```python
# Higher-Order Similarity: Mutual Information
if verbose:
    print("\n[HIGHER-ORDER] Mutual Information Analysis")

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    # Select subset of columns (MI is expensive)
    mi_cols = numeric_cols_no_target[:min(10, len(numeric_cols_no_target))]

    if target_column and target_column in real_data.columns:
        X_real_mi = real_data[mi_cols].fillna(0)
        X_synth_mi = synthetic_data[mi_cols].fillna(0)
        y_real_mi = real_data[target_column]
        y_synth_mi = synthetic_data[target_column] if target_column in synthetic_data.columns else None

        if y_synth_mi is not None:
            # Classification or regression
            if y_real_mi.nunique() <= 10:
                mi_real = mutual_info_classif(X_real_mi, y_real_mi, random_state=42)
                mi_synth = mutual_info_classif(X_synth_mi, y_synth_mi, random_state=42)
            else:
                mi_real = mutual_info_regression(X_real_mi, y_real_mi, random_state=42)
                mi_synth = mutual_info_regression(X_synth_mi, y_synth_mi, random_state=42)

            # MI preservation score
            from scipy.stats import pearsonr
            mi_correlation = pearsonr(mi_real, mi_synth)[0] if len(mi_real) > 1 else 0
            mi_correlation = max(0, mi_correlation)

            results['mutual_information_preservation'] = mi_correlation

            # Store MI vectors for visualization
            results['mi_real'] = mi_real
            results['mi_synth'] = mi_synth
            results['mi_cols'] = mi_cols

            if verbose:
                print(f"   [METRIC] MI Preservation: {mi_correlation:.3f}")
        else:
            results['mutual_information_preservation'] = np.nan
except ImportError:
    if verbose:
        print("   [SKIP] sklearn.feature_selection not available")
    results['mutual_information_preservation'] = np.nan
except Exception as e:
    if verbose:
        print(f"   [ERROR] MI calculation failed: {e}")
    results['mutual_information_preservation'] = np.nan
```

**Success Criteria**:
- [ ] Calculates MI preservation score
- [ ] Adds to `evaluation_summary.csv`
- [ ] Completes in <5 min per model
- [ ] Handles edge cases gracefully

**Files Modified**:
- MODIFIED: `src/evaluation/quality.py` (add MI section)

**Dependencies**: Task 2.4

---

### TASK 3.4: MI Comparison Visualization

**Priority**: P3

**Problem**: MI scores need visual representation.

**Approach**: Add visualization to src/visualization/section3.py

**Implementation**:

**Add to src/visualization/section3.py**:
```python
def create_mi_comparison(mi_real, mi_synth, mi_cols, mi_correlation, model_name,
                        results_dir, verbose=True):
    """
    Create mutual information comparison visualization.

    Returns:
    --------
    str : Path to saved file
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(mi_cols))
    width = 0.35

    ax.bar(x - width/2, mi_real, width, label='Real Data', alpha=0.8, color='blue')
    ax.bar(x + width/2, mi_synth, width, label='Synthetic Data', alpha=0.8, color='orange')

    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Mutual Information with Target', fontsize=12)
    ax.set_title(f'{model_name.upper()} - MI Preservation (Correlation: {mi_correlation:.3f})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mi_cols, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_file = results_dir / 'mutual_information_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: mutual_information_comparison.png")

    return str(output_file)
```

**Integrate into src/evaluation/quality.py**:

After MI calculation:
```python
# Visualize MI if computed
if 'mi_real' in results and results.get('mutual_information_preservation', np.nan) is not np.nan:
    from src.visualization.section3 import create_mi_comparison
    mi_viz_file = create_mi_comparison(
        results['mi_real'],
        results['mi_synth'],
        results['mi_cols'],
        results['mutual_information_preservation'],
        model_name,
        results_dir,
        verbose
    )
    results['files_generated'].append(mi_viz_file)
```

**Success Criteria**:
- [ ] Side-by-side bar chart created
- [ ] MI correlation shown in title
- [ ] File saved to Section-3 model folder

**Files Modified**:
- MODIFIED: `src/visualization/section3.py` (add function)
- MODIFIED: `src/evaluation/quality.py` (integrate viz)

**Dependencies**: Task 3.3

---

### TASK 3.5: Comprehensive TRTS Metrics

**Priority**: P1

**Problem**: trts_detailed_results.csv only has Accuracy and Time. Need 15+ metrics.

**Approach**: Create enhanced metrics calculator in src/evaluation/trts.py

**Implementation**:

**Step 1: Migrate comprehensive_trts_analysis() to src/evaluation/trts.py**

Copy from setup.py, enhance with new helper function.

**Step 2: Create comprehensive metrics calculator**:

Add to src/evaluation/trts.py:
```python
def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate all classification metrics.

    Returns:
    --------
    dict : All metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, balanced_accuracy_score,
        matthews_corrcoef, cohen_kappa_score,
        roc_auc_score, average_precision_score
    )

    metrics = {'accuracy': accuracy_score(y_true, y_pred)}

    # Binary classification detailed metrics
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['recall'] = metrics['sensitivity']
        metrics['tpr'] = metrics['sensitivity']

        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['tnr'] = metrics['specificity']

        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['ppv'] = metrics['precision']

        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['fdr'] = fp / (fp + tp) if (fp + tp) > 0 else 0
        metrics['for'] = fn / (fn + tn) if (fn + tn) > 0 else 0
        metrics['prevalence'] = (tp + fn) / (tp + tn + fp + fn)
        metrics['predicted_positive_rate'] = (tp + fp) / (tp + tn + fp + fn)
    else:
        # Multiclass
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['sensitivity'] = recall

    # F1, Balanced Accuracy, MCC, Kappa
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average='binary' if len(np.unique(y_true)) == 2 else 'macro',
        zero_division=0
    )
    metrics['f1_score'] = f1
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    # AUC metrics
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['auroc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['auprc'] = average_precision_score(y_true, y_pred_proba)
                metrics['average_precision'] = metrics['auprc']
            else:
                # Multiclass (one-vs-rest)
                metrics['auroc'] = roc_auc_score(y_true, y_pred_proba,
                                                multi_class='ovr', average='macro')
                metrics['auprc'] = average_precision_score(y_true, y_pred_proba,
                                                           average='macro')
        except:
            metrics['auroc'] = np.nan
            metrics['auprc'] = np.nan
    else:
        metrics['auroc'] = np.nan
        metrics['auprc'] = np.nan

    return metrics
```

**Step 3: Update TRTS scenarios to use comprehensive metrics**:

In `comprehensive_trts_analysis()`, replace accuracy-only calculations:
```python
# SCENARIO 1: TRTR
start_time = time.time()
rf_trtr = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
rf_trtr.fit(X_real_train, y_real_train)
y_pred_trtr = rf_trtr.predict(X_real_test)
y_pred_proba_trtr = rf_trtr.predict_proba(X_real_test)[:, 1] if len(np.unique(y_real_test)) == 2 else None
trtr_time = time.time() - start_time

trtr_metrics = calculate_comprehensive_metrics(y_real_test, y_pred_trtr, y_pred_proba_trtr)
trtr_metrics['training_time'] = trtr_time
trtr_metrics['status'] = 'success'
results['TRTR'] = trtr_metrics

# Repeat for TRTS, TSTR, TSTS scenarios...
```

**Step 4: Update CSV output in Section 5 notebooks**:

Replace trts_detailed_results.csv generation:
```python
detailed_results = []
for model_name, trts_results in trts_results_dict.items():
    for scenario in ['TRTR', 'TRTS', 'TSTR', 'TSTS']:
        if scenario in trts_results and trts_results[scenario].get('status') == 'success':
            row = {
                'Model': model_name,
                'Scenario': scenario,
                'Accuracy': trts_results[scenario]['accuracy'],
                'F1_Score': trts_results[scenario].get('f1_score', np.nan),
                'Precision': trts_results[scenario].get('precision', np.nan),
                'Recall': trts_results[scenario].get('recall', np.nan),
                'Sensitivity': trts_results[scenario].get('sensitivity', np.nan),
                'Specificity': trts_results[scenario].get('specificity', np.nan),
                'NPV': trts_results[scenario].get('npv', np.nan),
                'FPR': trts_results[scenario].get('fpr', np.nan),
                'FNR': trts_results[scenario].get('fnr', np.nan),
                'Balanced_Accuracy': trts_results[scenario].get('balanced_accuracy', np.nan),
                'MCC': trts_results[scenario].get('mcc', np.nan),
                'Cohen_Kappa': trts_results[scenario].get('cohen_kappa', np.nan),
                'AUROC': trts_results[scenario].get('auroc', np.nan),
                'AUPRC': trts_results[scenario].get('auprc', np.nan),
                'Average_Precision': trts_results[scenario].get('average_precision', np.nan),
                'Training_Time_Sec': trts_results[scenario]['training_time']
            }
            detailed_results.append(row)

detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv(f'{results_path}/trts_detailed_results.csv', index=False)
```

**Step 5: Update setup.py**:
```python
from src.evaluation.trts import *
```

**Success Criteria**:
- [ ] trts_detailed_results.csv has 15+ columns
- [ ] Works for binary and multiclass
- [ ] All 7 notebooks generate expanded CSV
- [ ] Performance <10 min for Section 5

**Files Modified**:
- NEW: `src/evaluation/trts.py` (migrate + enhance from setup.py)
- MODIFIED: `setup.py` (remove function, add import)
- MODIFIED: All 7 notebooks Section 5 (update CSV generation)

**Dependencies**: Task 0.1

---

### TASK 3.6: Privacy Metrics Calculation

**Priority**: P1

**Problem**: Need privacy assessment (memorization, re-identification risk).

**Approach**: Create NEW module src/evaluation/privacy.py

**Implementation**:

**NEW: src/evaluation/privacy.py**:
```python
"""
Privacy Metrics for Synthetic Data
Assesses memorization risk, re-identification, and diversity
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import stats


def calculate_privacy_metrics(real_data, synthetic_data, target_column=None,
                              n_neighbors=5, verbose=True):
    """
    Calculate privacy metrics.

    Returns:
    --------
    dict : Privacy metrics
    """
    if verbose:
        print("\n[PRIVACY] Calculating privacy metrics...")

    # Prepare numeric data
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
    if target_column and target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if len(numeric_cols) == 0:
        return {'error': 'No numeric columns for privacy analysis'}

    X_real = real_data[numeric_cols].fillna(0).values
    X_synth = synthetic_data[numeric_cols].fillna(0).values

    # Standardize
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)
    X_synth_scaled = scaler.transform(X_synth)

    privacy_metrics = {}

    try:
        # NN distance: Synthetic to Real (memorization detection)
        nn_real = NearestNeighbors(n_neighbors=min(n_neighbors, len(X_real)))
        nn_real.fit(X_real_scaled)
        distances_s2r, _ = nn_real.kneighbors(X_synth_scaled)

        privacy_metrics['nn_dist_synth_to_real_mean'] = float(distances_s2r[:, 0].mean())
        privacy_metrics['nn_dist_synth_to_real_min'] = float(distances_s2r[:, 0].min())
        privacy_metrics['nn_dist_synth_to_real_std'] = float(distances_s2r[:, 0].std())

        # Memorization risk (5th percentile threshold)
        threshold_mem = np.percentile(distances_s2r[:, 0], 5)
        privacy_metrics['memorization_risk'] = float((distances_s2r[:, 0] < threshold_mem).mean())

        # NN distance: Real to Synthetic (coverage)
        nn_synth = NearestNeighbors(n_neighbors=min(n_neighbors, len(X_synth)))
        nn_synth.fit(X_synth_scaled)
        distances_r2s, _ = nn_synth.kneighbors(X_real_scaled)

        privacy_metrics['nn_dist_real_to_synth_mean'] = float(distances_r2s[:, 0].mean())
        privacy_metrics['nn_dist_real_to_synth_min'] = float(distances_r2s[:, 0].min())
        privacy_metrics['nn_dist_real_to_synth_std'] = float(distances_r2s[:, 0].std())

        # Re-identification rate (very close matches)
        threshold_reid = 0.01
        reid_count = (distances_s2r[:, 0] < threshold_reid).sum()
        privacy_metrics['reidentification_rate'] = float(reid_count / len(X_synth))
        privacy_metrics['reidentification_count'] = int(reid_count)

        # Uniqueness/Diversity
        unique_synth = len(np.unique(X_synth_scaled, axis=0))
        privacy_metrics['unique_synthetic_rows'] = int(unique_synth)
        privacy_metrics['uniqueness_ratio'] = float(unique_synth / len(X_synth))

        # Distance distribution divergence
        ks_stat, ks_pval = stats.ks_2samp(distances_s2r[:, 0], distances_r2s[:, 0])
        privacy_metrics['distance_distribution_divergence'] = float(ks_stat)
        privacy_metrics['distance_distribution_pvalue'] = float(ks_pval)

        if verbose:
            print(f"   [METRIC] Memorization Risk: {privacy_metrics['memorization_risk']:.3f}")
            print(f"   [METRIC] Re-identification Rate: {privacy_metrics['reidentification_rate']:.3f}")
            print(f"   [METRIC] Uniqueness Ratio: {privacy_metrics['uniqueness_ratio']:.3f}")

    except Exception as e:
        privacy_metrics['error'] = str(e)
        if verbose:
            print(f"   [ERROR] Privacy calculation failed: {e}")

    return privacy_metrics
```

**Integrate into Section 5 notebooks**:

After TRTS analysis for each model:
```python
# Calculate privacy metrics
from src.evaluation.privacy import calculate_privacy_metrics

privacy_results = calculate_privacy_metrics(
    real_data=data_processed,
    synthetic_data=synthetic_data_dict[model_name],
    target_column=TARGET_COLUMN
)

# Store in results
trts_results_dict[model_name]['privacy_metrics'] = privacy_results

# Save privacy metrics CSV
privacy_data = []
for model_name, results in trts_results_dict.items():
    if 'privacy_metrics' in results:
        row = {'Model': model_name}
        row.update(results['privacy_metrics'])
        privacy_data.append(row)

privacy_df = pd.DataFrame(privacy_data)
privacy_df.to_csv(f'{results_path}/privacy_metrics.csv', index=False)
print(f"[SAVED] privacy_metrics.csv")
```

**Update setup.py**:
```python
from src.evaluation.privacy import *
```

**Success Criteria**:
- [ ] privacy_metrics.csv created
- [ ] Contains NN distances, memorization risk, re-identification rate
- [ ] All 7 notebooks generate CSV
- [ ] Performance <2 min per model

**Files Modified**:
- NEW: `src/evaluation/privacy.py`
- MODIFIED: `setup.py` (add import)
- MODIFIED: All 7 notebooks Section 5 (add privacy calculation)

**Dependencies**: Task 0.1

---

### TASK 3.7: Privacy Visualizations

**Priority**: P2

**Problem**: Privacy metrics need visual dashboard.

**Approach**: Add visualizations to src/visualization/section5.py

**Implementation**:

**Add to src/visualization/section5.py**:
```python
def create_privacy_dashboard(privacy_df, results_path, verbose=True):
    """
    Create comprehensive privacy metrics dashboard.

    Parameters:
    -----------
    privacy_df : pd.DataFrame
        Privacy metrics for all models
    results_path : str
        Directory to save output

    Returns:
    --------
    str : Path to saved file
    """
    if privacy_df.empty or 'Model' not in privacy_df.columns:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Privacy Analysis Dashboard - All Models', fontsize=16, fontweight='bold')

    # Top-Left: Memorization Risk
    axes[0, 0].barh(privacy_df['Model'], privacy_df['memorization_risk'],
                    color=['red' if x > 0.1 else 'orange' if x > 0.05 else 'green'
                           for x in privacy_df['memorization_risk']])
    axes[0, 0].axvline(x=0.05, color='orange', linestyle='--', label='Caution')
    axes[0, 0].axvline(x=0.1, color='red', linestyle='--', label='High Risk')
    axes[0, 0].set_xlabel('Memorization Risk (lower better)')
    axes[0, 0].set_title('Memorization Risk by Model')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # Top-Right: Re-identification Rate
    axes[0, 1].barh(privacy_df['Model'], privacy_df['reidentification_rate'],
                    color=['red' if x > 0.05 else 'orange' if x > 0.01 else 'green'
                           for x in privacy_df['reidentification_rate']])
    axes[0, 1].set_xlabel('Re-identification Rate (lower better)')
    axes[0, 1].set_title('Re-identification Risk')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # Bottom-Left: Uniqueness Ratio
    axes[1, 0].barh(privacy_df['Model'], privacy_df['uniqueness_ratio'],
                    color=['red' if x < 0.7 else 'orange' if x < 0.9 else 'green'
                           for x in privacy_df['uniqueness_ratio']])
    axes[1, 0].axvline(x=0.9, color='green', linestyle='--', label='High diversity')
    axes[1, 0].set_xlabel('Uniqueness Ratio (higher better)')
    axes[1, 0].set_title('Synthetic Data Diversity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Bottom-Right: NN Distance Comparison
    x = np.arange(len(privacy_df))
    width = 0.35
    axes[1, 1].bar(x - width/2, privacy_df['nn_dist_synth_to_real_mean'],
                   width, label='Synth→Real', alpha=0.8)
    axes[1, 1].bar(x + width/2, privacy_df['nn_dist_real_to_synth_mean'],
                   width, label='Real→Synth', alpha=0.8)
    axes[1, 1].set_ylabel('Mean NN Distance')
    axes[1, 1].set_title('Nearest Neighbor Distance Distribution')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(privacy_df['Model'], rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = f"{results_path}/privacy_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[VIZ] Saved: privacy_dashboard.png")

    return output_path
```

**Integrate into Section 5 notebooks**:

After privacy_metrics.csv created:
```python
# Create privacy dashboard
from src.visualization.section5 import create_privacy_dashboard
create_privacy_dashboard(privacy_df, results_path)
```

**Success Criteria**:
- [ ] 4-panel dashboard created
- [ ] Color-coded risk levels
- [ ] File saved to Section-5 folder
- [ ] All 7 notebooks generate dashboard

**Files Modified**:
- MODIFIED: `src/visualization/section5.py` (add function)
- MODIFIED: All 7 notebooks Section 5 (add dashboard call)

**Dependencies**: Task 3.6

---

### TASK 3.8: Early Stopping in Objective Functions

**Priority**: P3

**Problem**: Hyperparameter trials run unnecessarily long.

**Approach**: Migrate objective function to src/objective/functions.py, add Optuna pruning.

**Implementation**:

**Step 1: Migrate to src/objective/functions.py**:

Copy `enhanced_objective_function_v2()` from setup.py.

**Step 2: Add pruning support**:
```python
def enhanced_objective_function_v2(real_data, synthetic_data, target_column,
                                  verbose=False, trial=None):
    """
    Enhanced objective function with early stopping support.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object for pruning (optional)
    ... other params ...
    """
    # ... existing code ...

    # Report intermediate progress for pruning
    if trial:
        # After statistical similarity (10%)
        trial.report(stat_similarity * 0.1, step=50)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # After JS similarity (30%)
        partial_score = stat_similarity * 0.1 + js_similarity * 0.2
        trial.report(partial_score, step=100)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # After correlation (50%)
        partial_score += corr_preservation * 0.2
        trial.report(partial_score, step=200)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # ... continue with TRTS ...

    return final_score
```

**Step 3: Update Section 4 in notebooks**:

Add pruner to study creation:
```python
import optuna

# Create study with median pruner
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,   # Don't prune first 5 trials
        n_warmup_steps=50,    # Don't prune until step 50
        interval_steps=10     # Check every 10 steps
    )
)

# Objective wrapper
def objective(trial):
    # ... parameter sampling ...

    # Call enhanced function with trial
    score = enhanced_objective_function_v2(
        real_data=train_data,
        synthetic_data=synthetic_samples,
        target_column=TARGET_COLUMN,
        verbose=False,
        trial=trial  # Pass for pruning
    )

    return score

# Optimize with verbose progress
study.optimize(objective, n_trials=50, timeout=3600,
               callbacks=[lambda study, trial: print(f"Trial {trial.number}: {trial.value:.4f}")])
```

**Step 4: Update setup.py**:
```python
from src.objective.functions import *
```

**Success Criteria**:
- [ ] Pruning reduces time by 20-40%
- [ ] Best trial quality comparable (within 5%)
- [ ] Works across all 6 models
- [ ] Pruned trials logged correctly

**Files Modified**:
- NEW: `src/objective/functions.py` (migrate from setup.py)
- MODIFIED: `setup.py` (remove function, add import)
- MODIFIED: All 7 notebooks Section 4 (add pruner to study)

**Dependencies**: Task 0.1

---

## PHASE 4: FINALIZATION

**Goal**: Standardize Optuna, update docs, complete migration
**Duration**: 2-3 days
**Dependencies**: Phases 0-3 complete

---

### TASK 4.1: Standardize Optuna Implementation with Visualizations

**Priority**: P2

**Problem**: Optuna usage varies across model optimizations. Need consistent viz exports.

**Approach**: Add Optuna visualization exports after each optimization in Section 4.

**Implementation**:

**Add to all 6 model optimizations in Section 4**:

After `study.optimize()` completes:
```python
# Generate Optuna visualization exports
import optuna.visualization as vis

try:
    # Optimization history
    fig1 = vis.plot_optimization_history(study)
    fig1.write_image(f'{results_path}/optim_history_{MODEL_NAME}.png')

    # Parameter importance
    fig2 = vis.plot_param_importances(study)
    fig2.write_image(f'{results_path}/param_importance_{MODEL_NAME}.png')

    # Parallel coordinate plot
    fig3 = vis.plot_parallel_coordinate(study, params=list(study.best_params.keys())[:5])
    fig3.write_image(f'{results_path}/parallel_coord_{MODEL_NAME}.png')

    print(f"[VIZ] Saved Optuna visualizations for {MODEL_NAME}")
except Exception as e:
    print(f"[WARNING] Optuna viz failed: {e}")
```

**Success Criteria**:
- [ ] All 6 models generate 3 Optuna plots
- [ ] Plots saved to Section-4 results folder
- [ ] Consistent implementation across models
- [ ] All 7 notebooks updated

**Files Modified**:
- MODIFIED: All 7 notebooks Section 4 (add viz exports for each model)

**Dependencies**: Task 3.8 (needs objective function migrated)

---

### TASK 4.2: Update Documentation for Phase 3 Metrics

**Priority**: P2

**Problem**: Phase 1 READMEs described forward-compatible structure. Now update with actual Phase 3 details.

**Approach**: Update README generation functions with Phase 3 metrics.

**Implementation**:

**Update src/utils/documentation.py**:

Section 3 README already includes mode_collapse and MI columns (forward-compatible from Phase 1).

Verify and add any missing details:
- Confirm mode_collapse_analysis.csv columns documented
- Confirm privacy_metrics.csv reference added
- Update interpretation sections if needed

**Update Section 5 README** (add new function):
```python
def create_section5_readme(results_path, dataset_id, timestamp):
    """Create README for Section 5 outputs"""
    readme_content = f"""# Section 5: Final Model Comparison - Output Files

## CSV Files

- **trts_detailed_results.csv**: Comprehensive classification metrics for all TRTS scenarios
  - Columns: Model, Scenario, Accuracy, F1_Score, Precision, Recall, Sensitivity, Specificity, NPV, FPR, FNR, Balanced_Accuracy, MCC, Cohen_Kappa, AUROC, AUPRC, Average_Precision, Training_Time_Sec

- **privacy_metrics.csv**: Privacy assessment for all models
  - Columns: Model, nn_dist_synth_to_real_mean, memorization_risk, reidentification_rate, uniqueness_ratio, etc.

## Visualizations

- **trts_comprehensive_analysis.png**: 4-panel TRTS performance comparison
- **privacy_dashboard.png**: 4-panel privacy metrics comparison

---
*Dataset: {dataset_id}*
*Date: {timestamp}*
"""

    readme_path = f"{results_path}/README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"[DOCS] Created: Section-5/README.md")
    return readme_path
```

**Add to Section 5 in all notebooks**:
```python
from src.utils.documentation import create_section5_readme
create_section5_readme(results_path, DATASET_IDENTIFIER, SESSION_TIMESTAMP)
```

**Success Criteria**:
- [ ] Section 3 README accurate for Phase 3 metrics
- [ ] Section 5 README created
- [ ] All CSV columns documented
- [ ] All 7 notebooks updated

**Files Modified**:
- MODIFIED: `src/utils/documentation.py` (add Section 5 function)
- MODIFIED: All 7 notebooks Section 5 (add README generation)

**Dependencies**: Tasks 3.1-3.7 (needs all metrics implemented)

---

### TASK 4.3: Complete setup.py Migration

**Priority**: P3

**Problem**: Some untouched functions may still be in setup.py.

**Approach**: Move remaining CHUNK functions to appropriate src/ modules.

**Implementation**:

**Step 1: Audit setup.py**:

Identify any remaining functions not yet migrated:
- Data preprocessing functions → `src/data/preprocessing.py`
- Model factory functions → `src/models/factory.py`
- Any utility functions → appropriate `src/utils/` modules

**Step 2: Move to modules, update imports**

**Step 3: Final setup.py should be <100 lines**:

Just re-exports from src/ modules.

**Success Criteria**:
- [ ] setup.py is <100 lines (just imports/re-exports)
- [ ] All CHUNK functions migrated to src/
- [ ] All 7 notebooks still work with `from setup import *`
- [ ] No broken imports

**Files Modified**:
- MODIFIED: `setup.py` (final cleanup)
- NEW/MODIFIED: Various `src/` modules (move remaining functions)

**Dependencies**: All Phase 0-3 tasks complete

---

## 4. IMPLEMENTATION CHECKLIST & VERIFICATION

### Phase 0 Verification
- [ ] src/ directory structure created
- [ ] Core modules exist (config.py, utils/, __init__.py)
- [ ] setup.py successfully re-exports from src/
- [ ] All 7 notebooks import successfully

### Phase 1 Verification
- [ ] README.md in all Section-2 folders
- [ ] README.md in all Section-3 folders
- [ ] Column definitions accurate
- [ ] Forward-compatible for Phase 3

### Phase 2 Verification
- [ ] Correlation heatmap: dynamic fonts working
- [ ] Feature distributions: multi-file splitting working
- [ ] TRTS y-axis: no label overlap
- [ ] Section 2/3 viz moved to src/visualization/
- [ ] All 7 notebooks tested

### Phase 3 Verification
- [ ] Mode collapse detection working
- [ ] Mode collapse visualization created
- [ ] MI metrics calculated
- [ ] MI visualization created
- [ ] TRTS: 15+ metric columns
- [ ] Privacy metrics CSV created
- [ ] Privacy dashboard created
- [ ] Early stopping reduces time

### Phase 4 Verification
- [ ] Optuna viz exports for all 6 models
- [ ] Documentation updated for Phase 3
- [ ] setup.py <100 lines
- [ ] All functions migrated to src/

### Final Integration Test
- [ ] Full run: Alzheimer dataset, Sections 1-5
- [ ] Verify all new files generated
- [ ] Check CSV columns match specs
- [ ] Visual inspection of all plots
- [ ] Review git diff

---

## 5. REFERENCE MATERIALS

### 5.1 Testing Commands

**Display plot in notebook**:
```python
from IPython.display import Image, display
display(Image(f'{results_path}/correlation_heatmap.png'))
```

**Check CSV structure**:
```python
df = pd.read_csv(f'{results_path}/evaluation_summary.csv')
print(df.columns.tolist())
print(df.head())
```

**Verify module imports**:
```python
from setup import *
print(f"SESSION_TIMESTAMP: {SESSION_TIMESTAMP}")
print(f"create_correlation_heatmap: {create_correlation_heatmap}")
```

### 5.2 Git Workflow

```bash
# Create feature branch
git checkout -b feature/incremental-refactor

# Commit Phase 0
git add src/ setup.py
git commit -m "feat(phase-0): Create modular src/ architecture

- Create src/ directory structure with modules
- Setup thin re-export layer in setup.py
- Migrate core utilities (config, paths)
- All notebooks tested and working"

# Commit by phase
git add src/utils/documentation.py
git commit -m "feat(phase-1): Add README generation for Section 2 and 3"

git add src/visualization/section2.py
git commit -m "feat(phase-2): Implement dynamic correlation heatmap"

# Push and create PR
git push -u origin feature/incremental-refactor
```

---

## 6. TASK DEPENDENCY GRAPH

```
PHASE 0: Foundation
└── Task 0.1 (Create src/ structure) ────┐
                                          │
PHASE 1: Documentation                    │
├── Task 1.1 (Section 2 README) ──────────┤
└── Task 1.2 (Section 3 README) ──────────┤
                                          │
PHASE 2: Visualizations                   │
├── Task 2.1 (Correlation heatmap) ───────┤
├── Task 2.2 (Feature distributions) ─────┤
├── Task 2.3 (TRTS y-axis) ───────────────┤
├── Task 2.4 (Correlation comparison) ────┤
└── Task 2.5 (Distribution comparison) ───┤
                                          │
PHASE 3: Advanced Metrics                 │
├── Task 3.1 (Mode collapse detect) ──────┤
├── Task 3.2 (Mode collapse viz) ─── 3.1 ─┤
├── Task 3.3 (MI metrics) ────────────────┤
├── Task 3.4 (MI viz) ────────────── 3.3 ─┤
├── Task 3.5 (Comprehensive TRTS) ────────┤
├── Task 3.6 (Privacy metrics) ───────────┤
├── Task 3.7 (Privacy viz) ────────── 3.6 ─┤
└── Task 3.8 (Early stopping) ────────────┤
                                          │
PHASE 4: Finalization                     │
├── Task 4.1 (Optuna standard) ─── 3.8 ───┤
├── Task 4.2 (Update docs) ─────── 3.1-3.7┤
└── Task 4.3 (Complete migration) ── ALL ─┘
```

**Recommended Execution**: Proceed phase by phase, task by task.

**Estimated Total Time**: 12-15 days

---

## 7. SUCCESS METRICS

### Phase 0 Success
- src/ architecture established
- setup.py thin re-export layer working
- All notebooks compatible

### Phase 1 Success
- 100% output folders have README.md
- Documentation forward-compatible
- Column definitions accurate

### Phase 2 Success
- Dynamic fonts for all heatmaps
- Multi-file splitting for large datasets
- No visualization overlap issues
- Code moved to src/visualization/

### Phase 3 Success
- Mode collapse detection: 100% TP, <5% FP
- TRTS: ≥15 metric columns
- Privacy metrics: ≥8 metrics per model
- MI calculated correctly
- Visualizations for all new metrics
- Early stopping: 20-40% time reduction

### Phase 4 Success
- Optuna viz for all 6 models
- Documentation complete and accurate
- setup.py <100 lines
- All code in proper src/ modules

### Overall Success
- [ ] All 18 tasks completed
- [ ] All 7 notebooks work identically
- [ ] Full pipeline runs on 2+ datasets
- [ ] No regression in existing functionality
- [ ] Code follows SOLID principles
- [ ] ~60% reduction in setup.py size (3814 → ~100 lines)

---

**END OF EXECUTION PLAN**

**Ready for Phase 0 implementation!**
