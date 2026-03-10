---
title: "Project Evolution"
output: html_document
date: "2026-01-23"
---

# Clinical Synthetic Data Generation Framework - Project Evolution

**A Timeline of Development Across Branches**

---

## Executive Summary

This document traces the evolution of the Clinical Synthetic Data Generation Framework from its initial proof-of-concept through multiple rounds of AWS deployment refinement. The project has progressed through four major development phases, each represented by distinct Git branches that capture significant architectural and methodological improvements.

**Branch Overview:**

- **main** (Current) - Production branch with 8-model architecture and modular design
- **v1.0-old-main** (tag) - Archived: Initial successful local execution (Aug 2025)
- **v2.0-aws-round1** (tag) - Archived: First AWS deployment (Sep-Oct 2025)
- **v3.0-legacy-main** (tag) - Archived: 3-model config, stabilized release (Oct-Nov 2025)

---

## Branch Comparison Matrix

| Aspect | v1.0-old-main (tag) | v2.0-aws-round1 (tag) | v3.0-legacy-main (tag) | main (current) |
|--------|---------------------|----------------------|------------------------|----------------|
| **Status** | Archived (tag) | Archived (tag) | Archived (tag) | **Active** |
| **Environment** | Local Laptop | AWS SageMaker | AWS SageMaker | AWS SageMaker |
| **Models Supported** | 6 (Original) | 3 (CTGAN, CopulaGAN, TVAE) | 3 (CTGAN, CopulaGAN, TVAE) | 8 (All + PATE-GAN, MEDGAN) |
| **Datasets** | 4 | 4 | 4 | 4 |
| **Architecture** | Monolithic | Monolithic | Monolithic | Modular (src/) |
| **setup.py Size** | ~3,691 lines | ~3,500 lines | ~3,500 lines | 209 lines (94% reduction) |
| **Evaluation Metrics** | Basic | Enhanced | Enhanced | 30+ comprehensive |
| **Automation** | Manual | Semi-automated | Semi-automated | Fully automated |
| **Visualization** | Basic | Enhanced | Enhanced | Advanced (ROC/PR/Calibration) |
| **Code Quality** | Prototype | Production | Production | Enterprise-grade |

---

## Timeline of Major Milestones

### Phase 0: Genesis 

**Branch:** N/A (Pre-version control)

Initial research and prototyping phase:
- Evaluated synthetic data generation methods for clinical data
- Selected 6 GAN-based approaches for comprehensive benchmarking
- Established proof-of-concept with single dataset

---

### Phase 1: Local Success - **old-main**
**Timeline:** August 2025
**Environment:** Local Laptop
**Status:** First Complete Working Pipeline ✅

#### Key Achievement
**First successful end-to-end execution of all 6 synthetic data generation methods on local hardware.**

#### Major Commits
```
bc05e73 | 2025-08-29 | all running through end of section 5!
e592ab1 | 2025-08-28 | all working through end of section 5.3
d9cf492 | 2025-08-27 | working through end of 5.2
cc2e6d8 | 2025-08-27 | working through end of 5.1 with classification metrics
```

#### Technical Highlights

- ✅ **All 6 models working:** CTGAN, CTAB-GAN, CTAB-GAN+, GANerAid, CopulaGAN, TVAE
- ✅ **Complete 5-section pipeline:** Setup → Preprocessing → Configuration → Optimization → Evaluation
- ✅ **4 healthcare datasets:** Alzheimer's, Breast Cancer, Liver Disease, Pakistani Liver
- ✅ **Hyperparameter optimization:** Optuna-based Bayesian optimization
- ✅ **Basic evaluation metrics:** Statistical fidelity, utility preservation

#### Challenges Overcome

- GANerAid divisibility constraints with dataset sizes
- Section 4 variable scoping issues (locals() → globals())
- DataFrame dtype errors in performance visualization
- Index out-of-bounds errors with small datasets

#### Limitations

- ⚠️ Local hardware constraints (10+ hours per full run)
- ⚠️ No modular architecture
- ⚠️ Limited scalability
- ⚠️ Manual visualization generation

**Why Archived:** Successfully proven concept, but needed cloud infrastructure for practical deployment.

---

### Phase 2: AWS Migration - **AWS_Round1**

**Timeline:** September - October 2025
**Environment:** AWS SageMaker
**Status:** Cloud Deployment Foundation ✅

#### Key Achievement

**Successfully migrated framework to AWS SageMaker with enterprise-grade dependency management.**

#### Major Commits
```
6479ec2 | 2025-12-05 | adding dev-plans
2241feb | 2025-12-05 | First round of changes from AWS SageMaker
b6a2a6a | 2025-10-26 | docs: Add comprehensive installation instructions to README
588a55e | 2025-10-26 | fix: Add critical missing dependencies found by QC
ef7cff4 | 2025-10-26 | feat: Add curated requirements.txt for AWS deployment
76778bd | 2025-09-24 | feat: Complete chunk harmonization and add comprehensive README
```

#### Technical Highlights

- ✅ **AWS SageMaker deployment:** Successful cloud migration
- ✅ **Dependency management:** Curated requirements.txt with exact versions
- ✅ **Multi-dataset framework:** Harmonized notebook structure across all 4 datasets
- ✅ **Chunk ID standardization:** `CHUNK_{Major}_{Minor}_{Patch}_{Seq}` naming scheme
- ✅ **Comprehensive documentation:** README with installation instructions

#### Critical Discovery: The "3-Model Limitation"

**Due to corporate AWS environment restrictions, only 3 of 6 models could be deployed:**

- ✅ **CTGAN** - Standard GAN approach
- ✅ **CopulaGAN** - Statistical copula-based approach
- ✅ **TVAE** - Variational autoencoder approach
- ❌ **CTAB-GAN** - Blocked by dependency conflicts
- ❌ **CTAB-GAN+** - Blocked by dependency conflicts
- ❌ **GANerAid** - Blocked by custom implementation issues

#### AWS-Specific Fixes

- Pinned exact package versions for reproducibility
- Resolved sklearn 1.0+ compatibility issues
- Fixed CopulaGAN importlib warnings
- Addressed pandas/numpy dtype inconsistencies
- Implemented MICE imputation for missing values

#### Challenges Overcome
```
780b517 | 2025-09-18 | Fix CHUNK_024 CTAB-GAN+ error: Replace .isinf() with np.isinf()
66cc468 | 2025-09-18 | Comprehensive Section 2 pipeline fix
62730e2 | 2025-09-18 | Fix Section 4 categorical data preprocessing
```

**Why Archived:** Established AWS deployment methodology, but 3-model limitation required architectural rethinking for AWS_Round2.

---

### Phase 3: Production Stabilization - **main**

**Timeline:** October - November 2025
**Environment:** AWS SageMaker
**Status:** Current Production Release 🟢

#### Key Achievement

**Stable, production-ready release optimized for AWS SageMaker with 3-model configuration.**

#### Major Commits
```
5ee7e98 | 2025-11-06 | Merge branch 'main' of https://github.com/gcicc/tableGenCompare
4968e07 | 2025-11-06 | fix: Pin exact package versions for AWS SageMaker compatibility
b6e4522 | 2025-11-05 | Add AWS setup guide for CTAB-GAN and Deep Tabular GANs
```

#### Technical Highlights

- ✅ **Production stability:** Thoroughly tested on AWS SageMaker
- ✅ **Version pinning:** Exact dependency versions for reproducibility
- ✅ **Comprehensive documentation:** Setup guides and troubleshooting
- ✅ **Quality assurance:** Multi-dataset validation
- ✅ **3-model optimization:** CTGAN, CopulaGAN, TVAE fully functional

#### Use Cases

- **Baseline comparisons** for new synthetic data methods
- **Production deployment** for clinical data synthesis
- **Teaching/demonstration** of synthetic data generation
- **Reproducible research** with exact environment specification

**Current Status:** Active production branch, suitable for stable AWS deployments.

---

### Phase 4: Next-Generation Architecture - **AWS_Round2**

**Timeline:** December 2025 - 

**Environment:** AWS SageMaker

**Status:** Active Development 🚀

#### Key Achievement

**Complete architectural overhaul with modular design, comprehensive evaluation, and automated workflows.**

This phase represents a fundamental reimagining of the framework based on lessons learned from AWS_Round1 and production experience with main.

---

### 🎯 AWS_Round2: The Modular Revolution

#### A. Modular Architecture (Dec 6-12, 2025)

**The Problem:** 3,691-line monolithic setup.py was unmaintainable and difficult to test.

**The Solution:** Complete modularization into specialized components.

##### Major Commits
```
345045d | 2025-12-12 | feat(phase-4): Complete setup.py streamlining and modular architecture migration
4efda9a | 2025-12-11 | feat(phase-4): Migrate objective functions to src/objective
527890e | 2025-12-11 | feat(phase-3): Migrate evaluation functions to src/evaluation
ea643b7 | 2025-12-11 | feat(phase-2): Migrate data preprocessing functions to src/data
aee5bf1 | 2025-12-06 | feat(phase-1): Migrate model classes and imports to src/models
758606e | 2025-12-06 | feat(phase-0): Create modular src/ architecture
```

##### New Architecture
```
tableGenCompare/
├── setup.py (209 lines - 94.3% reduction!)
│   └── Thin backward-compatible re-export layer
│
└── src/
    ├── config.py              # Session management
    ├── compat.py              # Backward compatibility
    ├── models/                # Model implementations
    │   ├── base_model.py      # Base model interface
    │   └── implementations/   # CTGAN, CTAB-GAN, GANerAid, etc.
    ├── data/                  # Data preprocessing
    │   ├── cleaning.py        # MICE imputation, encoding
    │   ├── summary.py         # Data summaries
    │   └── visualization.py   # Data viz
    ├── evaluation/            # Evaluation framework
    │   ├── quality.py         # Statistical fidelity
    │   ├── trts.py            # TRTS framework
    │   ├── batch.py           # Batch evaluation
    │   └── hyperparameters.py # Optuna integration
    ├── objective/             # Objective functions
    │   └── enhanced.py        # Multi-objective optimization
    ├── visualization/         # Advanced visualizations
    │   ├── section4.py        # Optuna viz
    │   └── section5.py        # TRTS viz
    └── utils/                 # Utilities
        ├── paths.py           # Path management
        ├── parameters.py      # Parameter handling
        └── documentation.py   # Auto-documentation
```

##### Benefits

- ✅ **94.3% code reduction** in setup.py (3,691 → 209 lines)
- ✅ **Zero notebook changes** - full backward compatibility
- ✅ **Improved testability** - modular components
- ✅ **Enhanced maintainability** - clear separation of concerns
- ✅ **Better scalability** - easy to extend

---

#### B. Comprehensive Evaluation Suite (Dec 6-15, 2025)

**The Problem:** Basic metrics insufficient for robust evaluation of synthetic data quality.

**The Solution:** 30+ metric comprehensive evaluation framework.

##### Major Commits
```
1698ba7 | 2025-12-15 | feat(trts): Add comprehensive TRTS visualization suite with 30+ metrics
625135e | 2025-12-15 | feat(trts): Add comprehensive 30-metric TRTS evaluation for Sections 3 & 5
161578f | 2025-12-06 | feat(privacy): add privacy risk analysis and dashboard
a462541 | 2025-12-06 | feat(trts): expand to 15+ comprehensive classification metrics
8dfeda8 | 2025-12-06 | feat(phase-3): Add mutual information metrics and visualization
262a406 | 2025-12-06 | feat(phase-3): Add mode collapse detection and visualization
```

##### Evaluation Framework

**1. TRTS Analysis (Train Real/Test Synthetic Framework)**

- 4 scenarios: TRTR, TRTS, TSTR, TSTS
- 30+ classification metrics per scenario:
  - **Basic:** accuracy, balanced_accuracy, precision, recall, F1
  - **Advanced:** F-beta, specificity, sensitivity, TPR, TNR, NPV, PPV
  - **Error rates:** FPR, FNR, FDR, FOR
  - **Statistical:** MCC, Cohen's Kappa, Youden's J, FMI
  - **Probabilistic:** AUROC, AUPRC, Brier Score
  - **Population:** prevalence, predicted_positive_rate

**2. Privacy Risk Assessment**

- **DCR (Distance to Closest Record):** Measures memorization risk
- **NNDR (Nearest Neighbor Distance Ratio):** Detects overfitting
- **Memorization Score:** Percentage of exact replicas
- **Re-identification Risk:** Probability of identity disclosure
- **Privacy Score:** Composite privacy metric (0-1 scale)

**3. Statistical Fidelity**

- Distribution similarity (KS test, Wasserstein distance)
- Correlation structure preservation
- PCA comparison with outcome color-coding
- Mutual information metrics
- Mode collapse detection

**4. Utility Preservation**

- Cross-accuracy (Real→Synth, Synth→Real)
- Downstream task performance
- Feature importance alignment
- Prediction consistency

##### Advanced Visualizations
```
77c04f3 | 2025-12-15 | feat(batch): Auto-generate Optuna visualizations during batch evaluation
9310e16 | 2025-12-06 | feat(phase-3): add training loss visualization (Task 3.9)
```

**Automated Visualization Suite:**

- **ROC Curves:** 2×2 grid for all TRTS scenarios
- **Precision-Recall Curves:** Performance across thresholds
- **Calibration Curves:** Probability calibration assessment
- **Privacy Dashboard:** 4-panel privacy risk visualization
- **Optuna Visualizations:**
  - Optimization history
  - Parameter importance
  - Parallel coordinate plots
  - Multi-model summary comparison

---

#### C. Workflow Automation (Dec 15, 2025)

**The Problem:** Manual visualization generation was error-prone and time-consuming.

**The Solution:** Fully automated batch processing with integrated visualization.

##### Major Commits
```
77c04f3 | 2025-12-15 | feat(batch): Auto-generate Optuna visualizations during batch evaluation
70e2a66 | 2025-12-15 | refactor(notebooks): Remove redundant manual Optuna visualization cells
```

##### Automation Features

**1. Batch Evaluation Pipeline**
```python
# Single function call now handles everything!
results = evaluate_all_available_models(
    section_number=3,
    scope=globals()
)
```

**What it automatically does:**

- ✅ Detects all trained models in notebook scope
- ✅ Runs comprehensive quality evaluation
- ✅ Performs 30+ metric TRTS analysis
- ✅ Generates privacy risk assessment
- ✅ Creates all visualizations (ROC/PR/Calibration/Privacy)
- ✅ Auto-detects Optuna studies
- ✅ Generates Optuna visualizations (3 per model + summary)
- ✅ Saves all outputs to correct section directories
- ✅ Returns comprehensive results dictionary

**2. Section-Aware Organization**

- Section 3 results → `results/{dataset}/Section-3/`
- Section 4 results → `results/{dataset}/Section-4/` (Optuna viz)
- Section 5 results → `results/{dataset}/Section-5/`

**3. Notebook Cleanup**

- **432 lines removed** across 7 notebooks
- Eliminated redundant manual visualization code
- Streamlined workflow

---

#### D. Bug Fixes and Quality Improvements (Dec 16, 2025)

##### Major Commits
```
2618daf | 2025-12-16 | fix(data): Remove corrupted Unicode characters from summary.py
```

**Issues Resolved:**

- ✅ Fixed corrupted Unicode characters causing `SyntaxError: null bytes`
- ✅ Resolved Section 2 preprocessing failures
- ✅ Fixed privacy key mismatch in batch evaluation
- ✅ Corrected return value handling for privacy dashboard
- ✅ Cleaned up unnecessary imports

---

### AWS_Round2 Summary Statistics

#### Code Metrics

- **setup.py reduction:** 3,691 → 209 lines (94.3%)
- **New modular files:** 20+ specialized modules
- **Lines removed from notebooks:** 432 lines (automation)
- **Commits in AWS_Round2:** 25+ major feature commits

#### Evaluation Enhancements

- **Metrics:** 5 basic → 30+ comprehensive
- **TRTS scenarios:** 4 comprehensive scenarios
- **Privacy metrics:** 5 new privacy risk measures
- **Visualizations:** 8+ auto-generated charts per evaluation

#### Automation Improvements

- **Manual steps eliminated:** 6 visualization steps per model
- **File generation:** 10+ files per batch evaluation
- **Section organization:** Auto-routed to correct directories

#### Files Generated Per Evaluation

**Section 3 (Quality):**

- statistical_similarity.csv
- pca_comparison_with_outcome.png
- distribution_comparison.png
- correlation_comparison.png
- evaluation_summary.csv

**Section 4 (Optuna):**

- optim_history_{model}.png (×6 models)
- param_importance_{model}.png (×6 models)
- parallel_coord_{model}.png (×6 models)
- optuna_summary_all_models.png

**Section 5 (TRTS):**

- trts_comprehensive_analysis.png
- trts_summary_metrics.csv
- trts_detailed_results.csv
- privacy_dashboard.png
- privacy_summary.csv
- trts_roc_curves.png
- trts_pr_curves.png
- trts_calibration_curves.png

**Total:** 30+ files per full evaluation run, all automatically organized!

---

### Phase 5: Model Expansion and Quality Refinements - **AWS_Round2** (continued)

**Timeline:** January 2026

**Environment:** AWS SageMaker

**Status:** Active Development 🚀

#### Key Achievement

**Expanded model support to 8 generative models with improved quality metrics and consolidated workflow.**

---

#### A. New Model Implementations (Jan 2026)

**PATE-GAN and MEDGAN added to the framework, bringing total model count to 8.**

##### PATE-GAN (Private Aggregation of Teacher Ensembles)
- Privacy-preserving GAN with differential privacy guarantees
- Teacher ensemble approach for privacy-utility tradeoff
- Integrated into Section 4 batch hyperparameter optimization

##### MEDGAN (Medical GAN)
- Autoencoder-based GAN for medical record generation
- Pre-training phase followed by GAN training
- Designed specifically for discrete medical features

##### Integration Work
- Added to model factory with full hyperparameter search spaces
- Included in batch training pipeline
- Added to comprehensive evaluation suite

---

#### B. Quality Score Improvements (Jan 2026)

**Denormalization fixes for accurate quality score calculation.**

- Fixed denormalization step in quality scoring
- Ensured synthetic data properly transformed back to original scale
- Improved statistical fidelity metrics accuracy
- Corrected distribution comparison calculations

---

#### C. Batch Training Enhancements (Jan 2026)

**Improved batch optimization workflow for all 8 models.**

- Enhanced model factory for consistent instantiation
- Improved Optuna study management per model
- Better error handling and recovery during batch runs
- Results persistence to EBS for session recovery

---

#### D. Visualization and EDA Improvements (Jan 2026)

##### Categorical Visualization
- Added categorical variable distribution plots to EDA
- Improved handling of mixed categorical/numerical datasets
- Enhanced Section 2 preprocessing visualizations

##### Kaleido Fallback for Optuna Plots
- Implemented fallback rendering when Kaleido unavailable
- Ensures Optuna visualizations work across environments
- Graceful degradation for headless servers

##### Privacy Dashboard Enhancements
- Improved 4-panel privacy risk visualization
- Better DCR and NNDR calculation
- Enhanced memorization risk detection

---

#### E. Dependency Updates (Jan 2026)

**Critical package updates for compatibility and bug fixes.**

| Package | Old Version | New Version | Reason |
|---------|-------------|-------------|--------|
| dython | 0.6.8 | 0.7.12 | GANERAID compatibility fix |
| scikit-learn | 1.2.2 | 1.7.2 | Updated API support |
| sdv | 1.24.1 | 1.32.1 | New features and fixes |
| copulas | 0.12.3 | 0.14.0 | Dependency alignment |
| rdt | 1.17.1 | 1.19.0 | SDV compatibility |
| Faker | 37.5.3 | 40.1.2 | Bug fixes |
| plotly | 6.2.0 | 6.4.0 | Visualization improvements |

---

#### F. STG-Driver Notebook Consolidation (Jan 2026)

**Consolidated workflow into primary STG-Driver-breast-cancer.ipynb notebook.**

- Single entry point for complete pipeline execution
- All 8 models available in one notebook
- Streamlined Section 4 batch training interface
- Improved documentation and inline comments

---

### Phase 5 Summary Statistics

#### Model Expansion
- **Models:** 6 → 8 (added PATE-GAN, MEDGAN)
- **Privacy models:** 1 (PATE-GAN with differential privacy)

#### Quality Improvements
- Denormalization fixes for accurate metrics
- Categorical visualization support
- Kaleido fallback implementation

#### Dependency Updates
- 7+ key packages updated
- dython fix critical for GANERAID

---

### Phase 6: Branch Reorganization - January 2026

**Timeline:** January 23, 2026

**Objective:** Consolidate branch structure for clarity and maintainability

#### Key Changes

**Branch Consolidation:**
- Promoted `AWS_Round2` to become the new `main` branch
- Archived historical branches as tags for reference
- Simplified repository structure to single active branch

**Archive Tags Created:**
| Tag | Source Branch | Description |
|-----|---------------|-------------|
| v1.0-old-main | old-main | Initial local implementation (Aug 2025) |
| v2.0-aws-round1 | AWS_Round1 | First AWS migration (Sep-Oct 2025) |
| v3.0-legacy-main | main | 3-model stable release (Oct-Nov 2025) |

**Rationale:**
- AWS_Round2 represented the most advanced, feature-complete version
- Multiple active branches created confusion for collaborators
- Tags preserve historical state while simplifying daily workflow
- Single `main` branch aligns with modern Git best practices

**Migration Impact:**
- All new work should target `main` branch
- Historical versions remain accessible via tags
- Documentation updated to reflect new structure

---

### Phase 7: SDAC Framework Adoption — February 2026

**Timeline:** February 26, 2026

**Environment:** AWS SageMaker

**Status:** Active 🟢

#### Key Achievement

**Adopted the SEARCH Consortium's SDAC (Synthetic Data Anonymity and Credibility) framework as the unified evaluation taxonomy, expanding from 3 evaluation dimensions to 5.**

---

#### A. SDAC Evaluation Framework

**Aligned all evaluation output to the SDAC taxonomy with 5 dimensions:**

| SDAC Dimension | Metrics | Module |
|---|---|---|
| **Privacy** | DCR, NNDR, IMS, Re-identification Risk, MIA AUC | `src/evaluation/privacy.py` |
| **Fidelity** | JSD, KS Statistic, KL Divergence, Wasserstein Distance, Detection AUC, Correlation Similarity, Contingency Similarity | `src/evaluation/fidelity.py` |
| **Utility** | TSTR (Accuracy, F1, AUROC) across XGBoost/RF/LR, ML Efficacy, SRA | `src/evaluation/sdac_metrics.py` |
| **Fairness** | Demographic Parity Difference, Equalized Odds Difference, Disparate Impact Ratio | `src/evaluation/fairness.py` |
| **XAI** | Feature Importance Correlation, SHAP Distance | `src/evaluation/xai_metrics.py` |

**Unified orchestrator:** `src/evaluation/sdac_metrics.py` — single function call computes all 5 dimensions and outputs `sdac_evaluation_summary.csv`.

---

#### B. XGBoost as Primary Classifier

**Replaced Random Forest with XGBoost as the primary classifier throughout the evaluation pipeline.**

- TRTS framework (`src/evaluation/trts.py`): XGBoost runs all 4 scenarios (TRTR, TRTS, TSTR, TSTS) as primary; RF as secondary under `*_RF` keys
- SDAC Utility metrics: XGBoost-based TSTR accuracy/F1/AUROC as headline metrics
- Falls back to RF-only if XGBoost is not installed

---

#### C. New Evaluation Modules

| File | Description |
|---|---|
| `src/evaluation/sdac_metrics.py` | Unified SDAC orchestrator |
| `src/evaluation/fidelity.py` | KS, KL, Wasserstein, Detection AUC, Contingency Similarity |
| `src/evaluation/fairness.py` | Demographic Parity, Equalized Odds, Disparate Impact |
| `src/evaluation/xai_metrics.py` | Feature Importance Correlation, SHAP Distance |

---

#### D. New Visualizations

- **SDAC Radar Chart** — composite score per SDAC dimension, one trace per model
- **SDAC Heatmap** — models × metrics grid with per-column normalization, color-coded by SDAC category

---

#### E. Batch Pipeline Integration

- `evaluate_trained_models()` accepts `protected_col` and `compute_mia` parameters
- Section 3 runs with `compute_mia=False` (fast demo)
- Section 5.2 runs with `compute_mia=True` (full SDAC evaluation on optimized models)
- Fairness columns are NaN when `protected_col` is not specified

---

### Phase 7 Summary Statistics

#### Evaluation Expansion
- **Dimensions:** 3 (Privacy, Fidelity, Utility) → 5 (+Fairness, +XAI)
- **New modules:** 4 (sdac_metrics.py, fidelity.py, fairness.py, xai_metrics.py)
- **New metrics:** 12+ (KS, KL, WD, Detection AUC, Contingency Sim, Dem Parity, Eq Odds, Disp Impact, Feature Imp Corr, SHAP Dist, SRA, MIA AUC)
- **Primary classifier:** Random Forest → XGBoost

#### Output Changes
- **Replaced:** Scattered CSVs (`batch_evaluation_summary.csv`, `privacy_summary.csv`)
- **New:** Unified `sdac_evaluation_summary.csv` with SDAC category-prefixed columns
- **Retained:** `trts_detailed_results.csv` for drill-down analysis

---

### Phase 8: Persistent Environment and GANerAid Device Fix — March 2026

**Timeline:** March 10, 2026

**Environment:** AWS SageMaker

**Status:** Active 🟢

#### Key Achievement

**Eliminated per-session reinstallation by moving the conda environment to persistent EBS storage, and fixed the GANerAid CPU/CUDA device mismatch bug at its source.**

---

#### A. Persistent Conda Environment on EBS

**The Problem:** Every time the SageMaker notebook instance was stopped and restarted, the conda environment (installed under `/home/ec2-user/anaconda3/envs/`) was lost because that directory lives on the ephemeral root volume. This forced a full reinstall of all packages (~10 min) at the start of every session.

**The Solution:** Moved the conda environment to the persistent EBS volume at `/home/ec2-user/SageMaker/.envs/tablegen`. This volume survives instance stop/start cycles.

**New Scripts:**
| Script | Purpose | When to Run |
|--------|---------|-------------|
| `setup_env.sh` | Creates conda env on EBS, installs all deps, patches libraries, registers kernel | Once (first time only) |
| `on-start.sh` | Re-registers Jupyter kernel, ensures submodules initialized | Each session start (or via Lifecycle Config) |

**SageMaker Lifecycle Configuration:** `on-start.sh` can be installed as a Lifecycle Configuration "Start notebook" script for fully automatic startup — no manual steps needed.

---

#### B. GANerAid Device Bug Fix

**The Problem:** GANerAid's upstream library code hardcodes `torch.cuda.is_available()` checks in `utils.py`, `gan_trainer.py`, and `model.py` instead of respecting the `device` parameter passed at initialization. On GPU instances, this caused some tensors to be placed on CUDA while others remained on CPU, resulting in: `Input and hidden tensors are not at the same device, found input tensor at cuda:0 and hidden tensor at cpu`

**The Root Cause (3 files):**
- `utils.py` — `noise()` function sends noise tensor to CUDA unconditionally
- `gan_trainer.py` — `real_data_target()`, `fake_data_target()`, and `train_on_batch()` all use `torch.cuda.is_available()` instead of `gan.device`
- `model.py` — `GANerAidGenerator.forward()` creates `output = torch.zeros(...)` without specifying device, then does `.to(self.device)` on some but not all tensors

**The Fix:** Patched all 3 files to consistently use the `device` parameter:
- `noise()` now accepts a `device` kwarg
- `real_data_target()` / `fake_data_target()` accept a `device` parameter
- `GANerAidGenerator.init_hidden()` uses `device=self.device` in `torch.randn()`
- `GANerAidGenerator.forward()` creates output tensor with `device=self.device`
- All `.cuda()` calls removed in favor of `.to(device)`

Patched files are stored in `patches/` and applied by `setup_env.sh`.

The `GANerAidModel` wrapper continues to force CPU as a safety net due to other potential CUDA issues in the library.

---

#### C. tab-gan-metrics / dython Compatibility Fix

**The Problem:** GANerAid depends on `tab-gan-metrics`, which pins `dython==0.5.1`. Our project requires `dython>=0.7.12`. The newer dython renamed `compute_associations` to `_compute_associations`, breaking `tab-gan-metrics` imports.

**The Fix:** Install `tab-gan-metrics` with `--no-deps` and patch its import statements to fall back to `_compute_associations`. This is handled automatically by `setup_env.sh`.

---

#### D. Claude CLI PATH Persistence

Added `export PATH="$HOME/.local/bin:$PATH"` to `~/.bashrc` so the Claude Code CLI is available immediately on session start without reinstalling.

---

### Phase 8 Summary Statistics

#### Environment Improvements
- **Session startup time:** ~10 min (full reinstall) → ~2 sec (`on-start.sh`)
- **Scripts added:** 2 (`setup_env.sh`, `on-start.sh`)
- **Patches added:** 3 GANerAid library files + 2 tab-gan-metrics files

#### Bug Fixes
- GANerAid CPU/CUDA device mismatch: **fixed at source** (3 upstream files patched)
- tab-gan-metrics dython import: **fixed** (2 files patched)
- GANerAid intermittent availability: **fixed** (submodule init in `on-start.sh`)

---

## Development Philosophy Evolution

### old-main → AWS_Round1: **"Make it work in the cloud"**

- Focus: AWS migration and dependency management
- Challenge: Corporate environment restrictions
- Outcome: 3-model limitation discovered

### AWS_Round1 → main: **"Make it stable"**

- Focus: Production readiness and reproducibility
- Challenge: Version conflicts and edge cases
- Outcome: Stable 3-model configuration

### main → AWS_Round2: **"Make it excellent"**

- Focus: Architectural excellence and automation
- Challenge: Maintaining backward compatibility during refactor
- Outcome: Enterprise-grade modular framework

---

## Recommended Usage

### Use **main** Branch (Recommended)

- ✅ **All new development and research**
- ✅ **Production deployments**
- ✅ Advanced evaluation requirements (30+ metrics)
- ✅ Automated workflow with 8-model support
- ✅ Privacy risk assessment
- ✅ Modular architecture benefits
- ✅ Comprehensive visualization suite

### Historical Versions (Tags)

Historical versions are preserved as tags for reference only:

- **v1.0-old-main** - Original local implementation (Aug 2025)
- **v2.0-aws-round1** - First AWS migration (Sep-Oct 2025)
- **v3.0-legacy-main** - 3-model stable release (Oct-Nov 2025)

To checkout a historical version:
```bash
git checkout v1.0-old-main  # or v2.0-aws-round1, v3.0-legacy-main
```

---

## Key Lessons Learned

### Technical Insights

1. **Cloud Migration Complexity**
   - Corporate AWS environments have unexpected restrictions
   - Dependency conflicts multiply in constrained environments
   - Exact version pinning is essential for reproducibility

2. **Modular Architecture Value**
   - 94% code reduction proves value of separation of concerns
   - Backward compatibility is achievable with careful planning
   - Testing becomes tractable with small, focused modules

3. **Automation ROI**
   - Manual visualization steps are error-prone
   - Automated workflows improve reproducibility
   - Consistency across notebooks requires systematic automation

4. **Evaluation Depth**
   - Basic metrics miss critical synthetic data quality issues
   - Privacy risk requires specialized assessment
   - 30+ metrics provide comprehensive quality picture

### Project Management Insights

1. **Branch Strategy**
   - Feature branches for experiments
   - Stable main for production
   - Development branch (AWS_Round2) for innovation

2. **Documentation**
   - README files prevent knowledge loss
   - Commit messages are critical for understanding evolution
   - Timeline documents help onboard collaborators

3. **Incremental Progress**
   - Small, focused commits enable rollback
   - Phased migration reduces risk
   - Continuous validation catches regressions early

---

## Future Roadmap

### Short-Term (Q1 2026)

- [x] Resolve 3-model limitation in AWS environment (expanded to 8 models)
- [x] Add differential privacy mechanisms (PATE-GAN implementation)
- [x] Adopt SDAC evaluation framework (5-dimension taxonomy)
- [x] XGBoost as primary classifier throughout pipeline
- [x] Persistent conda environment on EBS (eliminates per-session reinstall)
- [x] Fix GANerAid CPU/CUDA device mismatch bug
- [x] SageMaker Lifecycle Configuration for automatic session startup
- [ ] Expand to additional healthcare datasets
- [ ] Implement automated quality gates

### Medium-Term (Q2-Q3 2026)

- [ ] GPU acceleration for larger datasets
- [ ] Real-time synthetic data generation API
- [ ] Interactive dashboard for result exploration
- [ ] Automated hyperparameter tuning strategies

### Long-Term (Q4 2026+)

- [ ] Multi-modal data support (tabular + imaging)
- [ ] Federated learning integration
- [ ] Production-grade deployment templates
- [ ] Comprehensive benchmark suite publication

---

## Collaborator Onboarding

### For New Team Members

1. **Clone the repository** - `main` branch contains the production-ready version
2. **Review README.md** for current architecture and setup
3. **Check docs/ folder** for detailed technical documentation
4. **Run test notebooks** to verify environment setup
5. **Read this timeline** to understand project evolution

### Quick Start Commands

```bash
# Clone repository (main branch is default)
git clone https://github.com/gcicc/tableGenCompare.git
cd tableGenCompare

# Review documentation
cat README.md
cat docs/Project-Evolution-Timeline.md

# Set up environment (AWS SageMaker)
pip install -r requirements.txt

# Run primary workflow notebook
jupyter notebook STG-Driver-breast-cancer.ipynb
```

---

## Acknowledgments

This project represents the culmination of multiple rounds of iteration, each building on lessons learned from production deployment on AWS SageMaker. The evolution from monolithic prototype to modular enterprise framework demonstrates the value of systematic refactoring and comprehensive evaluation.

**Key Contributors:**
- Architecture design and implementation
- AWS deployment and optimization
- Evaluation framework development
- Documentation and knowledge transfer

---

## Contact and Support

For questions about this project evolution or technical details:
- Review commit history for specific changes
- Check `docs/` folder for detailed technical documentation
- Consult README.md for current setup instructions

---

**Document Version:** 4.0
**Last Updated:** March 10, 2026
**Current Active Branch:** main
**Production Branch:** main
**Archived Tags:** v1.0-old-main, v2.0-aws-round1, v3.0-legacy-main
**Framework Version:** 8.0 (Persistent Environment + GANerAid Device Fix)
