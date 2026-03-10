# Clinical Synthetic Data Generation Framework

A comprehensive benchmarking suite for evaluating synthetic tabular data generation methods across multiple healthcare datasets.

## Overview

This project implements and compares the performance of state-of-the-art synthetic data generation methods specifically designed for clinical and healthcare tabular data. The framework provides a standardized pipeline for training, evaluating, and optimizing generative models.

**Primary Workflow:** Use `STG-Driver-breast-cancer.ipynb` as the main entry point. This consolidated notebook includes the complete pipeline with all 8 generative models.

## Synthetic Data Generation Methods

The framework evaluates **8 generative models**:

| Model | Type | Description |
|-------|------|-------------|
| **CTGAN** | GAN | Standard GAN approach for tabular data |
| **CTAB-GAN** | GAN | Enhanced preprocessing pipeline |
| **CTAB-GAN+** | GAN | Advanced stability with WGAN-GP losses |
| **GANerAid** | GAN | Purpose-built clinical data generator |
| **CopulaGAN** | Statistical | Copula-based approach |
| **TVAE** | VAE | Variational autoencoder for tabular synthesis |
| **PATE-GAN** | GAN | Privacy-preserving GAN with differential privacy |
| **MEDGAN** | GAN | Medical record generation with autoencoder |

## Datasets

The framework supports healthcare datasets including:
- **Breast Cancer** - Cancer diagnosis and prognosis data (primary)
- **Alzheimer's Disease** - Neurological condition classification
- **Liver Disease** - Hepatic condition assessment
- **Pakistani Liver Patient** - Regional liver disease dataset

---

## Quick Start

### Option 1: Local Installation

```bash
# Clone repository with submodules
git clone --recurse-submodules https://github.com/gcicc/tableGenCompare.git
cd tableGenCompare

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook STG-Driver-breast-cancer.ipynb
```

### Option 2: AWS SageMaker (Recommended for GPU)

See [AWS SageMaker Setup](#aws-sagemaker-setup) below.

---

## AWS SageMaker Setup

### 1. Access AWS SageMaker

1. Log in to AWS via Okta: **AWS TEC**
2. Select environment: `tec-rnd-sqs-dev`
3. Open **Amazon SageMaker AI** > **Notebook instances**

### 2. Create Notebook Instance

1. Click **Create notebook instance**
2. Recommended settings:
   - **Instance type:** `ml.g4dn.xlarge` (GPU-enabled)
   - **Volume:** 50 GB (or more for large datasets/models)
   - **IAM role:** Use existing role with appropriate permissions
3. Under **Git repositories**, add:
   - URL: `https://github.com/gcicc/tableGenCompare.git`
   - Branch: `main`
4. Create and wait until status shows **InService**
5. Click **Open JupyterLab**

### 3. Initialize Environment (First Time Only)

Open a terminal in JupyterLab and run:

```bash
source ~/anaconda3/bin/activate
conda init bash
exec bash

cd ~/SageMaker/tableGenCompare
bash setup_env.sh
```

This installs the conda env to the **persistent EBS volume** (`~/SageMaker/.envs/tablegen`)
so it survives instance stop/start cycles. It also initializes all git submodules and registers
the Jupyter kernel.

### 4. Select Kernel in JupyterLab

**Kernel > Change Kernel > Python (tablegen)**

### 5. Returning Sessions (Automatic)

The `on-start.sh` script re-registers the Jupyter kernel and ensures submodules
are initialized. Install it as a **SageMaker Lifecycle Configuration** for fully
automatic startup:

1. Go to **SageMaker Console > Notebook instances > Your instance > Edit**
2. Under **Lifecycle configuration**, create a new one
3. Paste the contents of `on-start.sh` into the **Start notebook** script
4. Save and restart the instance

**Or**, if you haven't set up the lifecycle config yet, just run manually:

```bash
bash ~/SageMaker/tableGenCompare/on-start.sh
```

No reinstalling, no `pip install`, no `conda create` — everything is already on the persistent volume.

---

## Important Notes & Gotchas

- **Python version:** Use Python 3.10 on SageMaker classic Notebook Instances (avoids compatibility issues)
- **Persistent environment:** The conda env is installed on the EBS volume (`~/SageMaker/.envs/tablegen`) so it survives instance stop/start. Only the kernel registration (root volume) needs refreshing — handled by `on-start.sh`.
- **scikit-learn:** Version 1.7.2+ is required (updated from earlier 1.2.2 pinning)
- **dython:** Version 0.7.12+ required; `tab-gan-metrics` is patched at install time for compatibility (see `setup_env.sh`)
- **Submodules:** CTAB-GAN, CTAB-GAN-Plus, and GANerAid are Git submodules — `setup_env.sh` and `on-start.sh` handle initialization automatically
- **GANerAid device bug:** The upstream GANerAid library hardcodes `torch.cuda.is_available()` instead of respecting the `device` parameter, causing CPU/CUDA tensor mismatches on GPU instances. Patched files in `patches/` are applied by `setup_env.sh` to fix this. The `GANerAidModel` wrapper also forces CPU as a safety net.

---

## Notebook Structure

The STG-Driver notebook follows a standardized 5-section pipeline:

### Section 1: Setup and Data Loading
- Environment configuration and library imports
- Dataset loading and initial exploration

### Section 2: Data Preprocessing and Analysis
- Comprehensive dataset overview and statistics
- Missing value analysis and MICE imputation
- Categorical encoding and feature preparation

### Section 3: Model Configuration
- Model factory setup and configuration management
- Hyperparameter space definitions for all 8 models

### Section 4: Hyperparameter Optimization
- Optuna-based Bayesian optimization for each model
- Performance evaluation using enhanced objective functions
- Batch training with comprehensive logging

### Section 5: Model Evaluation and Comparison (SDAC Framework)
- Synthetic data generation using best parameters
- **SDAC-aligned evaluation** across 5 dimensions: Privacy, Fidelity, Utility, Fairness, XAI
- Privacy risk assessment (DCR, NNDR, MIA, memorization, re-identification)
- Fidelity analysis (JSD, KS, KL, Wasserstein, Detection AUC)
- Utility preservation (TSTR with XGBoost/RF/LR, ML Efficacy, SRA)
- Fairness metrics (Demographic Parity, Equalized Odds, Disparate Impact)
- XAI metrics (Feature Importance Correlation, SHAP Distance)
- Cross-model comparison via SDAC radar chart and heatmap

---

## Modular Architecture

The codebase uses a clean modular structure for maintainability:

```
tableGenCompare/
├── setup.py              # Backward-compatible re-export layer
├── setup_env.sh          # ONE-TIME env setup (persistent EBS)
├── on-start.sh           # Per-boot kernel registration & submodule init
├── requirements.txt      # Python dependencies
├── STG-Driver-breast-cancer.ipynb  # Primary workflow notebook
│
├── patches/              # Upstream library bug fixes
│   ├── ganeraid_model.py    # Device-aware GANerAidGAN/Generator
│   ├── ganeraid_trainer.py  # Device-aware GanTrainer
│   └── ganeraid_utils.py    # Device-aware noise()
│
├── src/                  # Modular source code
│   ├── config.py         # Session management
│   ├── models/           # Model implementations
│   ├── data/             # Data preprocessing
│   ├── evaluation/       # SDAC evaluation framework
│   │   ├── sdac_metrics.py  # Unified SDAC orchestrator
│   │   ├── quality.py       # Statistical fidelity
│   │   ├── fidelity.py      # KS, KL, WD, Detection AUC
│   │   ├── trts.py          # TRTS framework (XGBoost primary)
│   │   ├── privacy.py       # DCR, NNDR, MIA, memorization
│   │   ├── fairness.py      # Demographic parity, equalized odds
│   │   ├── xai_metrics.py   # Feature importance, SHAP distance
│   │   └── batch.py         # Batch evaluation pipeline
│   ├── objective/        # Optuna objective functions
│   ├── visualization/    # Section-specific visualizations
│   └── utils/            # Utility functions
│
├── data/                 # Dataset directory
├── results/              # Generated outputs
│
└── CTAB-GAN/            # Git submodules
    CTAB-GAN-Plus/
    GANerAid/
```

All notebooks use `from setup import *` without changes - the thin re-export layer ensures backward compatibility.

---

## Key Features

- **SDAC Framework:** Evaluation aligned to the SEARCH Consortium's Synthetic Data Anonymity and Credibility (SDAC) taxonomy across 5 dimensions — Privacy, Fidelity, Utility, Fairness, XAI
- **Unified SDAC output:** Single `sdac_evaluation_summary.csv` with all metrics organized by SDAC category
- **XGBoost-primary classifiers:** XGBoost as default classifier across TSTR and utility evaluation, with RF and LR as secondary
- **30+ evaluation metrics:** Comprehensive TRTS analysis with statistical fidelity, utility, and privacy metrics
- **Automated batch training:** Single function call handles multi-model hyperparameter optimization
- **Privacy dashboard:** DCR, NNDR, MIA AUC, memorization risk, and re-identification assessment
- **Advanced visualizations:** SDAC radar chart, SDAC heatmap, ROC curves, PR curves, calibration plots, PCA comparisons
- **Optuna integration:** Bayesian optimization with automatic visualization generation

---

## Requirements

**System:**
- Python 3.10+
- 10+ GB disk space
- 8+ GB RAM (16+ recommended)
- GPU optional but recommended for faster training

**Key Dependencies:**
- pandas, numpy, scipy
- torch (PyTorch)
- scikit-learn >= 1.7.2
- xgboost == 2.1.3
- sdv >= 1.32.1 (includes CTGAN, CopulaGAN, TVAE)
- dython >= 0.7.12
- optuna >= 4.0.0

See `requirements.txt` for complete dependency list.

---

## Troubleshooting

**PyTorch installation fails**
- Ensure sufficient disk space (10+ GB)
- CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**CTAB-GAN import errors**
- Run: `git submodule update --init --recursive`
- Verify: `ls CTAB-GAN CTAB-GAN-Plus GANerAid`

**Out of memory during training**
- Reduce `n_trials` parameter in Section 4
- Start with `n_trials=5` for testing

**Kernel not found in JupyterLab**
- Run: `bash ~/SageMaker/tableGenCompare/on-start.sh` to re-register the kernel
- Or set up the SageMaker Lifecycle Configuration (see [Returning Sessions](#5-returning-sessions-automatic))

**GANerAid device mismatch (CPU/CUDA error)**
- The upstream GANerAid library ignores the `device` parameter. Re-run `setup_env.sh` to apply patches.
- The `GANerAidModel` wrapper forces CPU as a safety net; this is intentional.

**Environment lost after instance restart**
- The conda env lives on persistent EBS (`~/SageMaker/.envs/tablegen`) — it should survive restarts.
- Only the kernel registration needs refreshing: run `on-start.sh` or use a Lifecycle Configuration.

---

## AWS Folder

The `aws/` directory contains AWS CLI v2 bundled distribution for SageMaker deployment:
- `dist/` - Bundled AWS CLI v2 executables
- `install` - Installation script
- `README.md` - Installation instructions

This allows SageMaker notebook instances to run AWS CLI commands without manual installation.

---

## Contributing

This framework follows a systematic approach to synthetic data generation research, emphasizing reproducibility, comprehensive evaluation, and practical applicability to healthcare data challenges.

For project evolution and development history, see `docs/Project-Evolution-Timeline.md`.
