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
# Initialize conda
source ~/anaconda3/bin/activate
conda init bash
exec bash

# Create Python 3.10 environment (avoids wheel/build issues)
conda create -n tablegen python=3.10 -y
conda activate tablegen
python -m pip install -U pip setuptools wheel

# Navigate to repo and initialize submodules (CRITICAL!)
cd ~/SageMaker/tableGenCompare
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name tablegen --display-name "Python (tablegen)"
```

### 4. Select Kernel in JupyterLab

**Kernel > Change Kernel > Python (tablegen)**

### 5. Returning Sessions

Each time you start a new session:

```bash
source ~/anaconda3/bin/activate
conda activate tablegen
cd ~/SageMaker/tableGenCompare
```

---

## Important Notes & Gotchas

- **Python version:** Use Python 3.10 on SageMaker classic Notebook Instances (avoids compatibility issues)
- **scikit-learn:** Version 1.7.2+ is required (updated from earlier 1.2.2 pinning)
- **dython:** Version 0.7.12+ required for GANERAID compatibility fix
- **Submodules:** CTAB-GAN, CTAB-GAN-Plus, and GANerAid are Git submodules - always run `git submodule update --init --recursive`
- **GPU models:** For GANerAid with CUDA, instantiate with: `GANerAidModel(device="cuda")`

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

### Section 5: Model Evaluation and Comparison
- Synthetic data generation using best parameters
- Statistical fidelity and utility preservation analysis
- Privacy risk assessment
- Cross-model performance comparison and visualization

---

## Modular Architecture

The codebase uses a clean modular structure for maintainability:

```
tableGenCompare/
├── setup.py              # Backward-compatible re-export layer
├── requirements.txt      # Python dependencies
├── STG-Driver-breast-cancer.ipynb  # Primary workflow notebook
│
├── src/                  # Modular source code
│   ├── config.py         # Session management
│   ├── models/           # Model implementations
│   ├── data/             # Data preprocessing
│   ├── evaluation/       # Quality, TRTS, privacy metrics
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

- **30+ evaluation metrics:** Comprehensive TRTS analysis with statistical fidelity, utility, and privacy metrics
- **Automated batch training:** Single function call handles multi-model hyperparameter optimization
- **Privacy dashboard:** DCR, NNDR, memorization risk, and re-identification assessment
- **Advanced visualizations:** ROC curves, PR curves, calibration plots, PCA comparisons
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
- Re-run: `python -m ipykernel install --user --name tablegen --display-name "Python (tablegen)"`

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
