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
- **§2.2b Collinearity reduction** — residual re-parameterization of near-deterministic column pairs (e.g., `perimeter`/`radius`/`area`). Decisions table is surfaced for human review before §3/§4/§5 run on the reduced schema. See [docs/collinearity-reduction.md](docs/collinearity-reduction.md).
- EDA heatmap highlights collinearity-treated columns in red

### Section 3: Demo Model Training
- Train all 8 models with default parameters
- Batch evaluation with synthetic data restored to the full real schema before scoring
- Emits `restoration_health.csv` — per-pair residual-preservation diagnostic

### Section 4: Staged Hyperparameter Optimization
- Optuna-based Bayesian optimization for each model
- Three-stage: smoke (10 trials) → pilot (diminishing-returns analysis) → full
- HPO tunes on the reduced schema so the residual is the actual training signal

### Section 5: Final Model Comparison (SDAC Framework)
- Retrain with best parameters on the reduced schema
- Restore dropped columns on synthetic output before evaluation (metrics are computed on the full real schema)
- **SDAC-aligned evaluation** across 5 dimensions: Privacy, Fidelity, Utility, Fairness, XAI
- Privacy risk assessment (DCR, NNDR, MIA, memorization, re-identification)
- Fidelity analysis (JSD, KS, KL, Wasserstein, Detection AUC, Mixed-Association Similarity, **Association Preservation**)
- Utility preservation (TSTR with XGBoost/RF/LR, ML Efficacy, SRA)
- Fairness metrics (Demographic Parity, Equalized Odds, Disparate Impact)
- XAI metrics (Feature Importance Correlation, SHAP Distance)
- Cross-model comparison via SDAC radar chart and heatmap
- **Independent SDMetrics cross-check** — every §3 / §5 run also emits `sdmetrics_evaluation_summary.csv`, `sdmetrics_metric_catalog.csv`, `sdmetrics_radar_chart.png`, and `sdmetrics_heatmap.png` from the public [SDMetrics library](https://docs.sdv.dev/sdmetrics). All SDMetrics outputs are clearly captioned `Source: SDMetrics (sdv.dev)`, and overlapping columns are marked with `†`. The SDAC suite remains the primary scorecard; SDMetrics is a third-party sanity check and does not feed composite scoring.
- Emits `restoration_health.csv` per optimized model

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
│   │   ├── eda.py             # run_comprehensive_eda, association primitives
│   │   ├── preprocessing.py   # config-driven cleaning / encoding
│   │   ├── collinearity.py    # residual reparam engine (synced w/ sibling)
│   │   └── target_integrity.py
│   ├── evaluation/       # SDAC evaluation framework
│   │   ├── sdac_metrics.py  # Unified SDAC orchestrator
│   │   ├── association.py   # Mixed-association matrix (dython-wrapped)
│   │   ├── quality.py       # Statistical fidelity
│   │   ├── fidelity.py      # KS, KL, WD, Detection AUC, Assoc_Preservation
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
- **Collinearity reducer:** Residual re-parameterization of near-deterministic pairs so generators preserve physical relationships (perimeter/radius, label/code) end-to-end. Surfaces a human-reviewable decisions table and a per-pair `restoration_health.csv` diagnostic. Byte-identical to the engine in sibling project `multi-table-gen-compare`. See [docs/collinearity-reduction.md](docs/collinearity-reduction.md).
- **XGBoost-primary classifiers:** XGBoost as default classifier across TSTR and utility evaluation, with RF and LR as secondary
- **30+ evaluation metrics:** Comprehensive TRTS analysis with statistical fidelity, utility, and privacy metrics
- **Association Preservation metric:** Scorecard axis measuring how well generators preserve strong real-data pairwise associations (`|A_real| > 0.3`), computed via mixed-association matrix (Pearson / Cramér's V / eta)
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

## Documentation

Topic-focused references in `docs/`:

- [workflow.md](docs/workflow.md) — shortest happy path from fresh clone to completed smoke run
- [datasets.md](docs/datasets.md) — the four benchmarks: schema, source, intended use
- [models.md](docs/models.md) — all 10 generators (incl. TabDiffusion + GReaT) + remaining planned additions
- [evaluation.md](docs/evaluation.md) — SDAC axes, metrics, and composite scoring
- [collinearity-reduction.md](docs/collinearity-reduction.md) — residual re-parameterization (§2.2b feature)
- [applications.md](docs/applications.md) — five use cases for synthetic tabular clinical data
- [decisions.md](docs/decisions.md) — architectural decisions log (ADR-style)
- [glossary.md](docs/glossary.md) — terminology (SDAC, TRTS, etc.)
- [experiment_log.md](docs/experiment_log.md) — running log of notable runs and refactors
- [USER-GUIDE.md](docs/USER-GUIDE.md) — long-form per-section user guide

Also: [Project-Evolution-Timeline.md](docs/Project-Evolution-Timeline.md) for
the historical development timeline.

## Contributing

This framework follows a systematic approach to synthetic data generation research, emphasizing reproducibility, comprehensive evaluation, and practical applicability to healthcare data challenges.



```bash
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```