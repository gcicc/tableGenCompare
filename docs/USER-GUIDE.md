---
title: "User Guide — Clinical Synthetic Data Generation Framework"
date: "2026-02-26"
---

# User Guide

**Clinical Synthetic Data Generation Framework**

A practical guide to generating, evaluating, and comparing synthetic tabular data for healthcare applications.

---

## Table of Contents

- [1. Installation and Setup](#1-installation-and-setup)
- [2. Data Preprocessing and EDA](#2-data-preprocessing-and-eda)
- [3. Model Configuration and Training](#3-model-configuration-and-training)
- [4. Hyperparameter Optimization](#4-hyperparameter-optimization)
- [5. Model Evaluation and Comparison (SDAC)](#5-model-evaluation-and-comparison-sdac)
- [6. Generating Synthetic Samples](#6-generating-synthetic-samples)
- [7. SDAC Metrics Reference](#7-sdac-metrics-reference)
- [8. Architecture Reference](#8-architecture-reference)

---

## 1. Installation and Setup

### System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.10 |
| RAM | 8 GB | 16+ GB |
| Disk | 10 GB | 50 GB |
| GPU | Optional | NVIDIA (CUDA) |

### AWS SageMaker Setup (Recommended)

1. Log in to AWS via Okta → `tec-rnd-sqs-dev`
2. Open **SageMaker AI > Notebook Instances**
3. Create instance: `ml.g4dn.xlarge`, 50 GB volume
4. Add Git repo: `https://github.com/gcicc/tableGenCompare.git` (branch: `main`)
5. Open JupyterLab and run first-time setup:

```bash
source ~/anaconda3/bin/activate
conda init bash && exec bash
conda create -n tablegen python=3.10 -y
conda activate tablegen
cd ~/SageMaker/tableGenCompare
git submodule update --init --recursive
pip install -r requirements.txt
python -m ipykernel install --user --name tablegen --display-name "Python (tablegen)"
```

6. Select kernel: **Kernel > Change Kernel > Python (tablegen)**

### Local Installation

```bash
git clone --recurse-submodules https://github.com/gcicc/tableGenCompare.git
cd tableGenCompare
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook STG-Driver-breast-cancer.ipynb
```

### Key Dependencies

| Group | Packages |
|---|---|
| Core | pandas, numpy, scipy |
| ML | scikit-learn >= 1.7.2, xgboost == 2.1.3 |
| Deep Learning | PyTorch |
| Synthetic Data | sdv, ctgan, copulas, GANerAid |
| HPO | optuna >= 4.0.0 |
| Visualization | matplotlib, seaborn, plotly |

---

## 2. Data Preprocessing and EDA

### What This Section Does

Section 2 of the notebook runs `run_comprehensive_eda()` and data preprocessing:

- Dataset overview and summary statistics
- Missing value analysis and MICE imputation
- Categorical encoding
- Target variable distribution analysis
- Correlation heatmaps and distribution plots

### Outputs Produced

- `column_analysis.csv` — per-column statistics
- `target_analysis.csv` — target variable breakdown
- Distribution plots, correlation heatmaps

### Caveats

1. **No synthetic dataset perfectly replicates real data** — expect differences, especially in tail distributions
2. **Small datasets amplify imperfections** — fewer than 500 rows limits GAN learning
3. **Categorical-heavy data is harder for GANs** — many categories create sparse representations
4. **Class imbalance affects generation quality** — minority classes may be underrepresented
5. **Privacy-utility tradeoff is fundamental** — stronger privacy guarantees reduce fidelity
6. **Stochasticity across runs** — results vary between runs due to random initialization
7. **GAN training instability** — mode collapse and vanishing gradients are inherent risks

---

## 3. Model Configuration and Training

### Available Models

| Model | Type | Best Suited For |
|---|---|---|
| **CTGAN** | GAN | General-purpose tabular data with mixed types |
| **CTAB-GAN** | GAN | Datasets needing enhanced preprocessing |
| **CTAB-GAN+** | GAN | Stable training on complex datasets (WGAN-GP) |
| **GANerAid** | GAN | Clinical/healthcare-specific data |
| **CopulaGAN** | Statistical | Preserving column correlations |
| **TVAE** | VAE | Stable training without adversarial dynamics |
| **PATE-GAN** | GAN | Privacy-sensitive data (differential privacy) |
| **MEDGAN** | GAN | Discrete medical features |

### A Priori Expectations

- **CTAB-GAN+** expected to outperform CTAB-GAN due to WGAN-GP stability
- **CopulaGAN** strong at correlation preservation (copula-based architecture)
- **TVAE** more stable training than GANs (no adversarial dynamics)
- **GANerAid** may show domain-specific advantages on clinical data
- **PATE-GAN** strongest privacy but lower fidelity (differential privacy noise)

---

## 4. Hyperparameter Optimization

### How Optimization Works

- **Engine:** Optuna Bayesian optimization (TPE sampler)
- **Direction:** Maximize combined score
- **Trial counts:** Smoke test = 5 trials, Full = 50 trials
- **Pruning:** `MedianPruner` for CTGAN/CTAB-GAN+

### Objective Function

```
Combined_Score = 0.6 × Similarity + 0.4 × Accuracy
```

- **Similarity** = Earth Mover's Distance + correlation preservation
- **Accuracy** = TSTR framework with XGBoost classifier

Source: `src/objective/functions.py`

### Editing the Objective Function

- Change weights: modify lines 24-26 in `src/objective/functions.py`
- Replace classifier: swap the model in the accuracy calculation
- Custom objective: pass `custom_objective_fn` to `optimize_models_batch()`

### After a Trial Run

1. Review `study.best_params` for each model
2. Expand search boundaries if optimal values hit the edge of the range
3. Fix low-importance parameters to reduce search space
4. Move from smoke test to full mode (`n_trials=50`)

---

## 5. Model Evaluation and Comparison (SDAC)

### SDAC Framework

All evaluation output is organized according to the **SEARCH Consortium's SDAC** (Synthetic Data Anonymity and Credibility) taxonomy across **5 dimensions**:

| Dimension | What It Measures |
|---|---|
| **Privacy** | Risk of exposing real patient data through the synthetic data |
| **Fidelity** | How closely synthetic data matches real data distributions |
| **Utility** | Whether synthetic data is useful for downstream ML tasks |
| **Fairness** | Whether synthetic data preserves or amplifies demographic biases |
| **XAI** | Whether synthetic data preserves model explainability properties |

### Running Evaluation

```python
# Section 3 — quick demo (no MIA, no fairness)
results = evaluate_trained_models(section_number=3, scope=globals())

# Section 5 — full SDAC evaluation
results = evaluate_trained_models(
    section_number=5,
    scope=globals(),
    compute_mia=True,           # Enable Membership Inference Attack
    protected_col="gender"      # Optional: enable fairness metrics
)
```

### Output Files

| File | Description |
|---|---|
| `sdac_evaluation_summary.csv` | One row per model, all SDAC metrics as columns |
| `trts_detailed_results.csv` | 30+ metrics × 4 scenarios drill-down |
| `sdac_radar_chart.png` | Composite score per SDAC dimension per model |
| `sdac_heatmap.png` | Models × metrics grid, color-coded by SDAC category |

### How the Best Model Is Chosen

1. **Primary:** Model with highest `Combined_Score` (0.6 × Similarity + 0.4 × Accuracy)
2. **Secondary tiebreaker:** Privacy score

---

## 6. Generating Synthetic Samples

### In-Session Generation

```python
# Access trained model from evaluation results
model = results['CTGAN']['model']
synthetic_data = model.generate(n_samples=1000)
synthetic_data.to_csv("my_synthetic_data.csv", index=False)
```

### Retraining from Saved Parameters

```python
import pandas as pd
from src.models.model_factory import ModelFactory

# Load best parameters
params = pd.read_csv("results/dataset/best_parameters.csv")
best_params = params.iloc[0].to_dict()

# Recreate and train
model = ModelFactory.create("CTGAN", **best_params)
model.train(real_data, target_column="diagnosis")
synthetic = model.generate(n_samples=500)
```

---

## 7. SDAC Metrics Reference

All metrics are organized by the 5 SDAC dimensions. Each metric includes a plain-language definition, range, and interpretation guide.

---

### 7.1 Privacy Metrics

Privacy metrics measure how well synthetic data protects the confidentiality of real patient records.

#### DCR (Distance to Closest Record)

| | |
|---|---|
| **What it measures** | The average distance between each synthetic record and the nearest real record |
| **Plain language** | How far away is each fake record from the closest real patient? |
| **Range** | 0 to ∞ (higher = more private) |
| **Interpretation** | A value near 0 means synthetic records are nearly identical to real ones (bad). Higher values mean synthetic records are sufficiently different from any real patient. |
| **Source** | `src/evaluation/privacy.py` |

#### NNDR (Nearest Neighbor Distance Ratio)

| | |
|---|---|
| **What it measures** | Ratio of each synthetic record's distance to the nearest real record vs. its distance to other synthetic records |
| **Plain language** | Is each fake record closer to a real patient or to other fake records? |
| **Range** | 0 to ∞ (values > 1.0 are better) |
| **Interpretation** | NNDR < 1.0 means synthetic records are closer to real data than to other synthetic data — a sign of memorization. NNDR > 1.0 means synthetic records cluster with each other, not with real data. |
| **Source** | `src/evaluation/privacy.py` |

#### IMS (Individual Memorization Score)

| | |
|---|---|
| **What it measures** | Percentage of synthetic records that are near-exact copies of real records |
| **Plain language** | What fraction of fake records are essentially copies of real patients? |
| **Range** | 0 to 1 (lower = more private) |
| **Interpretation** | 0% = no memorization (ideal). Values above 5% indicate the model is copying real records rather than learning general patterns. |
| **Source** | `src/evaluation/privacy.py` — reported as `memorization_score` |

#### Re-identification Risk

| | |
|---|---|
| **What it measures** | Percentage of synthetic records close enough to a real record that someone could potentially link them back to a real patient |
| **Plain language** | Could someone look at a fake record and figure out which real patient it came from? |
| **Range** | 0 to 1 (lower = more private) |
| **Interpretation** | 0% = no re-identification risk. Values above 10% are concerning for patient privacy. |
| **Source** | `src/evaluation/privacy.py` |

#### MIA AUC (Membership Inference Attack AUC)

| | |
|---|---|
| **What it measures** | How well a classifier can distinguish whether a given record was in the original training set |
| **Plain language** | Can an attacker tell whether a specific person's data was used to train the model? |
| **Range** | 0.5 to 1.0 (closer to 0.5 = more private) |
| **Interpretation** | AUC = 0.5 means the attacker cannot distinguish members from non-members (ideal). AUC > 0.7 means significant membership leakage. AUC > 0.9 means severe privacy risk. |
| **Note** | Only computed when `compute_mia=True` (Section 5 full evaluation). Uses a shadow-model approach with Random Forest classifier. |
| **Reference** | Shokri et al., "Membership Inference Attacks Against Machine Learning Models" (2017) |
| **Source** | `src/evaluation/privacy.py` |

---

### 7.2 Fidelity Metrics

Fidelity metrics measure how closely the synthetic data matches the statistical properties of the real data.

#### JSD (Jensen-Shannon Divergence)

| | |
|---|---|
| **What it measures** | Symmetric measure of difference between the real and synthetic distributions for each column |
| **Plain language** | How different are the shapes of the real vs. synthetic data distributions? |
| **Range** | 0 to 1 (lower = more similar) |
| **Interpretation** | 0 = identical distributions. < 0.1 = excellent fidelity. > 0.3 = distributions differ substantially. Reported as the mean across all numeric columns. |
| **Source** | `src/evaluation/sdac_metrics.py` |

#### KS Statistic (Kolmogorov-Smirnov)

| | |
|---|---|
| **What it measures** | Maximum difference between the cumulative distribution functions of real and synthetic data |
| **Plain language** | What is the biggest gap between the real and synthetic data when you line them up from smallest to largest? |
| **Range** | 0 to 1 (lower = more similar) |
| **Interpretation** | 0 = identical CDFs. < 0.1 = excellent match. > 0.3 = significant distributional shift. Based on the standard two-sample KS test. |
| **Reference** | `scipy.stats.ks_2samp` |
| **Source** | `src/evaluation/fidelity.py` |

#### KL Divergence (Kullback-Leibler)

| | |
|---|---|
| **What it measures** | Information lost when the synthetic distribution is used to approximate the real distribution |
| **Plain language** | How much information do you lose by using the fake data instead of the real data? |
| **Range** | 0 to ∞ (lower = more similar) |
| **Interpretation** | 0 = identical distributions. < 0.1 = excellent. Unlike JSD, KL divergence is asymmetric and unbounded. Computed via histogram binning with smoothing to avoid division by zero. |
| **Source** | `src/evaluation/fidelity.py` |

#### Wasserstein Distance (Earth Mover's Distance)

| | |
|---|---|
| **What it measures** | Minimum "work" needed to transform one distribution into another |
| **Plain language** | How much effort would it take to reshape the fake data to look exactly like the real data? |
| **Range** | 0 to ∞ (lower = more similar) |
| **Interpretation** | 0 = identical distributions. Sensitive to both shape and location differences. More robust than KL divergence for distributions with non-overlapping support. |
| **Reference** | `scipy.stats.wasserstein_distance` |
| **Source** | `src/evaluation/fidelity.py` |

#### Detection AUC

| | |
|---|---|
| **What it measures** | How well a logistic regression classifier can tell real records from synthetic records |
| **Plain language** | If you mix real and fake records together, can a simple model tell them apart? |
| **Range** | 0.5 to 1.0 (closer to 0.5 = better fidelity) |
| **Interpretation** | AUC = 0.5 means the classifier cannot distinguish real from synthetic (excellent fidelity). AUC > 0.8 means synthetic data is easily distinguishable (poor fidelity). |
| **Source** | `src/evaluation/fidelity.py` |

#### Correlation Similarity

| | |
|---|---|
| **What it measures** | How well the synthetic data preserves the pairwise correlation structure between columns |
| **Plain language** | If column A and column B are correlated in the real data, are they still correlated the same way in the synthetic data? |
| **Range** | 0 to 1 (higher = better) |
| **Interpretation** | 1.0 = perfect correlation preservation. < 0.7 = some relationships between variables are being distorted. |
| **Source** | `src/evaluation/sdac_metrics.py` |

#### Contingency Similarity

| | |
|---|---|
| **What it measures** | For categorical columns, how well the synthetic data preserves the joint frequency distributions |
| **Plain language** | Are combinations of categorical values (e.g., "Male" + "Diabetic") equally common in real and synthetic data? |
| **Range** | 0 to 1 (higher = better) |
| **Interpretation** | 1.0 = identical category co-occurrence patterns. Low values indicate the model is not capturing interactions between categorical variables. |
| **Source** | `src/evaluation/fidelity.py` |

---

### 7.3 Utility Metrics

Utility metrics measure whether synthetic data is useful as a substitute for real data in downstream machine learning tasks.

#### TSTR Accuracy (Train Synthetic, Test Real)

| | |
|---|---|
| **What it measures** | Classification accuracy when a model is trained on synthetic data and tested on real data |
| **Plain language** | If you train a model using only fake data, how well does it perform on real patients? |
| **Range** | 0 to 1 (higher = better) |
| **Interpretation** | Close to TRTR accuracy = synthetic data is a good substitute. Large gap = synthetic data is missing important patterns. Reported for XGBoost (primary), RF, and LR. |
| **Source** | `src/evaluation/trts.py`, `src/evaluation/sdac_metrics.py` |

#### TSTR F1 Score

| | |
|---|---|
| **What it measures** | Harmonic mean of precision and recall for the TSTR scenario |
| **Plain language** | Balanced measure of how well a fake-data-trained model identifies positive cases in real patients |
| **Range** | 0 to 1 (higher = better) |
| **Interpretation** | More informative than accuracy when classes are imbalanced. Reported for XGBoost (primary). |
| **Source** | `src/evaluation/sdac_metrics.py` |

#### TSTR AUROC

| | |
|---|---|
| **What it measures** | Area under the ROC curve for the TSTR scenario |
| **Plain language** | Overall ability of a fake-data-trained model to rank real patients correctly |
| **Range** | 0.5 to 1.0 (higher = better) |
| **Interpretation** | 0.5 = random guessing. > 0.8 = good. > 0.9 = excellent. Threshold-independent measure. |
| **Source** | `src/evaluation/sdac_metrics.py` |

#### ML Efficacy Score

| | |
|---|---|
| **What it measures** | Ratio of TSTR accuracy to TRTR accuracy |
| **Plain language** | What fraction of real-data performance is preserved when training on synthetic data? |
| **Range** | 0 to ~1.0 (higher = better) |
| **Interpretation** | 1.0 = synthetic data is as useful as real data for ML. 0.8 = 80% of performance preserved. Values > 1.0 can occur but usually indicate overfitting. |
| **Source** | `src/evaluation/sdac_metrics.py` |

#### SRA (Synthetic Ranking Agreement)

| | |
|---|---|
| **What it measures** | Whether different ML models rank in the same order when trained on synthetic vs. real data |
| **Plain language** | If Model A beats Model B on real data, does it also beat Model B on synthetic data? |
| **Range** | -1 to 1 (higher = better, Spearman correlation) |
| **Interpretation** | 1.0 = identical model rankings. 0 = no agreement. Negative values indicate reversed rankings. Computed across RF, LR, and XGBoost. |
| **Source** | `src/evaluation/sdac_metrics.py` |

---

### 7.4 Fairness Metrics

Fairness metrics measure whether the synthetic data introduces or amplifies biases with respect to a protected attribute (e.g., gender, race, age group). These metrics require specifying `protected_col` — they are blank/NaN if no protected attribute is provided.

#### Demographic Parity Difference

| | |
|---|---|
| **What it measures** | Difference in positive prediction rates between protected groups |
| **Plain language** | Are different demographic groups predicted positive at the same rate? |
| **Range** | 0 to 1 (lower = more fair) |
| **Interpretation** | 0 = perfectly equal prediction rates across groups. Values > 0.1 indicate meaningful disparity. Computed on both real and synthetic data; comparing the two reveals whether the model amplifies bias. |
| **Source** | `src/evaluation/fairness.py` |

#### Equalized Odds Difference

| | |
|---|---|
| **What it measures** | Maximum difference in true positive rate and false positive rate between protected groups |
| **Plain language** | Do different demographic groups experience the same error rates? |
| **Range** | 0 to 1 (lower = more fair) |
| **Interpretation** | 0 = identical error rates across groups. High values mean the model is more accurate for one group than another. |
| **Source** | `src/evaluation/fairness.py` |

#### Disparate Impact Ratio

| | |
|---|---|
| **What it measures** | Ratio of positive prediction rates between the disadvantaged and advantaged groups |
| **Plain language** | Is one demographic group proportionally less likely to receive a positive prediction? |
| **Range** | 0 to ∞ (closer to 1.0 = more fair) |
| **Interpretation** | 1.0 = perfect parity. The "four-fifths rule" considers values below 0.8 as evidence of disparate impact. Values > 1.0 indicate the minority group is predicted positive more often. |
| **Source** | `src/evaluation/fairness.py` |

---

### 7.5 XAI (Explainability) Metrics

XAI metrics measure whether the synthetic data preserves the explainability properties of models trained on real data.

#### Feature Importance Correlation

| | |
|---|---|
| **What it measures** | Pearson correlation between Random Forest feature importances trained on real vs. synthetic data |
| **Plain language** | Do the same features matter for prediction in both real and synthetic data? |
| **Range** | -1 to 1 (higher = better) |
| **Interpretation** | 1.0 = identical feature importance rankings. > 0.8 = excellent preservation. Low values mean the synthetic data has shifted which features are most predictive. |
| **Source** | `src/evaluation/xai_metrics.py` |

#### SHAP Distance

| | |
|---|---|
| **What it measures** | Cosine distance between mean SHAP value vectors from models trained on real vs. synthetic data |
| **Plain language** | Do individual predictions get explained the same way when using real vs. synthetic training data? |
| **Range** | 0 to 1 (lower = better) |
| **Interpretation** | 0 = identical explanations. < 0.2 = good preservation. High values mean the model's reasoning changes when trained on synthetic data. |
| **Note** | Requires the `shap` library. Returns NaN if SHAP is not installed. |
| **Reference** | Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017) |
| **Source** | `src/evaluation/xai_metrics.py` |

---

### 7.6 TRTS Framework Detail

The TRTS (Train Real/Test Synthetic) framework evaluates synthetic data through 4 classification scenarios:

| Scenario | Train On | Test On | What It Shows |
|---|---|---|---|
| **TRTR** | Real | Real | Baseline (best possible performance) |
| **TSTR** | Synthetic | Real | Can synthetic data replace real data for training? |
| **TRTS** | Real | Synthetic | Does the model generalize to synthetic data? |
| **TSTS** | Synthetic | Synthetic | Internal consistency of synthetic data |

Each scenario reports 30+ classification metrics. The **primary classifier** is XGBoost; Random Forest runs as secondary (results stored under `*_RF` keys).

Key metrics per scenario: accuracy, balanced accuracy, precision, recall, F1, specificity, MCC, Cohen's Kappa, AUROC, AUPRC, Brier Score.

---

### 7.7 Quality Score Labels

| Score Range | Label |
|---|---|
| ≥ 0.80 | Excellent |
| 0.60 – 0.79 | Good |
| 0.40 – 0.59 | Fair |
| < 0.40 | Poor |

---

## 8. Architecture Reference

### Directory Structure

```
tableGenCompare/
├── setup.py                    # Backward-compatible re-export layer
├── requirements.txt            # Python dependencies
├── STG-Driver-*.ipynb          # Workflow notebooks (one per dataset)
│
├── src/
│   ├── config.py               # Session management and defaults
│   ├── models/
│   │   ├── model_factory.py    # ModelFactory.create()
│   │   ├── search_spaces.py    # Hyperparameter search spaces
│   │   ├── batch_training.py   # Batch training and retraining
│   │   ├── batch_optimization.py  # Optuna batch HPO
│   │   └── implementations/    # Per-model wrappers
│   ├── data/
│   │   ├── preprocessing.py    # MICE imputation, encoding
│   │   ├── eda.py              # run_comprehensive_eda()
│   │   └── summary.py          # Data summaries
│   ├── evaluation/
│   │   ├── sdac_metrics.py     # Unified SDAC orchestrator
│   │   ├── quality.py          # JSD, correlation, ML utility
│   │   ├── fidelity.py         # KS, KL, WD, Detection AUC
│   │   ├── trts.py             # 4-scenario TRTS (XGBoost primary)
│   │   ├── privacy.py          # DCR, NNDR, MIA, memorization
│   │   ├── fairness.py         # Demographic parity, equalized odds
│   │   ├── xai_metrics.py      # Feature importance, SHAP distance
│   │   └── batch.py            # Batch evaluation pipeline
│   ├── objective/
│   │   └── functions.py        # Optuna objective function
│   ├── visualization/
│   │   ├── section4.py         # Optuna visualizations
│   │   └── section5.py         # TRTS, privacy, SDAC charts
│   └── utils/
│       ├── paths.py            # Path management
│       └── parameters.py       # Parameter save/load
│
├── data/                       # Input datasets
├── results/                    # Generated outputs (per dataset, per section)
├── docs/                       # Documentation
│
├── CTAB-GAN/                   # Git submodule
├── CTAB-GAN-Plus/              # Git submodule
└── GANerAid/                   # Git submodule
```

### Results Directory Structure

```
results/{dataset-name}/{date}/
├── Section-3/
│   ├── sdac_evaluation_summary.csv
│   ├── trts_detailed_results.csv
│   ├── sdac_radar_chart.png
│   ├── sdac_heatmap.png
│   ├── trts_roc_curves.png
│   ├── trts_pr_curves.png
│   └── trts_calibration_curves.png
├── Section-4/
│   ├── optim_history_{model}.png
│   ├── param_importance_{model}.png
│   └── optuna_summary_all_models.png
└── Section-5/
    ├── sdac_evaluation_summary.csv
    ├── trts_detailed_results.csv
    ├── privacy_dashboard.png
    ├── sdac_radar_chart.png
    └── sdac_heatmap.png
```

---

**Document Version:** 1.0
**Last Updated:** February 26, 2026
**Framework Version:** 7.0 (SDAC Evaluation Framework)
