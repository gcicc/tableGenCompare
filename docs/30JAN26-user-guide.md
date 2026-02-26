# Plan to Implement

## Plan: Create User Guide (`docs/USER-GUIDE.md`)

### Task
Create a comprehensive User Guide for the Clinical Synthetic Data Generation Framework at `docs/USER-GUIDE.md`.

---
## Document Structure

### Front Matter
- Title, purpose statement, last‐updated date
- Table of Contents with anchor links to all sections

---
## Section 1: Installation and Setup

1. **System Requirements** — Python 3.10+, RAM/disk, GPU optional
2. **AWS SageMaker Setup** — Step‐by‐step (instance type, conda env, kernel selection, returning sessions) from README
3. **Local Installation** — `git clone --recurse-submodules`, venv, pip install, Jupyter launch
4. **Dependency Overview** — Grouped table from requirements.txt (Core, ML, Deep Learning, Synthetic Data, HPO, Visualization)
5. **Notebook Configuration** — Table of all `NOTEBOOK_CONFIG_DEFAULTS` fields from `src/config.py:25–56` with types, defaults, descriptions. Example config block.
6. **Troubleshooting** — PyTorch, CTAB‐GAN imports, OOM, kernel issues

---
## Section 2: Data Preprocessing and EDA (Pipeline Section 2)

1. **What This Section Does** — `run_comprehensive_eda()` from `src/data/eda.py`, preprocessing from `src/data/preprocessing.py`
2. **Outputs Produced** — Tables: `column_analysis.csv`, `target_analysis.csv`, `target_balance_metrics.csv`, distribution plots, correlation heatmap, etc.
3. **Preprocessing Pipeline** — Missing value strategies, categorical encoding, row subsetting
4. **Caveats & Expectation Management** — 7 key caveats:
   - No synthetic dataset perfectly replicates real data
   - Small datasets amplify imperfections
   - Categorical‐heavy data is harder for GANs
   - Class imbalance affects generation quality
   - Privacy‐utility tradeoff is fundamental
   - Stochasticity across runs
   - GAN training instability (mode collapse, vanishing gradients)

---
## Section 3: Model Configuration and Training (Pipeline Section 3)

### 3.1 Available Models
For each model (CTGAN, CTAB‐GAN, CTAB‐GAN+, GANerAid, CopulaGAN, TVAE, PATE‐GAN, MEDGAN):
- Source/library
- Architecture summary
- Key default parameters
- A priori performance expectations
- Best suited for
- Source file path

**A priori expectations:**
- CTAB‐GAN+ expected to outperform CTAB‐GAN (WGAN‐GP stability)
- CopulaGAN strong at correlation preservation (copula‐based)
- TVAE more stable training than GANs (no adversarial dynamics)
- GANerAid may show domain‐specific advantages on clinical data
- PATE‐GAN strongest privacy but lower fidelity (differential privacy noise)

### 3.2 Outputs Produced
Per‐model evaluation summaries, statistical similarity, plots, batch summary CSV.

---
## Section 4: Hyperparameter Optimization (Pipeline Section 4)

### 4.1 How Optimization Works
Optuna Bayesian optimization, maximize direction, trial counts (smoke=5, full=50), `MedianPruner` for CTGAN/CTABGAN+. Code in `src/models/batch_optimization.py`.

### 4.2 Objective Function
`enhanced_objective_function_v2()` from `src/objective/functions.py:24–302`:
- **Formula:** `Combined_Score = 0.6 * Similarity + 0.4 * Accuracy`
- Similarity = EMD + correlation preservation
- Accuracy = TRTS framework w/ RandomForest (`n_estimators=50`, `max_depth=10`)
- Two pruning checkpoints

### 4.3 Editing Objective Function
- Change weights: `src/objective/functions.py:24–26`
- Replace classifier
- Custom objective via `custom_objective_fn` to `optimize_models_batch()`

### 4.4 Hyperparameters Being Tuned
Tables from `src/models/search_spaces.py:119–458` including constraints.

### 4.5 Editing Search Spaces
Examples for modifying ranges, adding parameters, fixing parameters.

### 4.6 Expanding After Trial Run
1. Review `study.best_params`
2. Expand boundaries if values hit edge
3. Fix low‐importance parameters
4. Move from debug → full mode

### 4.7 Outputs Produced
`best_parameters.csv`, summary CSVs, study objects, visualizations.

---
## Section 5: Model Evaluation & Comparison (Pipeline Section 5)

### 5.1 Retraining with Best Parameters
`train_models_batch_with_best_params()` (`src/models/batch_training.py:378–649`).

### 5.2 Comprehensive Evaluation
Three dimensions:
- **Quality** — statistical similarity, JSS, correlation, PCA, ML utility
- **TRTS** — 4 scenarios (TRTR, TSTS, TRTS, TSTR), 30+ metrics
- **Privacy** — DCR, NNDR, memorization, re‐identification

### 5.3 How Best Model Is Chosen
`src/visualization/section5.py:264`:
- Primary: model with max `Combined_Score`
- Secondary: Privacy score

### 5.4 Outputs
All Section 5 artifacts: TRTS analysis, privacy dashboard, ROC/PR, etc.

---
## Section 6: Generating Samples

### 6.1 In‐Session
Access trained model, call `model.generate(n_samples)`.

### 6.2 Retraining from Saved Parameters
Load `best_parameters.csv`, recreate model, train, generate.

### 6.3 Model Persistence
- **Full:** CTGAN, TVAE, CopulaGAN (pickle)
- **Metadata only:** CTAB‐GAN, CTAB‐GAN+

---
## Section 7: Metrics Glossary

Technical + plain English for each metric.

### 7.1 Statistical Fidelity Metrics
JSS, Wasserstein Distance, Correlation Preservation, PCA Similarity, MI Preservation

### 7.2 ML Utility (TRTS)
30+ classification metrics.

### 7.3 Privacy Metrics
DCR, NNDR, Memorization, Re‐ID, Privacy Score

### 7.4 Quality Labels
- **Excellent:** ≥ 0.80
- **Good:** 0.60–0.79
- **Fair:** 0.40–0.59
- **Poor:** < 0.40

---
## Section 8: Architecture Reference

- Annotated directory tree
- Operation → module → key function map
- Results directory structure

---
## Critical Files
| File | Used For |
|------|-----------|
| README.md | Installation, system requirements |
| src/config.py | Config schema |
| src/models/search_spaces.py | Hyperparameter search spaces |
| src/objective/functions.py | Objective function |
| src/visualization/section5.py | Best model selection |
| src/evaluation/quality.py | Quality metrics |
| src/evaluation/trts.py | TRTS metrics |
| src/evaluation/privacy.py | Privacy metrics |
| src/models/batch_training.py | Training & best‐param retraining |
| src/utils/parameters.py | Parameter save/load |
| src/models/model_factory.py | ModelFactory.create() |
| src/data/eda.py | EDA function |

---
## Verification Checklist
1. Confirm referenced file paths exist
2. Validate code examples
3. Validate metric definitions
4. Confirm search space ranges
5. Confirm objective function formula
6. Render markdown to verify formatting

