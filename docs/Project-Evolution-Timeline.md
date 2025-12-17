---
title: "Project Evolution"
output: html_document
date: "2025-12-16"
---

# Clinical Synthetic Data Generation Framework - Project Evolution

**A Timeline of Development Across Branches**

---

## Executive Summary

This document traces the evolution of the Clinical Synthetic Data Generation Framework from its initial proof-of-concept through multiple rounds of AWS deployment refinement. The project has progressed through four major development phases, each represented by distinct Git branches that capture significant architectural and methodological improvements.

**Branch Overview:**

- **old-main** (Aug 2025) - Initial successful local execution
- **AWS_Round1** (Sep-Oct 2025) - First AWS deployment and multi-dataset expansion
- **main** (Oct-Nov 2025) - Stabilized AWS-compatible release
- **AWS_Round2** (Dec 2025) - Advanced architecture with modular design and comprehensive evaluation

---

## Branch Comparison Matrix

| Aspect | old-main | AWS_Round1 | main | AWS_Round2 |
|--------|----------|------------|------|------------|
| **Status** | Archived | Archived | Stable | Active Development |
| **Environment** | Local Laptop | AWS SageMaker | AWS SageMaker | AWS SageMaker |
| **Models Supported** | 6 (All) | 3 (CTGAN, CopulaGAN, TVAE) | 3 (CTGAN, CopulaGAN, TVAE) | 6 (All) |
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

## Recommended Usage by Branch

### When to Use **old-main**

- ❌ **Not recommended** - Historical reference only
- Academic interest in original implementation
- Understanding initial design decisions

### When to Use **AWS_Round1**

- ❌ **Not recommended** - Superseded by main
- Learning about AWS migration challenges
- Understanding multi-dataset harmonization

### When to Use **main**

- ✅ **Production deployments** requiring stability
- Environments with strict version control
- 3-model configuration (CTGAN, CopulaGAN, TVAE)
- Conservative deployments avoiding bleeding-edge features

### When to Use **AWS_Round2** (Current)

- ✅ **All new development and research**
- Advanced evaluation requirements (30+ metrics)
- Automated workflow needs
- Privacy risk assessment
- Modular architecture benefits
- Comprehensive visualization suite
- Future-proof codebase

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

- [ ] Resolve 3-model limitation in AWS environment
- [ ] Expand to additional healthcare datasets
- [ ] Add differential privacy mechanisms
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

1. **Start with main branch** to understand stable production version
2. **Review AWS_Round2 README** for current architecture
3. **Check docs/Task-*.md files** for detailed technical plans
4. **Run test notebooks** to verify environment setup
5. **Read this timeline** to understand project evolution

### Quick Start Commands

```bash
# Clone repository
git clone https://github.com/gcicc/tableGenCompare.git
cd tableGenCompare

# Checkout development branch
git checkout AWS_Round2

# Review documentation
cat README.md
cat docs/Project-Evolution-Timeline.md

# Set up environment (AWS SageMaker)
pip install -r requirements.txt

# Run example notebook
jupyter notebook STG-BreastCancer-testing.ipynb
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

**Document Version:** 1.0
**Last Updated:** December 16, 2025
**Current Active Branch:** AWS_Round2
**Production Branch:** main
**Framework Version:** 4.0 (Modular Architecture)
