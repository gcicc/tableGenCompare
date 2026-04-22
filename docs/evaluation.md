# Evaluation framework (SDAC)

Every synthetic dataset produced by a model in this repo is graded under
**SDAC — Synthetic Data Anonymity & Credibility** — five axes, one composite
score per axis, one aggregate score per model, visualized as a radar chart
and a heatmap in §5.

- **Privacy** — how much of the real data is memorized/leaked?
- **Fidelity** — how closely do synthetic columns and joint structure match
  real?
- **Utility** — does a downstream classifier trained on synthetic transfer
  to real?
- **Fairness** — does the synth preserve (or introduce) disparities across
  a protected attribute?
- **XAI** — do feature importances and SHAP explanations match between
  real- and synth-trained models?

Orchestrated by `src/evaluation/sdac_metrics.py::compute_sdac_tabular_metrics`;
one row per model in `sdac_evaluation_summary.csv`.

---

## Axes and metrics

### Privacy

| Metric | Flat-dict key | Higher is... | Implementation |
|--------|---------------|--------------|----------------|
| DCR (Distance to Closest Record) | `Privacy_DCR` | better | `src/evaluation/privacy.py::calculate_privacy_metrics` |
| NNDR (Nearest-Neighbor Distance Ratio) | `Privacy_NNDR` | better | same |
| IMS (Individual Memorization Score) | `Privacy_IMS` | better | same |
| Re-ID risk | `Privacy_ReID_Risk` | lower | same |
| Composite Privacy Score | `Privacy_Score` | better (0–1) | same |
| MIA AUC | `Privacy_MIA_AUC` | 0.5 is best | `src/evaluation/privacy.py::compute_mia_auc` |

### Fidelity

| Metric | Flat-dict key | Higher is... | Implementation |
|--------|---------------|--------------|----------------|
| JSD mean similarity | `Fidelity_JSD` | better | `src/evaluation/sdac_metrics.py::_compute_jsd_mean` |
| KS mean | `Fidelity_KS` | lower | `src/evaluation/fidelity.py::compute_ks_statistic` |
| KL mean | `Fidelity_KL` | lower | `src/evaluation/fidelity.py::compute_kl_divergence` |
| Correlation similarity | `Fidelity_Corr_Sim` | better | `src/evaluation/sdac_metrics.py::_compute_correlation_similarity` |
| Wasserstein mean | `Fidelity_WD` | lower | `src/evaluation/fidelity.py::compute_wasserstein_mean` |
| Detection AUC | `Fidelity_Detection_AUC` | 0.5 is best | `src/evaluation/fidelity.py::compute_detection_auc` |
| Contingency similarity | `Fidelity_Contingency_Sim` | better | `src/evaluation/fidelity.py::compute_contingency_similarity` |
| **Association preservation** | `Fidelity_Assoc_Preservation` | **better** | `src/evaluation/fidelity.py::association_preservation` — 1 − mean\|A_real − A_synth\| over pairs where \|A_real\| > 0.3 |

`Fidelity_Assoc_Preservation` is the scorecard axis for the collinearity
port. See [collinearity-reduction.md](collinearity-reduction.md) for why the
0.3 signal filter matters (it prevents marginal-independent samplers from
spuriously winning).

### Utility (TRTS framework)

Four scenarios across multiple classifiers (XGBoost primary; RF, LR
secondary):

- **TRTR** (Train Real → Test Real) — baseline.
- **TRTS** (Train Real → Test Synth) — checks synth distribution.
- **TSTR** (Train Synth → Test Real) — **the one that matters** for downstream
  use.
- **TSTS** (Train Synth → Test Synth) — internal consistency check.

Flat-dict keys of the form `Utility_TSTR_Acc_RF`, `Utility_ML_Efficacy`,
`Utility_SRA`. Implementation: `src/evaluation/trts.py`,
`src/evaluation/trts_framework.py`.

### Fairness

| Metric | Flat-dict key | Implementation |
|--------|---------------|----------------|
| Demographic Parity Diff | `Fairness_Dem_Parity` | `src/evaluation/fairness.py::compute_fairness_metrics` |
| Equalized Odds Diff | `Fairness_Eq_Odds` | same |
| Disparate Impact Ratio | `Fairness_Disp_Impact` | same |

Requires `NOTEBOOK_CONFIG["protected_col"]`. Skipped (N/A) otherwise.

### XAI (explainability)

| Metric | Flat-dict key | Implementation |
|--------|---------------|----------------|
| Feature Importance Correlation | `XAI_Feat_Imp_Corr` | `src/evaluation/xai_metrics.py::compute_feature_importance_correlation` |
| SHAP distance | `XAI_SHAP_Dist` | `src/evaluation/xai_metrics.py::compute_shap_distance` |

SHAP is optional (`shap` package); failures degrade to NaN and are logged.

---

## Composite scoring

Per-axis composites are built in `src/visualization/section5.py::create_sdac_radar_chart`:

- **Privacy** = `Privacy_Score` (already composite).
- **Fidelity** = mean of the in-scope fidelity components with known "higher
  is better" polarity — `Fidelity_JSD`, `1 - Fidelity_KS`,
  `1 - |Fidelity_Detection_AUC - 0.5| * 2`, `Fidelity_Corr_Sim`, and
  **`Fidelity_Assoc_Preservation`**.
- **Utility** = `Utility_ML_Efficacy`.
- **Fairness** = mean of `1 - Fairness_Dem_Parity` and `Fairness_Disp_Impact`.
- **XAI** = `XAI_Feat_Imp_Corr`.

Radar polygon area (normalized to the 5-axis maximum) ranks models on a
single scalar when needed.

---

## Notebook surfaces

| Section | What is produced | Main output path |
|---------|------------------|------------------|
| §2.3 | EDA: distributions, association heatmap (flagged-column highlighting from collinearity reducer), target analysis, column analysis | `results/{dataset}/{date}/Section-2/` |
| §3.2 | Default-param scorecard across all models; `restoration_health.csv` for collinearity reducer diagnostics | `results/{dataset}/{date}/3/` |
| §4 | Optuna study per model; pilot-mode estimates; full-mode continuation | `results/{dataset}/{date}/Section-4/` |
| §5.2 | Optimized-param scorecard; radar + heatmap; `sdac_evaluation_summary.csv`; `restoration_health.csv` | `results/{dataset}/{date}/5/` |

Each model also gets a per-model subfolder under §3 and §5 with its own
dashboards: per-column fidelity, TRTS scenario comparison, correlation
comparison, PCA comparison, privacy dashboard, and — when the collinearity
reducer is active — the per-pair `restoration_health.csv` contributed by
that model.

---

## What's *not* in the scorecard

- **Training loss curves** — optional per-model dashboards; not part of the
  axis composition. Useful for diagnosing non-convergence, not for ranking.
- **Runtime / cost** — emitted in §4 pilot output but not a scorecard axis.
- **Relational / cross-table axes** — single-table; N/A. See the sibling
  `multi-table-gen-compare` project for that extension.
