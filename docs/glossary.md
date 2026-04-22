# Glossary

Terms that recur across the codebase and docs.

## Synthetic-data generation

**Generator** — model that learns a data distribution from a real
training set and produces new rows drawn from the learned distribution.
This repo compares eight: CTGAN, TVAE, CopulaGAN, CTAB-GAN, CTAB-GAN+,
PATE-GAN, MEDGAN, GANerAid. See [models.md](models.md).

**GAN** (Generative Adversarial Network) — generator + discriminator
trained adversarially. Most of our models are GANs in one form or another.

**VAE** (Variational Autoencoder) — encoder-decoder architecture that
learns a latent distribution and samples from it. TVAE is the VAE in this
lineup.

**Copula-based** — separate the marginal distributions from the joint
dependency structure and learn them independently. CopulaGAN combines
copula preprocessing with a GAN.

**Diffusion** — generative family that learns to reverse a noising
process. Not yet in the repo; see [models.md §Planned additions](models.md).

## Evaluation

**SDAC** — Synthetic Data Anonymity & Credibility. The 5-axis evaluation
framework used throughout §3 and §5: Privacy, Fidelity, Utility, Fairness,
XAI. See [evaluation.md](evaluation.md).

**TRTS** — Train-Real / Test-Synth (and its permutations). Four scenarios
to stress utility preservation:
- **TRTR** = Train Real, Test Real (baseline)
- **TRTS** = Train Real, Test Synth (does synth look like a valid test set?)
- **TSTR** = Train Synth, Test Real (**the scenario that matters** — does
  a classifier trained on synth generalize to real?)
- **TSTS** = Train Synth, Test Synth (internal consistency check)

**Fidelity** — how well the synthetic data reproduces the marginal and
joint structure of the real data. Measured by KS, KL, Wasserstein, JSD,
Detection AUC, correlation similarity, and
**`association_preservation`** (the scorecard axis added by the
collinearity port).

**Privacy / Anonymity** — how much the synthetic data leaks about the
real data. Measured by DCR, NNDR, IMS, Re-ID risk, MIA AUC.

**DCR** (Distance to Closest Record) — per-row distance from each synth
row to the nearest real row. Larger is better for privacy.

**NNDR** (Nearest-Neighbor Distance Ratio) — ratio of the nearest real
neighbor's distance to the *second*-nearest. Close to 1 means the synth
row sits equidistant from two real rows (good); close to 0 means it
collapsed onto one real row (bad).

**MIA** (Membership Inference Attack) — classifier that tries to infer
whether a given row was in the generator's training set. AUC of 0.5 is
ideal (random guessing); higher AUC means the generator memorized
identifiable rows.

**Utility** — how useful the synthetic data is for a downstream ML task.
In this repo: does a classifier trained on synth (TSTR) hit accuracy /
F1 / AUROC close to a classifier trained on real (TRTR)?

**ML Efficacy** — composite utility score rolling TSTR performance into
a single 0–1 number.

**SRA** (Synthetic Ranking Agreement) — do the generators' relative
utility rankings on synth data match their rankings on real data?
Higher is better.

**Detection AUC** — how easily a classifier can distinguish real from
synth rows. AUC of 0.5 means indistinguishable (great); higher means the
classifier easily tells them apart (bad for fidelity).

**JSD** (Jensen-Shannon Divergence) — symmetric KL-derived divergence
between two distributions. Bounded in [0, 1]; lower is better. We report
`1 - JSD` as a similarity score.

## Fairness

**Demographic Parity Difference** — absolute difference in positive
prediction rates across groups of a protected attribute. Lower is more
fair.

**Equalized Odds Difference** — max absolute difference in true-positive
rate and false-positive rate across groups. Lower is more fair.

**Disparate Impact Ratio** — ratio of positive-prediction rates between
groups. Higher (closer to 1) is more fair.

## Explainability

**XAI** — Explainable AI. Here: feature-importance correlation between
real- and synth-trained classifiers + SHAP-distance between their
explanations.

**SHAP** — Shapley-value-based per-prediction feature attribution. Our
XAI axis includes a real-vs-synth SHAP-vector distance.

## Data-access terms

**DUA** (Data Use Agreement) — a legal contract between a data provider
and a recipient that governs how sensitive data (clinical records, PHI,
patient-identifiable information) may be used, retained, and shared.
Common in healthcare research. Synthetic tabular data is one strategy for
enabling work *around* a DUA: the synthetic output is not the real data
and typically falls outside the DUA's scope, so it can be distributed to
people or CI systems that the DUA wouldn't cover.

**PHI** (Protected Health Information) — patient-identifiable health data
regulated under HIPAA in the US. Most clinical datasets involve PHI, which
is why DUAs and secure enclaves exist in the first place.

**Secure enclave** — a compute environment with controlled network and
data egress where PHI / DUA-bound data can be analyzed. Running everything
inside an enclave is slow and collaborator-hostile; synthetic drop-ins
(one use case for this repo) let most work happen outside.

## Pipeline / infrastructure

**Driver notebook** — one of `STG-Driver-{breast-cancer2, Alzheimer2,
diabetes2, liver-train2}.ipynb`. Per-dataset variation is confined to §2;
§3/§4/§5 are byte-identical across all four. See [decisions.md §9](decisions.md).

**Staged HPO** — the §4 optimization strategy: smoke → pilot → full. See
[decisions.md §6](decisions.md).

**Checkpoint** — `SectionCheckpoint` persists intermediate state between
notebook runs so re-running §1 doesn't re-do §2 preprocessing.
`FRESH_START = True` in §1 wipes them.

**Collinearity reducer** — §2.2b pipeline step that re-parameterizes
near-deterministic column pairs. See
[collinearity-reduction.md](collinearity-reduction.md).

**Residual column** — a column of the form `foo__resid` added by the
collinearity reducer. The reducer fits `foo = f(partner) + residual` on
real data; the generator learns `residual`; the notebook reconstructs
`foo` from `partner + residual` before evaluation.

**`restore_dropped`** — the inverse of the collinearity reducer, applied
to synthetic output before evaluation. Ensures §3/§5 metrics are computed
on the full real-schema frame.

**`restoration_health.csv`** — per-pair diagnostic emitted in §3.2 and
§5.2. `real_assoc`, `synth_assoc`, `delta` per treated pair; a large
`delta` flags that the generator didn't preserve the residual as designed.

## Sibling project

**`multi-table-gen-compare`** — sister repo at
`C:\ForGit\gcicc\multi-table-gen-compare`, targeting relational /
multi-table clinical data. Shares the collinearity engine and the
`association_preservation` metric byte-identically with this repo. See
[collinearity-reduction.md §Sync with multi-table-gen-compare](collinearity-reduction.md).
