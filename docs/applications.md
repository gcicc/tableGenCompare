# Applications of synthetic tabular clinical data

Why go to the trouble of generating synthetic versions of datasets we
already have? Five use cases the single-table benchmarking in this repo is
built to support.

## 1. ML pipeline validation under data-access restrictions

You're building an ML pipeline (feature engineering, model training,
evaluation harness) for a clinical dataset that sits behind a **Data Use
Agreement (DUA)** — a legal contract restricting how the real data can be
shared — or inside a secure enclave. You want your CI, your team's local
dev, and your external ML collaborators to exercise the pipeline end-to-end
without touching real patient data.

**What this repo gives you.** Per-dataset synthetic drop-ins with the same
schema, column types, and marginal distributions as the real data. Our
`sdac_evaluation_summary.csv` shows — explicitly, per metric — how closely
each candidate generator reproduces the real statistics.

**How to use it.** Run §5 to completion; export the best model's synth
output; distribute that CSV as the public-safe testbed. Every downstream
consumer gets consistent, reproducible data.

## 2. External collaboration without DUAs

You want to publish a benchmark, blog post, or tutorial that demonstrates
a modeling approach on realistic healthcare data. You can't distribute the
real data.

**What this repo gives you.** Models that preserve column-column
relationships at the scorecard-measurable level (see
`Fidelity_Assoc_Preservation` in [evaluation.md](evaluation.md)). When the
collinearity reducer is active, physically-meaningful near-deterministic
pairs (perimeter/radius, weight/BMI, label/code) survive end-to-end — not
just the marginals.

**How to use it.** Pick the model with the highest composite Fidelity
score on your dataset; generate and release at the size you need.

## 3. Privacy-preserving data augmentation

You have enough real data to train a decent downstream classifier but want
to explore whether adding synthetic rows improves generalization, rare-class
coverage, or calibration — without re-introducing the original subjects.

**What this repo gives you.** Privacy metrics (DCR, NNDR, IMS, Re-ID risk,
MIA AUC) per model, so you can pick the generator whose synth is most
*dissimilar* to real data at the row level while still being
distributionally faithful. PATE-GAN is included specifically as the
differentially-private baseline for this use case.

**How to use it.** Compare models on the Privacy axis of the radar chart,
then augment your train set with the chosen model's output and re-evaluate
your downstream classifier.

## 4. Model / scientist training on realistic data

You're teaching a class, running an onboarding exercise, or building a
tutorial. Real patient data is off-limits but public datasets are too
obviously "clean." You want something realistic-looking with real-looking
correlations, imbalance, and feature diversity.

**What this repo gives you.** Synthetic versions of four public healthcare
benchmarks (Alzheimer, breast-cancer, diabetes, liver) plus a documented
pipeline that lets students reproduce the exact synth generation if they
want to go deeper.

**How to use it.** Pick a dataset; run the full pipeline; distribute the
§5 synthetic output. The `docs/` folder doubles as a reading guide for
students interested in the generator landscape or the SDAC evaluation
framework.

## 5. Benchmarking generator research

You're a researcher publishing a new synthetic-data generator. You want to
compare against a stable baseline of established methods across multiple
datasets and multiple evaluation axes — not just "my model's AUROC beat
CTGAN's on one dataset."

**What this repo gives you.** Eight established generators with consistent
HPO and scorecard. A staged optimization pipeline that surfaces time
estimates before committing hours of compute. A scorecard that reports
fidelity, utility, privacy, fairness, and XAI axes separately so your
method's strength (or weakness) on any one axis is legible.

**How to use it.** Add your wrapper under
`src/models/implementations/`, register it, add an HPO search space, run
§3–§5 across the four datasets, compare on `sdac_evaluation_summary.csv`
and the radar charts. See [models.md](models.md) §Adding a new model.

## The real unlock

Synthetic tabular data is useful in direct proportion to how faithfully it
carries the joint structure that mattered in the real data. Marginals are
easy; pairwise correlations are medium; near-deterministic relationships
(physical constraints, code-label pairs) are the hardest and the first to
break. This repo's scorecard and collinearity reducer are the two levers
for diagnosing and fixing that last category.
