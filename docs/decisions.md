# Architectural decisions

ADR-style log of non-obvious design choices in `tableGenCompare`. Not a
changelog — just the ones that would puzzle a future reader.

---

## 1. Single-table scope, separate repo for multi-table

**Context.** Real clinical-trial data is relational (CDISC SDTM has 20+
linked tables). ML benchmark datasets (UCI, Kaggle) are almost always
single-table. Generators that handle both (SDV HMA, REALTabFormer) exist
but add significant complexity.

**Decision.** `tableGenCompare` is single-table only. The relational
extension lives in a separate sibling repo, `multi-table-gen-compare`.

**Reasoning.** The two projects target different stakeholders (ML
practitioner vs. pharma pipeline engineer), different generators (8 vs.
~5), different metrics (SDAC vs. SDAC + relational axes). Keeping them
separate prevents the single-table path from inheriting multi-table's
constraint machinery and lets each iterate at its own pace.

**Revisit when.** If a shared pharma stakeholder wants one notebook that
toggles between modes. So far, everyone wants one or the other.

---

## 2. `setup.py` re-export shim, not a proper package install

**Context.** Notebooks at the repo root start with `from setup import *`.
A naive packaging of `src/` as an installable wheel would require changing
every notebook.

**Decision.** Keep `setup.py` as a thin re-export module that does
`from src.data.preprocessing import *`, `from src.models.registry import *`,
etc. No `pip install -e .`; just `sys.path` / PYTHONPATH.

**Reasoning.** Notebooks stay portable across dev environments (local,
SageMaker) without an install step. Reorganizations inside `src/` need only
re-export edits in `setup.py`, not notebook edits. setup.py has grown
steadily smaller as code has moved into `src/` — it is the boundary, not the
shared bin.

**Cost.** Import order matters in `setup.py`; occasional `ImportError`
symptoms point to ordering bugs there, not real module errors.

**Revisit when.** We need to share code with another project as an
importable package. At that point, either extract shared code into a
separate wheel or make the whole repo installable.

---

## 3. SageMaker envs live on EBS, not instance-local disk

**Context.** SageMaker notebook instances reset their root disk on
stop/start. Re-installing the env every time would be 15+ minutes lost.

**Decision.** `setup_env.sh` creates the conda env at
`/home/ec2-user/SageMaker/.envs/tablegen` (the persistent EBS volume).
`on-start.sh` re-registers the Jupyter kernel each boot.

**Reasoning.** EBS survives instance stop/start. The conda env is
byte-for-byte reproducible after a stop.

**Gotcha.** Don't `conda create` into the default location on the root
disk — first stop/start will wipe it.

---

## 4. Python 3.10 on SageMaker, Python 3.12+ locally

**Context.** SageMaker's bundled Amazon Linux has known-good
compatibility with Python 3.10 (the setup_env.sh uses it explicitly).
Local dev (Windows, macOS) is typically 3.12+.

**Decision.** Pin SageMaker env to 3.10. Don't pin a version for local dev.

**Reasoning.** SDV 1.36, torch CUDA wheels, and GANerAid all install
cleanly on 3.10 in SageMaker. 3.12+ local dev hits occasional transitive
dep friction but works for most of the codebase (notebook code, metric
evaluation, visualizations). Don't block local iteration on the SageMaker
pin.

**Revisit when.** Amazon Linux SageMaker image ships a higher Python
default.

---

## 5. 8 generators registered by default

**Context.** Benchmarking fewer generators means less evidence; more means
more moving parts to keep working.

**Decision.** CTGAN, TVAE, CopulaGAN, CTAB-GAN, CTAB-GAN+, PATE-GAN,
MEDGAN, GANerAid — all registered, all enabled by default in every
driver's `NOTEBOOK_CONFIG["models_to_run"]`.

**Reasoning.** Each generator pulls its weight on at least one dataset
(CTAB-GAN on mixed; CopulaGAN on correlated-numeric; PATE-GAN as the DP
baseline; etc.). Dropping any of them weakens the comparison story for a
meaningful use case.

**Cost.** Full §4 HPO across all 8 is a 4–6 hour job on the smallest
dataset (breast-cancer). Staged optimization (smoke → pilot → full) lets a
user decide where to stop.

---

## 6. Staged HPO: smoke → pilot → full

**Context.** Running the full HPO budget blind is wasteful when one model
has a broken search space or trains 10× slower than the others.

**Decision.** §4 runs in three modes:
1. **Smoke** (`tuning_mode: "smoke"`) — 10 trials per model, short budgets.
   Surfaces time estimates and flags erroring models.
2. **Pilot** (implicit, always runs) — emits diminishing-returns analysis
   and a smoke recommendations table.
3. **Full** (`tuning_mode: "full"`, requires manual continuation cell) —
   full budget, possibly with additional batches.

**Reasoning.** The user sees early evidence of where budget should go and
can stop or redirect before committing hours of compute.

**Revisit when.** A generator's smoke/full gap becomes predictable enough
that we can skip the staged step.

---

## 7. SDAC framework for evaluation

**Context.** Synthetic-data evaluation metrics sprawl: distributional
fidelity, classifier utility, privacy, fairness, explainability, relational
integrity, mode collapse, …

**Decision.** Adopt **SDAC** (Synthetic Data Anonymity & Credibility) from
the SEARCH Consortium as the top-level taxonomy. Five axes: Privacy,
Fidelity, Utility, Fairness, XAI. Each axis has a small set of metrics that
roll up into a per-axis composite, then into a radar polygon per model.

**Reasoning.** Gives stakeholders a stable vocabulary. Radar chart + per-axis
composites are a teachable summary; the flat-dict `sdac_evaluation_summary.csv`
underneath preserves every metric for deeper review. See [evaluation.md](evaluation.md).

---

## 8. Collinearity reducer in §2, restore in §3.2 / §5.2

**Context.** Near-deterministic column pairs (perimeter/radius,
label/code) cause generators to learn both axes independently. Synth output
has the marginals right but loses the cross-column relationship.

**Decision.** Fit the reducer once in §2.2b on the preprocessed real frame.
§3 (demo), §4 (HPO), and §5 (final) all train on the **reduced** schema.
Restore dropped columns via `restore_dropped` right before §3.2 / §5.2
evaluation — so metrics are computed on the full real-schema frame and §3/§5
are directly comparable.

**Reasoning.** HPO tunes against the actual training signal (the residual
column). §5 retrains on the same schema. Restoration happens exactly once
per phase at the schema boundary. No train/eval schema mismatch.

**Alternative considered.** Pushing reduction into each model's
`sanitize_output` hook. Rejected because it would thread the context
through 8 wrappers and hide the decisions from the notebook user.

See [collinearity-reduction.md](collinearity-reduction.md) for the full
strategy table, override format, and sync-with-sibling governance.

---

## 9. Driver-notebook discipline: per-dataset diffs confined to §2

**Context.** Four driver notebooks
(`STG-Driver-{breast-cancer2,Alzheimer2,diabetes2,liver-train2}.ipynb`)
all run the same pipeline. Over time, per-dataset tweaks can drift into §3,
§4, or §5 and become silent behavioral differences.

**Decision.** Only §2.1 (`NOTEBOOK_CONFIG`) is allowed to differ across
drivers. §3, §4, §5 cells must be byte-identical across all four notebooks.
Per-dataset behavior is expressed as config values, not cell code.

**Reasoning.** Cross-dataset comparisons are only meaningful if the code is
the same. Patches land in one place, not four. Verified by post-edit
hashing of §3/§4/§5 cells.

**Enforcement.** After any cross-notebook edit, diff or hash the §3/§4/§5
cells across drivers. They must match.

---

## 10. Association metric: byte-identical to sibling, not dython

**Context.** `src/evaluation/association.py::compute_mixed_association_matrix`
wraps `dython` and is used throughout the existing pipeline. Sibling
`multi-table-gen-compare` has its own `mixed_association_matrix` in
`src/data/eda.py` using hand-rolled Pearson / Cramér's V / eta.

**Decision.** Both coexist. The collinearity reducer and the
`association_preservation` scorecard metric use the sibling's
implementation (added verbatim to `src/data/eda.py` in this project).
Everything else keeps using the dython-wrapped version.

**Reasoning.** Keeps `association_preservation` byte-identical across the
two repos so the scorecard axis reports the same values on the same data.
Deprecating the dython version in favor of the sibling's would be a bigger,
separate refactor.

**Revisit when.** We do a metric-module consolidation pass.
