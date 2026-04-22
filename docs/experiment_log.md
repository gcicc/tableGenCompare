# Experiment log

Running log of notable end-to-end runs, refactors, and platform changes.
Newest entries at the top. Keep entries short; link out to the PR or plan
file for detail.

## Entry template

```markdown
## YYYY-MM-DD — <short title>

- **What:** one-line summary.
- **Context:** why it happened (incident, planned work, hypothesis).
- **Outcome:** what changed on disk; links to PRs / plan files.
- **Next:** open follow-ups, if any.
```

---

## 2026-04-22 — Collinearity reducer ported from sibling

- **What:** Ported the residual re-parameterization engine from
  `multi-table-gen-compare/src/data/collinearity.py` into
  `tableGenCompare/src/data/collinearity.py` (byte-identical) plus
  supporting helpers (`_cramers_v`, `_eta`, `mixed_association_matrix` in
  `src/data/eda.py`), the `association_preservation` scorecard metric in
  `src/evaluation/fidelity.py`, §2.2b / §3.2 / §5.2 wiring across all four
  STG-Driver notebooks, and a new radar-chart axis via
  `Fidelity_Assoc_Preservation`.
- **Context:** Generators on breast-cancer were learning
  `mean_radius`, `mean_perimeter`, and `mean_area` as independent axes,
  losing the physical near-determinism. Sibling project already had a
  residual-reparam fix; keeping the two repos in sync was the goal.
- **Outcome:**
  - New file `src/data/collinearity.py` (byte-identical to sibling).
  - Extended `src/data/eda.py`, `src/evaluation/fidelity.py`,
    `src/evaluation/sdac_metrics.py`, `src/visualization/section2.py`,
    `src/visualization/section5.py`.
  - Four driver notebooks patched with identical §2.2b / §3.2 / §5.2
    cells (hashed and verified identical).
  - Sibling project got the `flagged_columns` heatmap parameter
    backported and a "Sync with tableGenCompare" governance section in
    `docs/collinearity-reduction.md`.
  - Round-trip on real breast-cancer data reconstructs to 3e-16 relative
    error.
  - See `docs/collinearity-reduction.md`.
- **Next:**
  - Run the full pipeline on SageMaker to populate
    `restoration_health.csv` for all 8 generators.
  - Backport any post-merge fixes from whichever repo finds them first
    (paired-PR rule).

---

## <historical entries predate this log>

Earlier project history is captured in the commit log and in
`docs/Project-Evolution-Timeline.md` (for pre-SDAC development milestones).
