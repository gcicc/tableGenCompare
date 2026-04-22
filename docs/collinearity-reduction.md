# Collinearity-driven residual re-parameterization

Near-deterministic column pairs (`perimeter`/`radius`, `weight`/`bmi`,
`label`/`code`) wreck downstream synthesizers: the generator tries to learn
both axes independently, they drift apart, and cross-column associations that
were near-1 in real data collapse in synthetic output.

`tableGenCompare` handles those pairs by **re-parameterizing** rather than
naively dropping one side. At training time we factor the redundant column
into a deterministic prediction from its partner plus a residual; the
generator learns the *residual* as an independent-ish primitive. Post-sample
we reconstruct the original column from the synthesized partner plus the
synthesized residual — preserving both the near-perfect relationship *and*
its realistic imperfection.

## Where this lives

| Component | Path |
|-----------|------|
| Engine | `src/data/collinearity.py` |
| Association helpers (`_cramers_v`, `_eta`, `mixed_association_matrix`) | `src/data/eda.py` |
| Scorecard metric `association_preservation` | `src/evaluation/fidelity.py` |
| Heatmap highlighter (`flagged_columns` parameter) | `src/visualization/section2.py::create_mixed_association_heatmap` |
| Radar integration (`Fidelity_Assoc_Preservation` axis) | `src/visualization/section5.py`, `src/evaluation/sdac_metrics.py` |
| Driver-notebook wiring | §2.1 config, §2.2b fit, §2.3 EDA, §3.2/§5.2 restore + health CSV |

The feature is ported from the sibling project `multi-table-gen-compare` and
kept byte-identical where possible. See [Sync governance](#sync-with-multi-table-gen-compare)
below.

## How it works

One threshold (**0.975** by default). Only pairs at or above it are treated;
everything below is left for the generator to handle directly.

Four strategies, one per pair kind, selected automatically with per-pair
override hooks:

| Pair kind | Decomposition | Reconstruction |
|-----------|---------------|----------------|
| numeric × numeric (additive; default when any negative) | `e = p − (a + b·r)` | `p* = (a + b·r*) + e*` |
| numeric × numeric (multiplicative; default when both > 0) | `e = log p − (a + b·log r)` | `p* = exp((a + b·log r*) + e*)` |
| categorical × categorical | partner-value → mode-dropped lookup | `p* = lookup[r*]` (exact at V=1) |
| numeric × categorical (numeric dropped) | `e = num − group_mean(cat)` | `num* = group_mean(cat*) + e*` |
| numeric × categorical (categorical dropped) | nearest-mean lookup | `cat* = argmin_k |group_mean(k) − num*|` |

**Positivity auto-detect:** if both numeric columns are strictly positive in
real data → multiplicative; else additive. Overridable per pair.

**Drop policy:** higher missingness loses; alphabetical tiebreak.
Deterministic and override-able.

**Residual columns** are written back into the reduced frame with the reserved
suffix `__resid`. The synthesizer sees them as ordinary continuous columns and
learns their distribution alongside the rest of the schema. On restoration
they are consumed and removed — the final synthetic schema matches the real
data.

**Transitive chains** (A↔B↔C all above threshold): residuals are restored in
topological order so partners come back first. Error propagates through the
chain's weakest link; bump `collinearity_threshold` to 0.99+ if chains get
deep and this matters.

## Pipeline placement

```
§2.1 NOTEBOOK_CONFIG
     ↓
§2.2  load_and_preprocess_from_config  →  data (full schema)
     ↓
§2.2b fit_collinearity_reducer         →  COLLIN_CTX, data_reduced
       (review decisions table; set collinearity_overrides and re-run if needed)
     ↓
§2.3  run_comprehensive_eda            →  heatmap w/ flagged columns in red
     ↓
§3.1  train_models_batch (on reduced schema)
§3.2  restore_dropped → synthetic_data_*        →  evaluate → restoration_health.csv
     ↓
§4    Staged HPO (on reduced schema; residual is the training signal)
     ↓
§5.1  train_models_batch_with_best_params (on reduced schema)
§5.2  restore_dropped → synthetic_*_final       →  evaluate → restoration_health.csv
```

**Key design point:** HPO in §4 tunes on the reduced schema — the generator
learns the residual distribution directly, and §5 retrains against the same
signal. Restoration happens exactly once per phase, just before evaluation,
so §3/§5 metrics are computed on the full real-schema frame and are directly
comparable across sections.

## NOTEBOOK_CONFIG keys

```python
NOTEBOOK_CONFIG = {
    # ... existing keys ...
    "collinearity_enabled":   True,
    "collinearity_threshold": 0.975,
    "collinearity_overrides": {},   # see override format below
}
```

**Override format** (keys are alphabetically ordered column pairs):

```python
NOTEBOOK_CONFIG["collinearity_overrides"] = {
    ("BMIBL", "HEIGHTBL"): {"strategy": "additive"},     # force additive
    ("USUBJID", "code"):   {"skip": True},                # leave alone
    ("foo", "bar"):        {"keep": "foo"},               # override drop policy
}
```

Supported per-pair keys:
- `"skip": True` — don't touch this pair
- `"keep": "<col>"` — force this column to be kept (drop the other)
- `"strategy": "additive" | "multiplicative"` — numeric × numeric only

Review the §2.2b decisions table first; add overrides; re-run the notebook.
The decisions table has one row per pair with `col_a, col_b, kind, score,
strategy, dropped, kept, residual_col, note`.

## Validation: restoration health

`§3.2` and `§5.2` emit `restoration_health.csv` comparing real vs synth
association on every treated pair:

```
dropped      partner        strategy                 real_assoc  synth_assoc  delta  model
mean_radius  mean_perimeter residual_multiplicative  0.9979      0.9971       0.0008 ctgan
```

**A large `delta` is the diagnostic.** If the residual reparam worked,
`synth_assoc` stays within a small tolerance of `real_assoc`. A large gap
means the generator did not preserve the residual as designed — check the
per-column fidelity of the `__resid` column in the scorecard and review that
model's detection AUC.

## Scorecard axis: `Fidelity_Assoc_Preservation`

`association_preservation` (defined in `src/evaluation/fidelity.py`) reports:

```
1 − mean(|A_real − A_synth|)   over pairs where |A_real| > 0.3
```

restricted to the mixed-association matrix (Pearson / Cramér's V / eta). The
`|A_real| > 0.3` filter prevents marginal-independent samplers from spuriously
winning: without it, most off-diagonal real associations are ~0 and their
synth is also ~0, which is not a meaningful signal.

`src/evaluation/sdac_metrics.py` computes this as `Fidelity_Assoc_Preservation`
on every model; `src/visualization/section5.py::create_sdac_radar_chart`
picks it up automatically as a new fidelity component.

## Disabling the feature

Set `NOTEBOOK_CONFIG["collinearity_enabled"] = False`. The reducer returns an
empty context, the decisions table is empty, no residual columns are created,
no heatmap tick labels are colored red, and §3.2 / §5.2 emit no
`restoration_health.csv`. Downstream metrics are computed exactly as before
this feature landed.

## Sync with `multi-table-gen-compare`

This project and `multi-table-gen-compare` maintain byte-identical copies of
the engine, helpers, and scorecard metric. Any change to the files listed
below **must be applied to both repos in a paired PR** (not sequential PRs —
drift is easy to introduce):

| File | Rule |
|------|------|
| `src/data/collinearity.py` | Byte-identical across both repos |
| `src/data/eda.py::_cramers_v` | Byte-identical |
| `src/data/eda.py::_eta` | Byte-identical |
| `src/data/eda.py::mixed_association_matrix` | Byte-identical |
| `src/evaluation/fidelity.py::association_preservation` | Byte-identical |
| `src/visualization/section2.py::create_mixed_association_heatmap` | `flagged_columns` parameter kept parallel |

Sibling doc: `multi-table-gen-compare/docs/collinearity-reduction.md` has the
reciprocal governance section and authoritative background on the strategies.

**Sanctioned divergences** (each deliberate; reasons documented):

| Sibling has | tableGenCompare has | Why |
|-------------|---------------------|-----|
| `src/data/reductions.py::CDISC_REDUCTIONS` (hand-curated SDTM code↔label map) | — | tableGenCompare datasets aren't CDISC-formatted. Revisit if CDISC data enters this project. |
| `src/data/reductions.py::reduce_tables` (multi-table orchestrator) | — | Single-table; call primitives directly from the notebook. |
| — | HPO (§4) pipeline and `Fidelity_Assoc_Preservation` wired into `sdac_metrics.py` / §5 radar | Sibling has no HPO and a different scorecard layout. |

## Known limitations

- **Pearson, not Spearman.** Highly non-linear but monotone pairs hide below
  the threshold. Swap `numpy.corrcoef` for `scipy.stats.spearmanr` in
  `_pair_score` if your data warrants it; residual strategies are unaffected.
- **Residual columns don't shrink the schema.** We swap `p` for `p__resid`,
  so column count is unchanged. For token-budget generators at large schemas,
  consider `collinearity_threshold: 0.99+` to limit which pairs get the
  residual treatment.
- **No `keep_priority` list.** Default is missingness + alphabetical. Easy to
  extend via `_choose_drop` (10 lines) if needed.
- **Transitive chains error propagation.** See [How it works](#how-it-works);
  bump threshold to 0.99+ for deep chains.
