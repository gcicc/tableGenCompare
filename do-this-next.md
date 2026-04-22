# Do this next — AWS pickup

**Context.** Collinearity reducer + Association-Preservation metric landed
yesterday (2026-04-22) on `main`. Local Positron env was proven green on
imports. Tomorrow's job: validate end-to-end on SageMaker.

This file is an operational artifact — delete after the smoke run is green.

---

## 1. Pull the changes

Open a terminal in SageMaker JupyterLab and run:

```bash
cd ~/SageMaker/tableGenCompare

# Pull latest main
git fetch origin
git pull --ff-only origin main

# Update submodules (CTAB-GAN, CTAB-GAN-Plus, GANerAid commits may have moved)
git submodule update --init --recursive --force

# (Optional) confirm the three new commits landed
git log --oneline -6
#   expect the top 3 to be, in order:
#     <hash> Documentation: topic docs + README / USER-GUIDE updates
#     <hash> Add dython 0.7.x compat shim so tab-gan-metrics / GANerAid load on local envs
#     <hash> Add collinearity-driven residual re-parameterization (port from multi-table-gen-compare)
```

## 2. Environment sanity check

On SageMaker, `setup_env.sh` already does the `tab_gan_metrics` hand-edit
patch (SageMaker's baseline), so the `src/_compat_dython.py` shim will be a
**redundant no-op** — it only kicks in if `compute_associations` is missing
from `dython.nominal`. Safe to leave in place either way.

If the kernel isn't showing after the pull:

```bash
bash on-start.sh          # re-registers the Python (tablegen) kernel
```

If the conda env itself got wiped (rare but possible after long stops):

```bash
bash setup_env.sh         # 15-20 min; only needed on true first boot
```

## 3. First run: breast-cancer, smoke mode

Open **`STG-Driver-breast-cancer2.ipynb`** — it's the cleanest validation
case because it has the textbook collinear triple
(`mean_radius` / `mean_perimeter` / `mean_area`).

1. Kernel: **Python (tablegen)**
2. §1 cell — **set `FRESH_START = True`** (wipes stale `section_3.2`/`section_5.2`
   checkpoints from before the collinearity port; otherwise the restore step
   and `restoration_health.csv` emit get silently skipped)
3. §2.1 — confirm the new keys are present:
   ```python
   'collinearity_enabled':   True,
   'collinearity_threshold': 0.975,
   'collinearity_overrides': {},
   ```
   Also consider setting `tuning_mode: "smoke"` for a ~10-min first pass.
4. **Run All** and stop at §2.2b output — review the decisions table. Expected:
   - `mean_perimeter` / `mean_radius` → `residual_multiplicative`
   - `mean_area` / `mean_perimeter` → `residual_multiplicative`
   - `mean_area` / `mean_radius` → `skip (transitive)`
5. Check the §2.3 heatmap — `mean_radius`, `mean_perimeter`, `mean_area`
   tick labels should render **red bold**.
6. Let §3.1 / §3.2 run. Confirm
   `results/breast-cancer-data/<date>/3/restoration_health.csv` gets written
   with `delta < ~0.01` on a reasonably-trained model (CTGAN-default is the
   easy one to eyeball).
7. §4 pilot → §5.1 / §5.2 — confirm the radar chart in
   `results/.../5/sdac_radar_chart.png` now has an additional fidelity
   signal flowing through `Fidelity_Assoc_Preservation`. Per-pair
   diagnostic in `5/restoration_health.csv`.

## 4. If something errors

Most likely culprits and quick diagnostics:

| Symptom | Check |
|---------|-------|
| `ImportError: cannot import name '_cramers_v'` | The shim loaded out of order. Confirm `setup.py` line ~14 imports `src._compat_dython` **before** `from src import *`. |
| `KeyError: 'collinearity_enabled'` in §2.2b | You're on old NOTEBOOK_CONFIG. Verify cell 5 has the new keys after the pull. |
| `restoration_health.csv` empty or missing | Either `COLLIN_CTX.ops` was empty (no pair cleared threshold — fine for some datasets) or checkpoints loaded. Re-run with `FRESH_START = True`. |
| `NameError: COLLIN_FLAGGED not defined` in §2.3 | §2.2b didn't execute. Run cell 9 (§2.2b fit) before cell 11 (§2.3 EDA). |
| Model crashes on a `__resid` column | Flag the pair via `collinearity_overrides: {('col_a', 'col_b'): {'skip': True}}` and re-run. |
| Any other unexpected failure | `FRESH_START = True`, restart kernel, run from §1. If still broken, capture the traceback + the `synthetic_data_*` keys in globals(). |

## 5. Once the breast-cancer smoke is green

1. Repeat on the other three drivers (Alzheimer, diabetes, liver-train) —
   §3/§4/§5 cells are byte-identical across drivers, so they should behave
   the same. Per-dataset variation is confined to §2.
2. Promote to `tuning_mode: "full"` on whichever datasets you want baselined.
3. Check the `Fidelity_Assoc_Preservation` column in `sdac_evaluation_summary.csv` —
   it should separate the 8 generators meaningfully (CTGAN/TVAE typically
   score higher than `independent_baseline` would have before the signal
   filter).
4. Delete this file (`rm do-this-next.md`) and commit.

## Reference

Background: [`docs/collinearity-reduction.md`](docs/collinearity-reduction.md)
— strategy table, override format, sync-with-sibling governance.

Full doc index: [`README.md#documentation`](README.md).
