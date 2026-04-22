# Workflow — first-run guide

Getting from a fresh clone to a completed smoke run on breast-cancer.

For the full project README (AWS SageMaker setup, architecture details), see
[../README.md](../README.md). This file is the shortest happy path.

## 0. Prerequisites

- Windows 10/11, Linux, or macOS
- Python 3.10 (SageMaker bundled env) or 3.12+ (local dev — see
  [decisions.md §4](decisions.md))
- ~5 GB free disk (conda env + torch + dataset CSVs)
- NVIDIA GPU optional; CPU runs fine for breast-cancer smoke mode

Submodules — CTAB-GAN, CTAB-GAN-Plus, GANerAid — initialize on clone:

```bash
git clone --recurse-submodules https://github.com/gcicc/tableGenCompare.git
cd tableGenCompare
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## 1. Environment

### On SageMaker (recommended for GPU)

```bash
cd ~/SageMaker/tableGenCompare
bash setup_env.sh
```

This creates the conda env at `/home/ec2-user/SageMaker/.envs/tablegen`
(persistent across instance stop/start), installs deps, patches GANerAid,
initializes submodules, and registers the Jupyter kernel. One-time; future
boots are handled by `on-start.sh`.

### Locally

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
# Optional: GPU torch
.venv/Scripts/python.exe -m pip uninstall -y torch
.venv/Scripts/python.exe -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 torch
```

Register the venv as a Jupyter kernel:

```bash
.venv/Scripts/python.exe -m ipykernel install --user \
    --name tablegencompare --display-name "Python (tablegencompare)"
```

## 2. Verify imports

```bash
.venv/Scripts/python.exe -c "
from setup import *
print('pandas:', pd.__version__)
print('torch:', torch.__version__ if 'torch' in dir() else '(not loaded)')
print('sdv:', __import__('sdv').__version__)
"
```

Expect the `[OK] Essential data science libraries imported successfully!`
banner and no errors. CTAB-GAN warnings about "not available: No module
named 'model'" are expected outside SageMaker — the submodule only resolves
from the `setup_env.sh` kernel.

## 3. Open a driver notebook

Start with breast-cancer (smallest, ~1 min per model in smoke mode):

- `STG-Driver-breast-cancer2.ipynb`

Select the `tablegen` (SageMaker) or `tablegencompare` (local) kernel.

## 4. Step through §1–§5

- **§1 Setup** — imports everything from `setup.py`. If this cell errors,
  the env is wrong; don't continue.
- **§2.1 NOTEBOOK_CONFIG** — review / edit. `tuning_mode: "smoke"` for a
  fast first run. `collinearity_enabled: True` by default.
- **§2.2 Load and preprocess** — produces `data`, `original_data`,
  `categorical_columns`, `metadata`.
- **§2.2b Collinearity reduction** — fits the reducer, displays the
  decisions table. On breast-cancer the
  `mean_radius`/`mean_perimeter`/`mean_area` triple gets treated. See
  [collinearity-reduction.md](collinearity-reduction.md).
- **§2.3 EDA** — association heatmap with flagged columns in red,
  feature distributions, column analysis.
- **§3.1/§3.2** — train all 8 generators with default params; evaluate on
  the full real schema (collinearity-restored synth). Check
  `results/{dataset}/{date}/3/restoration_health.csv` — delta should be
  small on residualized pairs for models that trained successfully.
- **§4** — staged HPO. Pilot runs automatically. Smoke mode stops after
  pilot; full mode continues (§4.3, §4.5).
- **§5.1/§5.2** — retrain with best params; final scorecard with radar
  chart and `sdac_evaluation_summary.csv`.

## 5. Know when you're done

A successful first run on breast-cancer produces (in
`results/breast-cancer-data/{date}/`):

- `Section-2/mixed_association_heatmap.png` (with red tick labels for
  flagged columns)
- `Section-2/collinearity_decisions.csv` (implicit — shown in §2.2b output)
- `3/sdac_evaluation_summary.csv` — one row per model
- `3/restoration_health.csv` — one row per (model, treated pair)
- `Section-4/*.html` — Optuna optimization history, parameter importance,
  parallel coordinates per model
- `5/sdac_evaluation_summary.csv`, `5/sdac_radar_chart.png`,
  `5/sdac_heatmap.png` — final scorecard
- `5/restoration_health.csv` — final-model residual-preservation diagnostic

## Common gotchas

- **Wrong kernel.** JupyterLab often defaults to a different Python. Check
  with `!python -c "import sys; print(sys.executable)"` at the top of a
  notebook — it should point inside your env.
- **`from setup import *` fails.** You're on the wrong kernel or missing
  submodules. Run `git submodule update --init --recursive`, re-run
  `setup_env.sh` (SageMaker) or `pip install -r requirements.txt` (local).
- **GANerAid device errors.** `setup_env.sh` patches these; if you built
  the env manually, copy the files in `patches/` into the GANerAid
  site-packages directory (`setup_env.sh` has the paths).
- **Section 5 is empty / errors.** Usually §4 didn't finish. Check
  `RUN_MODE` and re-run §4 end-to-end.
- **Checkpoint resumption shows old data.** Set `FRESH_START = True` in §1
  and re-run.

## Next steps after a successful first run

1. Move to another dataset: open `STG-Driver-Alzheimer2.ipynb` (mixed-type
   schema), `STG-Driver-diabetes2.ipynb` (small + balanced), or
   `STG-Driver-liver-train2.ipynb` (30k rows).
2. Full-mode HPO: set `tuning_mode: "full"` in §2.1 and budget 4–6 hours.
3. Review the output files: [evaluation.md](evaluation.md) documents every
   metric key and how composites are built.
