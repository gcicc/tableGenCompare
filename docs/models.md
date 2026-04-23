# Models

Ten single-table generators wrapped in this repo; all implement the same
interface (`src/models/base_model.py::SyntheticDataModel`):

```python
class SyntheticDataModel(ABC):
    def train(self, data: pd.DataFrame, **kwargs) -> dict: ...
    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame: ...
    def sanitize_output(self, synth, real, target_column, task_type) -> pd.DataFrame: ...
```

Each concrete wrapper lives under `src/models/implementations/` and is
registered in `src/models/registry.py`. `train_models_batch` (§3.1) and
`train_models_batch_with_best_params` (§5.1) iterate over the registered set
and inject the dataset's `categorical_columns` as `discrete_columns` where
each library expects it.

## Training pipeline (applies to every model)

```
raw data ──► preprocess (§2.2) ──► collinearity reduce (§2.2b) ──► reduced data
                                                                       │
                                                                       ▼
                                                                train/generate
                                                                       │
                                                                       ▼
            full-schema synth ◄── restore_dropped (§3.2/§5.2) ◄── synth reduced
```

The reducer is a no-op when `collinearity_enabled: False`. See
[collinearity-reduction.md](collinearity-reduction.md) for the design and
placement rationale.

---

## Registered models

| Model | Family | Library | Wrapper | Notes |
|-------|--------|---------|---------|-------|
| CTGAN | GAN | SDV | `ctgan_model.py` | Canonical tabular GAN; conditional vector for discrete columns |
| TVAE | VAE | SDV | `tvae_model.py` | Variational autoencoder; fast training |
| CopulaGAN | Statistical + GAN | SDV | `copulagan_model.py` | Gaussian copula preprocessing + GAN |
| CTAB-GAN | GAN | CTAB-GAN submodule (in `CTAB-GAN/`) | `ctabgan_model.py` | Enhanced preprocessing (mode-specific normalization) |
| CTAB-GAN+ | GAN | CTAB-GAN-Plus submodule | `ctabganplus_model.py` | WGAN-GP + long-tail-aware losses |
| PATE-GAN | DP-GAN | ydata-synthetic / custom | `pategan_model.py` | Differentially private; fewer HPO knobs |
| MEDGAN | GAN | custom | `medgan_model.py` | Autoencoder + GAN; designed for medical records |
| GANerAid | GAN | GANerAid submodule | `ganeraid_model.py` | Clinical-data-specialized GAN; patched for device awareness |
| TabDiffusion | Diffusion | HuggingFace `diffusers` | `tabdiffusion_model.py` | DDPM for tabular data; custom denoising network with sinusoidal time embeddings |
| GReaT | LLM-based | [`kathrinse/be_great`](https://github.com/kathrinse/be_great) | `great_model.py` | Fine-tunes GPT-2 / distilgpt2 on serialized rows |

All ten are enabled by default in every driver's
`NOTEBOOK_CONFIG["models_to_run"]` list. To run a subset, edit that list.

---

## Planned additions (not yet implemented)

Modern tabular generators that belong in this comparison but haven't been
wrapped yet:

| Model | Family | Upstream | Why it's wanted |
|-------|--------|----------|-----------------|
| **TabDDPM (synthcity)** | Diffusion | [`rotot0/tab-ddpm`](https://github.com/rotot0/tab-ddpm) | The reference TabDDPM implementation. Not wired up because `synthcity` pins `torch<2.3`, which conflicts with this env. Our diffusion coverage currently comes from the custom **TabDiffusion** wrapper (HF `diffusers`). |
| **TabDiff** | Diffusion | Follow-up diffusion architectures (various) | Alternative diffusion parameterization; useful to separate "diffusion helps" from "this specific diffusion model helps." |

Implementation checklist for each: new `src/models/implementations/<name>_model.py`
subclassing `SyntheticDataModel`; registration in `src/models/registry.py`; HPO
search space in `src/models/search_spaces.py`; add to each driver's
`models_to_run` default; smoke-test on breast-cancer. See [decisions.md](decisions.md)
§5 for why the current 10 are registered by default.

---

## Hyperparameter optimization (§4)

HPO is Optuna-based and **staged**: smoke → pilot → full. See
[decisions.md](decisions.md) for the rationale.

- **Smoke mode** (`tuning_mode: "smoke"`): 10 trials per model, short budgets;
  surfaces time estimates and flags any model that errors immediately.
- **Full mode** (`tuning_mode: "full"`): continuation after pilot review,
  with diminishing-returns gates.

Each model exposes an `optuna_search_space()` method consumed by
`StagedOptimizationManager` in `src/models/staged_optimization.py`. Search
spaces per model live in `src/models/search_spaces.py`.

HPO trains on the **reduced** schema (post-collinearity). The residual
column is part of the training signal, so HPO tunes against what the model
will actually be asked to learn in §5.

---

## Submodule-backed models

CTAB-GAN, CTAB-GAN+, and GANerAid are git submodules under the repo root.
Initialize once on first clone:

```bash
git submodule update --init --recursive
```

`setup_env.sh` runs this automatically on SageMaker first-boot. GANerAid has
known device-awareness bugs that `patches/` addresses — they get applied by
`setup_env.sh` and should not need manual intervention.

---

## Adding a new model

1. Create `src/models/implementations/<new>_model.py` that inherits from
   `SyntheticDataModel`. Implement `train`, `generate`, `sanitize_output`.
2. Register in `src/models/registry.py`.
3. Add an HPO search space in `src/models/search_spaces.py`.
4. Add the model's name to the default `models_to_run` list in each driver's
   `NOTEBOOK_CONFIG` (all four drivers, per the uniformity rule — §2.1 is
   the *only* place per-dataset variation is allowed).
5. Smoke-test against breast-cancer before running the full suite.

---

## Post-generation sanitization

Every model's `generate()` output passes through `sanitize_output()`, which
enforces:
- Column order matches the training frame.
- Target column respects the declared schema (`enforce_target_schema` in
  `src/data/target_integrity.py` — clips to the real domain, casts to int
  for binary classification, etc.).
- NaN/inf rows get dropped or repaired before the reducer sees them.

This is the sanctioned place for post-gen integrity enforcement. Collinearity
restoration happens **after** sanitize (in the notebook §3.2 / §5.2 restore
loops) so any synth-schema repair is done before restoration runs its
`partner_col + residual` lookups.
