# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Bolero is

Bolero is the code base for the paper *"Bolero: predicting cell-state-specific gene
regulation from DNA sequence."* It is a **cell-state-conditioned sequence-to-function
model**: it takes a 524,288 bp one-hot DNA sequence **plus** an atlas-scale cell-state
embedding (and optional conditioning variables such as tissue, developmental age, or an
AP-1 activity score) and predicts cell-state-specific chromatin accessibility and
transcript abundance at 32 bp resolution (16,384 output bins). It is built on top of the
Borzoi/Flashzoi genomics transformer and trained on Bolero-10M (10.8M cells, 36 datasets,
6 mammals).

The manuscript lives in `paper/Draft.md` (with figure PNGs) — read it for the scientific
framing. Note `paper/` is git-ignored (local reference only).

## Companion package: bolerodata

This project spans **two repos**. `bolero` (this repo) is the **main** package: the model,
training, inference, interpretation code and the pixi environment (PyTorch/CUDA, Ray,
flash-attn). Its companion, [`bolerodata`](https://github.com/liuhlab/bolerodata) (local
clone usually at `../bolerodata`), is a **lightweight registry** for the *data behind the
model*: the single-cell datasets (Bolero-10M), the trained model zoo, QTL collections and
differential-accessibility records — mapping short keys to on-disk artifacts under the lab's
`$STANDARD_DIR` data lake.

- `bolerodata` is listed as a **git-based pixi dependency** in this repo's
  `[tool.pixi.pypi-dependencies]`, so `pixi install` brings it in. It carries no GPU stack
  and is meant to run **inside this environment**. (It is not a PyPI/`[project.dependencies]`
  dependency, so a plain `pip install bolero` from PyPI does not pull it.)
- The dependency is **one-directional at the environment level but cyclic at the API level**:
  `bolerodata` imports `bolero` (mostly lazily) to build `Genome` objects, predictors
  (`bolero.tl.predict`), and IGV browsers (`bolero.pl.igv`). Keep `bolero`'s imports of
  `bolerodata` (if any) lazy to avoid an import cycle.
- **Scope split:** model / sequence / training / inference / attribution machinery → here in
  `bolero`. "Which datasets / models / QTLs / DA records exist and where their files are" →
  `bolerodata`. See `bolerodata/CLAUDE.md` for its internals.

## Important: the manuscript is the source of truth for what's "real"

This is a research code base that grew during the project, so it contains **vestigial and
experimental code** — models, heads, presets, tasks, and helper functions that were tried
but are **not** part of the final published results. Also note that **much of the actual
result-generation code lives outside this repository**; this repo is the reusable package
being finalized to ship with the paper. Do **not** assume every class/function here is
load-bearing.

Treat **`paper/Draft.md` (especially the Methods) as the authoritative spec** for what the
final results actually used. Finalization proceeds **piece by piece, driven by the author** —
do **not** proactively audit, flag, or prune "vestigial" code; work on the specific piece
under discussion, and when it's unclear whether something is final vs. vestigial, ask rather
than guess.

## Environment & common commands

The project uses **[pixi](https://pixi.sh)** as the single source of truth for its
environment (conda + PyPI deps, all pinned in `pyproject.toml` / `pixi.lock`). There is no
`conda`/`venv`/`pip install` step and no `activate`.

Three pixi environments: `default` (runtime, GPU stack), `dev` (adds test/lint/notebook/docs
tooling), `docs` (lightweight, CUDA-free, mkdocs only).

```bash
pixi install                 # runtime env (full GPU stack: torch+CUDA, ray, flash-attn, triton)
pixi install -e dev          # dev env (tests, lint, notebooks)

pixi run <cmd>               # run a command inside the default env (no activate step)
pixi shell -e dev            # or drop into an interactive shell

# Verify the install (versions + GPU/CUDA/flash-attn status)
pixi run python -c "import bolero; bolero.print_environments()"

# Tests (there is a `test` task in the dev env)
pixi run -e dev test                                   # = coverage run -m pytest -v --color=yes
pixi run -e dev pytest tests/test_flash_attn.py -v     # a single test file
pixi run -e dev pytest tests/test_flash_attn.py::test_name -v   # a single test

# Lint / format (ruff via pre-commit; `lint` task in the dev env)
pixi run -e dev lint         # = pre-commit run --all-files

# Docs (docs env; served/built/deployed via mkdocs)
pixi run -e docs docs-serve  # live preview
pixi run -e docs docs-build  # mkdocs build --strict
pixi run -e docs docs-deploy # build + push to gh-pages
```

Notes:
- The GPU stack (`flash-attn`, `triton`, CUDA) needs an NVIDIA GPU with driver CUDA ≥ 12.0.
  `cuda-version` is pinned to `12.6` as a *build target* (most driver-permissive), not a
  requirement — see `docs/installation.md`.
- The test suite is currently just `tests/test_flash_attn.py` (a GPU smoke test that runs a
  flash-attention forward pass). CI (`.github/workflows/test.yaml`) runs it on a GPU-less
  runner with `CONDA_OVERRIDE_CUDA=12.1`, so imports must not require a live GPU at import time.
- Versioning is `hatch-vcs` from git tags (`v*`); pushing a `v*` tag publishes to PyPI
  (`.github/workflows/release.yaml`).

## Runtime setup inside code

- `bolero.init(...)` initializes Ray (object store, spilling, CPU count) — call it before
  using the Ray data/inference pipelines.
- Model checkpoints, prediction outputs, and large data are git-ignored by pattern
  (`**/*.pt`, `**/*.ckpt`, `**/*.npy`, `**/*.joblib`, `**/*.json`, `data/`, `**/wandb/`).
  The only committed binary assets are under `src/bolero/pkg_data/` (blacklists, Borzoi
  sequence/target tables, JASPAR motif PWM dicts).

## Package layout & import conventions

Follows the scverse convention under `src/bolero/`: **`pp`** (preprocessing), **`tl`**
(tools), **`pl`** (plotting).

**Import by fully-qualified module path.** Only the top-level package and `pp` curate
re-exports; `tl/__init__.py`, `pl/__init__.py`, and most `tl/*/__init__.py` (including
`tl/model/borzoi/`) are empty. So use e.g. `from bolero.tl.model.borzoi.model_lora import
BorzoiLoRA`, not a package-level shortcut. What *is* re-exported at `bolero.*`:
`Genome`, `Sequence` (from `pp`); `hg38_splits`, `mm10_splits` (5-fold CV chromosome
splits, from `tl.generic.train_helper`); `init`, `print_environments`.

## Core architecture (the big picture)

The one thread that ties the whole codebase together:

> **A frozen Borzoi backbone + per-layer *conditional* LoRA adapters whose low-rank weights
> are generated on-the-fly by MLPs from a cell-state embedding. This can be "collapsed" into
> a plain, cell-state-specific DNA→track model for fast inference and attribution.**

### 1. Model — `tl/model/borzoi/`
- `model.py` — `Borzoi(nn.Module)` reimplements the Flashzoi backbone (conv DNA stem →
  residual conv tower → U-net → 8 transformer layers w/ flash-attention → separable convs →
  final joined convs). Constants: `BORZOI_INPUT_LEN=524288`, `BORZOI_OUTPUT_LEN=16384`.
  `BorzoiWithOutputHead` adds the *original* bulk human/mouse tracks (baseline only).
- `model_lora.py` — **`BorzoiLoRA(Borzoi)` is the main model class** ("Bolero"). It loads
  the frozen base checkpoint, adds an output head + optional signal model, and
  `convert_to_lora()` swaps Linear/Conv layers for conditional-LoRA variants. `collapse_lora(embedding)`
  bakes a specific cell embedding into a plain model. Variants: `BorzoiLoRAMulti`
  (multi-dataset), `BorzoiLoRAwithArches`.
- `module_output.py` — output heads: **count** (accessibility, softplus + Poisson-multinomial
  loss), **delta/velocity** (paired reference→target difference, MSE — used for the
  differential-accessibility evaluations), **scooby** (hypernetwork head, baseline),
  **gene_count** (Decima-style RNA head for eQTL), **eqtl**, **dual_atac_mc**.
- `model_lora_config.py` — `LORA_CONFIG_FUNCTIONS`; default preset `all_conditional`
  assigns per-module LoRA ranks and holds attention q/k/v projections non-conditional.
  `all_conditional_scaling` is used for the parameter-scaling ablation.
- `model_flow.py` — flow-matching / ODE wrappers over a signal model (track generation).
- `train.py` — `BorzoiLoRATrainer` (and `Multi*`, `Arch*`) — top-level `.train()` drives
  wandb setup → fit loop → test.

### 2. Conditioning engine — `tl/generic/` (model-agnostic, consumed by Borzoi)
- `module_lora_cond.py` — `ConditionalLoRALinear` / `ConditionalLoRAConv`: the adapter
  weight `A·B` is *produced by `EmbeddingMLP`s from the embedding* (not learned directly),
  so each batch item gets its own effective weight; `.collapse(embedding)` bakes it in.
  `convert_to_conditional_lora_model()` / `collapse_lora_model_()` walk the module tree.
- `module_lora.py` — plain (non-conditional) LoRA primitives for layers configured
  `default_conditional=False`.
- `module_embedding.py` — `EmbeddingMLP` (B-side zero-initialized so LoRA starts as
  identity); `ConditionEmbeddingModule` turns a cell embedding + condition metadata into the
  vector fed to every LoRA MLP.
- `train.py` — `GenericTrainer` base (config merge/validate across dataset+model+trainer,
  wandb resume, checkpointing, multi-GPU DataParallel, EMA). `train_helper.py` holds the
  chromosome CV splits, online metric accumulators, and `make_borzoi_scheduler` (warmup +
  polynomial decay). `ema.py` is an EMA shadow-model.

### 3. Data pipeline — `pp/` and `tl/dataset/`
Data flow: single-cell fragments are pre-aggregated into a **parquet_db** (directory of
parquet files; each row is a meta-region holding a gzip-compressed CSR `metacell × base`
coverage matrix). Training/inference assembles conditioned batches by: read parquet →
decompress bytes to CSR → merge cell rows into **pseudobulks** (with their embeddings) →
slice the model window → fetch DNA one-hot → reverse-complement augment → collate.
- `pp/genome.py` — `Genome` (downloads UCSC fasta/chrom.sizes, global coords, blacklist,
  whole-genome one-hot in zarr, region→one-hot, motif DB). `pp/seq.py` — `Sequence` + one-hot
  helpers. `pp/ray_chunk_dataset.py` / `genome_chunk_dataset.py` / `gene_dataset.py` — the
  *builders* that produce a parquet_db.
- `tl/dataset/ray_dataset.py` — `RayGenomeChunkDataset` (streaming Ray training pipeline;
  `BorzoiDataset` in `model/borzoi/dataset.py` subclasses it), `RayRegionDataset` (inference).
- `tl/dataset/parquet_db.py` — `GenomeParquetDB`: DuckDB + Ray random-access query used by
  the inference datamanager. `transforms.py` / `sc_transforms.py` / `file_transforms.py` are
  the Ray map ops. **Ray is the parallel data engine** throughout.
- `tl/pseudobulk/` — how cells become conditioned units: SEACells metacells (`seacell.py`),
  paired pseudobulkers for delta/velocity training (`paired_pseudobulk.py`), KNN/geosketch
  pseudobulk construction (`_knn.py`).

### 4. Inference — `tl/predict/`
- `predictor_borzoi.py` — **`BorzoiPredictor`** (and `BorzoiSignalPredictor` for the
  signal/flow model) is the main inference class. It owns a model + a
  `GenericGenomeDataManager` and exposes high-level *task* methods mapping to the paper:
  `prediction_task`/`inference_task`, `caqtl_task`/`qtl_task`, `eqtl_task`, `peak_task`,
  `attribution_task`, `evolution_task`. `BorzoiInputXGradient` wraps the **collapsed**
  cell-specific model with `captum.InputXGradient` for per-base attribution.
- `datamanager.py` — `GenericGenomeDataManager` assembles inference batches from the
  parquet_db / reference bigwigs + pseudobulk embeddings.
- `dna_gen.py` — the **directed-evolution / enhancer-design** engine (paper's synthetic
  enhancers): `DNASynthesisFactory` (build/mutate sequences), `DNAEvolutionFactory`
  (beam-search random mutagenesis with selection toward a target cell-type × state pattern).
- `task_manager.py` — `caQTLManager` / `eQTLManager` / `PeakManager` (ref/alt substitution,
  peak/gene aggregation). `callbacks.py`, `track.py`, `task_aggregate.py` handle pre/post
  steps, bigwig track output, and gathering per-batch results.

### 5. Interpretation — `tl/motif/`, `tl/chromvar/`, `tl/footprint/`, `tl/structure/`
Consume the attribution scores from §4:
- `tl/motif/` — `modisco.py` (TF-MoDISco seqlet discovery, `ModiscoResults`), `finemo.py`
  (finemo-gpu genome-wide hit calling), `seqlet_tomtom.py` (seqlet→PWM via TomTom, cross-
  pseudobulk aggregation), `scan.py`/`jaspar.py` (MOODS / JASPAR scanning).
- `tl/chromvar/chromvar.py` — GPU chromVAR (`compute_deviations`), source of the **AP-1
  activity score** used as a conditioning variable in the score model (paper's "Bolero-Score").
- `tl/footprint/` — TF footprinting from ATAC signal. `tl/structure/` — AlphaFold3 (`af3.py`),
  ESM-C (`esm.py`), AlphaFold DB (`afdb.py`) for the TF-cooperativity structural analysis.

## Paper ↔ code map (quick reference)

- **"Bolero"** = `BorzoiLoRA` with the `all_conditional` LoRA preset.
- **"Bolero-Score" (AP-1)** = the same model fine-tuned with a chromVAR AP-1 score
  (`tl/chromvar/`) as an extra conditioning variable.
- **Velocity / paired-difference (DAR) predictions** = the `delta` output head + paired
  pseudobulkers.
- **Variant-effect (caQTL/eQTL)** = `caQTLManager`/`eQTLManager` + `caqtl_task`/`eqtl_task`.
- **Directed enhancer evolution** = `DNAEvolutionFactory` + `evolution_task`.
- **Cross-species prediction** = DNA-only mode of the count head, run per-species via `Genome`.
- **Borzoi fold-0 train/valid/test split** = `hg38_splits` / `mm10_splits`.

## Conventions

- Formatting/linting: **ruff** (line length 88, numpy-docstring convention) via
  `.pre-commit-config.yaml`; `pyproject.toml [tool.ruff]` lists the enabled rules and
  per-file ignores. New Python should carry numpy-style docstrings on public functions/classes.
- Config pattern: many models/trainers/datasets use a `default_config` dict +
  `create_from_config` + `bolero.utils.validate_config` (a `"REQUIRED"` sentinel marks
  mandatory keys).
