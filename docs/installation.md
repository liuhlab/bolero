# Installation

This guide walks through a complete, from-scratch install of `bolero` on a Linux
machine with an NVIDIA GPU (tested on an H100 node with driver CUDA 12.6).

## What is pixi?

[pixi](https://pixi.sh) is a fast, cross-platform package manager built on the
conda ecosystem. `bolero` uses it as the single source of truth for its
environment: conda packages (Python, PyTorch + CUDA, bioinformatics tools) and
PyPI packages (`bolero` itself and its Python dependencies) are all declared in
`pyproject.toml` and pinned in `pixi.lock`.

You do **not** need conda, mamba, or a base Python installed first — pixi brings
its own Python interpreter into the environment.

## 1. Install pixi

pixi is a single static binary. Install it with the official script:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

This installs to `~/.pixi/bin`. Open a new shell (or `source ~/.bashrc`) so it is
on your `PATH`, then confirm:

```bash
pixi --version        # e.g. pixi 0.72.0
```

## 2. Clone bolero

```bash
git clone https://github.com/liuhlab/bolero.git
cd bolero
```

## 3. Install bolero and its dependencies

From the repo root, let pixi build the environment from `pyproject.toml` /
`pixi.lock`. This creates `.pixi/envs/default/`, downloads the pinned conda and
PyPI packages (including a CUDA-enabled PyTorch build), and installs `bolero`
itself in editable mode:

```bash
pixi install            # runtime environment
```

For development (tests, docs, linting, notebooks) install the richer `dev`
environment instead:

```bash
pixi install -e dev
```

### Run commands in the environment

There is no `activate` step. Either prefix a command with `pixi run`:

```bash
pixi run python -c "import bolero; print(bolero.__version__)"
```

...or drop into an interactive shell with the environment on `PATH`:

```bash
pixi shell              # or: pixi shell -e dev
python -c "import bolero"
```

### What `pixi install` pulls in

The GPU stack — including `ray` (pinned to `2.34`), `flash-attn`, `triton`, and
the CUDA runtime + headers — is fully declared in `pyproject.toml`, so a single
`pixi install` sets everything up. There is **no** manual `pip install` step for
`ray` or `flash-attn`.

Two details are worth knowing if you ever bump versions:

- **`ray`** is declared under `[tool.pixi.pypi-dependencies]` (not
  `[project.dependencies]`) so the `2.34` pin does not leak into bolero's
  published wheel metadata and break skypilot installs.
- **`flash-attn`** comes from **conda-forge**, not a pip wheel URL, so its C++
  ABI matches conda-forge PyTorch automatically. It also pulls `triton` (used by
  flash-attn's rotary path) and `cuda-cudart-dev` (triton JIT-compiles a shim
  against `cuda.h` at first launch).

### CUDA / driver compatibility

`pyproject.toml` pins `cuda-version = "12.6"`. This is the CUDA **build target**,
**not** a "you must have exactly CUDA 12.6" requirement:

- **Installable** on any NVIDIA driver reporting **CUDA ≥ 12.0** (`nvidia-smi`) —
  every GPU package in the lock declares only `__cuda >= 12`.
- **Fully supported** on driver **≥ 12.6** (12.6, 12.7, 12.8, 12.9, 13.x); newer
  drivers always run an older CUDA build.
- On driver **12.0–12.5** it relies on CUDA 12 minor-version forward-compatibility
  (generally works; the less-tested direction).

Why 12.6 specifically: conda-forge only builds `triton` for `cuda126` and
`cuda129`+ (there is no `cuda120`–`cuda125` build), so 12.6 is the **lowest**
target available — and a lower target loads on **more** (older) drivers. Pinning
here is therefore the *most* driver-permissive choice; leaving it unpinned would
let the solver lock `cuda129` builds that need a newer driver in practice.

To retarget a different CUDA (e.g. you specifically want a newer toolchain), edit
the `cuda-version` pin in `pyproject.toml` and re-run `pixi install` to re-solve.

## 4. Verify the install

`bolero` ships a top-level `print_environments()` helper that reports the key
versions and GPU/CUDA status in one call:

```bash
pixi run python -c "import bolero; bolero.print_environments()"
```

Example output on an H100 node (versions may differ):

```
----- bolero environment -----
bolero          : 0.0.36.dev...
python          : 3.11.15
platform        : Linux-5.15.0-...-x86_64-with-glibc2.35
torch           : 2.4.1
torch CUDA      : 12.0
CUDA available  : True
  GPU 0         : NVIDIA H100 80GB HBM3 (79 GB)
flash-attn      : 2.6.3 (import OK)
ray             : 2.34.0
numpy           : ...
pandas          : ...
scvi-tools      : ...
transformers    : ...
```

To additionally confirm `flash-attn` is not just importable but actually runs a
forward pass on the GPU, run its test (needs the `dev` environment for pytest):

```bash
pixi install -e dev
pixi run -e dev pytest tests/test_flash_attn.py -v
```

## Requirements & notes

- **GPU / CUDA** — any NVIDIA driver reporting **CUDA ≥ 12.0** works (see
  [CUDA / driver compatibility](#cuda--driver-compatibility) above). The CUDA
  runtime, headers, and a CUDA-enabled PyTorch build all come from conda-forge
  via pixi, so you do **not** need a system CUDA toolkit installed. `flash-attn`
  and `triton` only run on a CUDA GPU.
- **Platform** — the environment is currently locked to `linux-64` only.
- **Docs environment** — building the docs does not require a GPU:
  `pixi install -e docs` provides a lightweight, CUDA-free environment with just
  the mkdocs tooling (`pixi run -e docs docs-serve`).
