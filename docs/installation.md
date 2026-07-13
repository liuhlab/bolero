# Installation

`bolero` uses [pixi](https://pixi.sh) to manage its full environment — Python,
PyTorch + CUDA, and all dependencies are pinned in the repo, so installation is a
single command. You do **not** need conda or a system Python first.

**Requirements:** Linux (`linux-64`) with an NVIDIA GPU (driver CUDA ≥ 12.0).

## 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Open a new shell (or `source ~/.bashrc`) so pixi is on your `PATH`, then check:

```bash
pixi --version
```

## 2. Clone bolero

```bash
git clone https://github.com/liuhlab/bolero.git
cd bolero
```

## 3. Install

From the repo root:

```bash
pixi install            # default runtime environment (full GPU stack)
```

This resolves and installs every pinned dependency — PyTorch + CUDA, Ray,
flash-attn, scvi-tools — plus `bolero` itself. Two companion packages are pulled
straight from **git** during this step, so you need network/git access:

- [`bolerodata`](https://github.com/liuhlab/bolerodata) — the dataset / model-zoo
  registry that maps short keys to on-disk artifacts in the lab data lake.
- [`SEACells`](https://github.com/liuhlab/SEACells_pixi) — a lightly-repinned fork
  used for metacell construction (CPU).

The project defines three environments; install whichever you need:

| Environment | Install | Contents |
| --- | --- | --- |
| `default` | `pixi install` | runtime: full GPU stack (torch + CUDA, Ray, flash-attn) |
| `dev` | `pixi install -e dev` | adds tests, linting, notebooks, JupyterLab |
| `docs` | `pixi install -e docs` | lightweight, **CUDA-free** — builds the docs site only |

## 4. Run bolero

There is no `activate` step. Prefix commands with `pixi run`:

```bash
pixi run python -c "import bolero"
```

...or open a shell with the environment ready:

```bash
pixi shell
python -c "import bolero"
```

### Use bolero in Jupyter

Register the environment as a Jupyter kernel named `bolero`:

```bash
pixi run install-kernel
```

Then start JupyterLab and select the **bolero** kernel:

```bash
pixi run jupyter lab
```

## 5. Verify

`bolero.print_environments()` reports the key versions and GPU status:

```bash
pixi run python -c "import bolero; bolero.print_environments()"
```

Expected output (versions and GPU will differ):

```
----- bolero environment -----
bolero          : 2026.7.10
python          : 3.11.15
platform        : Linux-5.15.0-177-generic-x86_64-with-glibc2.35
torch           : 2.4.1
torch CUDA      : 12.0
CUDA available  : True
  GPU 0         : NVIDIA H100 80GB HBM3 (79 GB)
flash-attn      : 2.6.3 (import OK)
ray             : 2.34.0
numpy           : 2.4.6
pandas          : 2.3.3
scvi-tools      : 1.4.2
transformers    : 5.13.0
```

If `CUDA available` is `True` and `flash-attn` reports `import OK`, you are ready to go.
