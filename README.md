# bolero

**Bolero: predicting cell-state-specific gene regulation from DNA sequence.**

Bolero is a cell-state-conditioned sequence-to-function model. It takes a 524,288 bp
one-hot DNA sequence **plus** an atlas-scale cell-state embedding (and optional
conditioning such as tissue, developmental age, or a TF-activity score) and predicts
cell-state-specific chromatin accessibility and transcript abundance at 32 bp resolution
(16,384 output bins).

Under the hood it is a **frozen Borzoi/Flashzoi backbone with per-layer conditional LoRA
adapters** whose low-rank weights are generated on the fly from the cell-state embedding —
so every cell state gets its own effective network. For a chosen cell state those adapters
can be "collapsed" into a plain DNA→track model for fast inference and base-level
attribution. Bolero is trained on **Bolero-10M**: 10.8M cells across 36 datasets and 6
mammals.

📖 **Documentation & tutorials:** <https://liuhlab.github.io/bolero/>

## System requirements

- **Operating system:** Linux, x86-64 (`linux-64`). macOS and Windows are not supported.
- **Hardware:** one NVIDIA GPU with a driver that supports CUDA 12.0 or later. The model has no
  CPU-only mode. Installation itself needs no GPU.
- **Disk:** about 25 GB free for the quick-start downloads; full datasets need more.
- **Software:** [pixi](https://pixi.sh) installs everything else, pinned in `pyproject.toml`
  and `pixi.lock`. Main versions: Python 3.11, PyTorch 2.4 (CUDA build), CUDA 12.6 as the build
  target, flash-attn and triton (conda-forge), Ray 2.34. Git dependencies:
  [`bolerodata`](https://github.com/liuhlab/bolerodata), lab forks of SEACells and MOODS that
  install cleanly in this environment, and finemo-gpu.
- **Tested on:** Ubuntu 22.04.5 LTS, NVIDIA H100 80 GB, NVIDIA driver 560.35.05, pixi 0.72.2.

## Installation

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # install pixi, then open a new shell
git clone https://github.com/liuhlab/bolero.git
cd bolero
pixi install            # runtime env; or: pixi install -e dev
```

`pixi install` also pulls the git dependencies above. **Typical install time:** about 3 min for
the clone and `pixi install` with an empty package cache, measured on a Linux server with a
fast network. Allow longer on a slower connection. See
[docs/installation.md](docs/installation.md) for the environments, Jupyter setup and checks.

## Demo

The quick start predicts chromatin accessibility with the pretrained Bolero model for 8
held-out genomic windows (hg38) in 5 blood cell types from the ChromiumPBMC dataset, and
compares the predictions with the measured single-cell ATAC signal. Every input downloads
from the public HuggingFace repositories
[`arcinstitute/bolero-models`](https://huggingface.co/arcinstitute/bolero-models) and
[`arcinstitute/bolero-data`](https://huggingface.co/datasets/arcinstitute/bolero-data) on the
first run, about 12 GB in total.

```bash
pixi run quickstart     # executes docs/tutorials/00_quickstart.ipynb in place
```

**Expected output** in `docs/tutorials/quickstart_output/`:

- `pred_peak_data.feather`, `true_peak_data.feather`: predicted and measured accessibility of
  each peak in the predicted windows, one column per cell type.
- `summary_stats.joblib.gz`: Pearson r and R² between the two for each cell type. In the
  reference run, Pearson r was 0.88–0.93 across 598 peaks.
- `pred_vs_measured.png`: predicted vs measured accessibility per peak, and the predicted and
  measured 32 bp tracks of one window for each cell type.

The committed notebook
[`docs/tutorials/00_quickstart.ipynb`](docs/tutorials/00_quickstart.ipynb) shows these outputs
from the reference run.

**Expected run time:** about 10 min of downloads and one-time setup on the first run (cached
afterwards), then about 25 s of prediction on one NVIDIA H100 80 GB.

## Instructions for use

The [documentation](https://liuhlab.github.io/bolero/) is an ordered tutorial series that
runs the full workflow on your own data, from single cells to a trained model and its
predictions:

- **Cell embedding & metacells**: build a joint cell-state embedding and SEACells metacells.
- **Datasets**: aggregate single-cell fragments into the parquet coverage database, then form
  pseudobulks with reference signal.
- **Training**: fit Bolero on a single dataset, or on the full multi-dataset ATAC (+ gene)
  atlas.
- **Prediction & variant effect**: predict accessibility, score caQTLs and eQTLs, and compute
  per-base DNA attributions.
- **Bolero-Score**: condition the model on a chromVAR TF-activity score (for example AP-1).
- **Cross-species**: run the trained atlas model on any species' genome, DNA only, with no
  retraining.

## Companion package

`bolero` pairs with [`bolerodata`](https://github.com/liuhlab/bolerodata), a lightweight
registry that maps short keys to the datasets (Bolero-10M), trained model zoo, and QTL
collections behind the paper. It carries no GPU stack, runs inside this environment, and is
installed automatically by `pixi install`.

## Citation

If you use Bolero in your work, please cite the paper *"Bolero: predicting cell-state-specific
gene regulation from DNA sequence"* (Hanqing Liu et al.). Full citation details will be added
on publication.

## License

MIT — see [LICENSE](LICENSE).
