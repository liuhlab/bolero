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

## What you can do with it

The documentation is a runnable, ordered tutorial series that walks the full workflow —
from raw single cells to a trained model and its predictions:

- **Cell embedding & metacells** — build a joint cell-state embedding and SEACells metacells.
- **Datasets** — aggregate single-cell fragments into the parquet coverage database, then form pseudobulks with reference signal.
- **Training** — fit Bolero on a single dataset, or the full multi-dataset ATAC (+ gene) atlas.
- **Prediction & variant effect** — predict accessibility, score caQTLs / eQTLs, and compute per-base DNA attributions.
- **Bolero-Score** — condition the model on a chromVAR TF-activity score (e.g. AP-1).
- **Cross-species** — run the trained atlas model on any species' genome, DNA-only, no retraining.

## Installation

`bolero` uses [pixi](https://pixi.sh), which sets up the full GPU stack (PyTorch + CUDA,
`ray`, `flash-attn`) in one step:

```bash
git clone https://github.com/liuhlab/bolero.git
cd bolero
pixi install            # runtime env; or: pixi install -e dev
```

`pixi install` also pulls the two companion git dependencies automatically. See
[docs/installation.md](docs/installation.md) for requirements and verification.

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
