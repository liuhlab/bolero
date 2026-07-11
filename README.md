# bolero

**Bolero: predicting cell-state-specific gene regulation from DNA sequence.**

A cell-state-conditioned sequence-to-function model — a frozen Borzoi/Flashzoi backbone
with conditional LoRA adapters — that predicts cell-state-specific chromatin accessibility
and transcript abundance from DNA sequence.

Documentation: <https://liuhlab.github.io/bolero/>

## Installation

`bolero` uses [pixi](https://pixi.sh), which sets up the full GPU stack (PyTorch + CUDA,
`ray`, `flash-attn`) in one step:

```bash
git clone https://github.com/liuhlab/bolero.git
cd bolero
pixi install            # runtime env; or: pixi install -e dev
```

See [docs/installation.md](docs/installation.md) for details.

---

> README and documentation are a work in progress.
