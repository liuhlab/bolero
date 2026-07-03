# Installation

Install [pixi](https://pixi.sh), then:

```bash
git clone https://github.com/liuhlab/bolero.git
cd bolero
pixi install            # runtime env; or `pixi install -e dev` for tests/docs/lint/notebooks
pixi shell              # activate the environment (or: pixi shell -e dev)
pixi run python -c "import bolero"
```

## GPU / optional heavy dependencies

Requires an NVIDIA driver supporting CUDA 12. `ray` and `flash-attn` are **not** managed by
pixi (skypilot / pinned-ABI wheel constraints) and are installed manually into the environment:

```bash
pixi run pip install "ray==2.34"
pixi run pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```
