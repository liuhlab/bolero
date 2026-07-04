"""Confirm flash-attn is installed correctly and usable by bolero's model code.

flash-attn ships as a compiled extension pinned to a specific Python / PyTorch /
CUDA ABI (see ``[tool.pixi.pypi-dependencies]`` in ``pyproject.toml``). A wheel that
imports but was built for the wrong ABI will fail at CUDA call time, so these tests
exercise an actual forward pass on the GPU rather than just importing the module.
"""

import pytest
import torch

# flash-attn only runs on CUDA; skip the whole module on CPU-only machines.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="flash-attn requires a CUDA GPU",
)


def test_flash_attn_importable():
    """The compiled flash-attn extension and its MHA module import cleanly."""
    import flash_attn
    from flash_attn.modules.mha import MHA  # noqa: F401

    assert flash_attn.__version__


def test_bolero_flash_attention_forward():
    """bolero's FlashAttention block runs a real forward pass on the GPU.

    Uses the module's default dims (head_dim must stay >= the hard-coded
    rotary_emb_dim of 128, so the sizes are not shrunk).
    """
    from bolero.tl.model.borzoi.module import FlashAttention

    dim, heads, batch, seqlen = 1536, 8, 2, 16
    module = FlashAttention(dim=dim, heads=heads).cuda().half()

    x = torch.randn(batch, seqlen, dim, device="cuda", dtype=torch.float16)
    out = module(x)

    assert out.shape == (batch, seqlen, dim)
    assert torch.isfinite(out).all()
