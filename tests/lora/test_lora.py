# %%
from bolero.tl.generic.module_lora_cond import EmbeddingMLP
import torch
# %%
mlp = EmbeddingMLP(10, 100, (5, 20), 120, 1, 1)


# %%
x = torch.rand(64, 3, 10)
emb_weights = None
# %%
mlp(x, emb_weights).shape
# %%
