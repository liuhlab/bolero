# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
# %%
import math
def get_positional_features_central_mask(positions, features, seq_len):
    pow_rate = math.exp(math.log(seq_len + 1) / features)
    center_widths = torch.pow(
        pow_rate, torch.arange(1, features + 1, device=positions.device)
    ).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    feature_functions = [
        get_positional_features_central_mask,
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size is not divisible by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings
# %%
embeddings = get_positional_embed(4096, 32, 'cpu')
# %%
embeddings.shape
# %%
plt.imshow(embeddings.numpy(), aspect='auto')

# %%
embeddings.max()
# %%
seq_len = 1536
positions = torch.arange(-seq_len + 1, seq_len)
# %%
emb = get_positional_features_central_mask(positions, 768, 1536)
# %%
plt.imshow(emb)
# %%
plt.imshow(embeddings.numpy(), cmap='coolwarm')
# %%

def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., : ((t2 + 1) // 2)]


# %%
rel_emb = relative_shift(torch.ones(2, 8, 4096, 8191))

# %%
rel_emb.shape
# %%
