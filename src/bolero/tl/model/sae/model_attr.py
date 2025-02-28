# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from tangermeme.seqlet import recursive_seqlets
# from vector_quantize_pytorch import VectorQuantize
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from bolero.tl.generic.module import DNA_CNN, DilatedCNN


# def attr_jaccard_loss(X, Y, max_shift=12):
#     """
#     X: true attribution scores, shape (batch, seq_len, 4)
#     Y: predicted attribution scores, shape (batch, seq_len, 4)

#     Computes a jaccard-like alignment score over a range of shifts.
#     It pads Y to allow shifting, computes a score for each shift, and then
#     uses 1 - best_score as the loss (so that a higher alignment yields a lower loss).
#     """
#     batch, seq_len, channels = X.size()
#     # Pad Y along the sequence dimension on both sides.
#     Y_padded = F.pad(Y, (0, 0, max_shift, max_shift))

#     scores = []
#     # Loop over possible shifts (assumed to be small)
#     for shift in range(max_shift):
#         # Slice the padded Y to get a shifted version with the same sequence length as X.
#         Y_shift = Y_padded[:, shift : shift + seq_len, :]

#         # Compute element-wise sign alignment
#         sign = torch.sign(X) * torch.sign(Y_shift)
#         # Compute absolute values
#         X_abs = X.abs()
#         Y_abs = Y_shift.abs()
#         # Compute element-wise minimum (times sign) and maximum values
#         min_val = torch.where(Y_abs > X_abs, X_abs, Y_abs) * sign
#         max_val = torch.max(X_abs, Y_abs)

#         # Sum over sequence and channel dimensions
#         min_sum = min_val.sum(dim=[1, 2])
#         max_sum = max_val.sum(dim=[1, 2])
#         # Compute the alignment score for this shift
#         score = min_sum / max_sum
#         scores.append(score.unsqueeze(1))

#     # Concatenate scores for each shift (batch, max_shift)
#     scores = torch.cat(scores, dim=1)
#     # Select the best (maximum) alignment score for each sample
#     best_scores, _ = scores.max(dim=1)
#     # Use 1 - best_score as the loss (so that a higher alignment yields a lower loss)
#     loss = 1.0 - best_scores
#     return loss


# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()

#         n_filters = 512
#         dna_kernel_size = 9

#         def _dilation_func(x):
#             return 2 ** (x // 2 + 1)

#         self.enc = nn.Sequential(
#             DNA_CNN(n_filters=n_filters, dna_kernel_size=dna_kernel_size),
#             DilatedCNN(
#                 n_filters=n_filters,
#                 bottleneck=n_filters,
#                 n_blocks=8,
#                 dia_kernel_size=3,
#                 groups=8,
#                 activation=nn.GELU(),
#                 batch_norm=True,
#                 batch_norm_momentum=0.1,
#                 dilation_func=_dilation_func,
#                 bipass_connect=False,
#             ),
#         )

#     def forward(self, x):
#         x = self.enc(x)
#         return x


# class Decoder(nn.Module):
#     def __init__(self, ks=3):
#         super().__init__()
#         modules = []
#         for n in range(5):
#             modules += [
#                 nn.ConvTranspose1d(
#                     4 * 2 ** (5 - n), 4 * 2 ** (4 - n), kernel_size=ks, padding=ks // 2
#                 ),
#                 nn.ReLU(),
#                 nn.Upsample(scale_factor=2, mode="nearest"),
#             ]
#         self.dec = nn.Sequential(*modules)
#         self.ks = ks

#     def forward(self, x):
#         x = self.dec(x)
#         return x


# class MotifVQ(nn.Module):
#     def __init__(self, n_latent=128, n_motifs=1024, decay=0.8, heads=1, motif_size=32):
#         super().__init__()
#         # self.enc = Encoder(ks=3)
#         self.enc = nn.Conv1d(4, n_latent, kernel_size=motif_size, stride=motif_size)
#         self.vq = VectorQuantize(
#             dim=n_latent,
#             codebook_size=n_motifs,
#             decay=decay,  # the exponential moving average decay, lower means the dictionary will change faster
#             commitment_weight=1.0,
#             heads=heads,
#             separate_codebook_per_head=False,
#         )
#         self.dec = nn.ConvTranspose1d(
#             n_latent, 4, kernel_size=motif_size, stride=motif_size
#         )
#         self.motif_size = motif_size

#     def forward(self, x):
#         # x_in: (bs, 4, l)
#         x_in = x

#         x = F.relu(self.enc(x_in))
#         x = rearrange(x, "b c l -> b l c")
#         x, indices, commit_loss = self.vq(x)
#         x = F.relu(x)
#         x = rearrange(x, "b l c -> b c l")
#         x_recon = self.dec(x)

#         recon_loss = self.get_reconstruction_loss(x_in, x_recon)
#         total_loss = recon_loss + commit_loss

#         data = {
#             "x_recon": x_recon,
#             "commit_loss": commit_loss,
#             "recon_loss": recon_loss,
#             "loss": total_loss,
#             "indices": indices,
#         }
#         return data

#     @torch.no_grad()
#     def motifs_from_code(self):
#         motifs = []
#         for i in range(self.vq.codebook.shape[0]):
#             with torch.no_grad():
#                 x = self.vq.codebook[i]
#                 x = x.unsqueeze(1).unsqueeze(0)
#                 motif = self.dec(F.relu(x))
#                 motifs.append(motif)
#         motifs = torch.concatenate(motifs)
#         return motifs

#     def get_reconstruction_loss(self, x, x_recon):
#         """Soft Dynamic Time Warping Loss"""

#         # Compute loss in forward direction
#         # loss = F.mse_loss(x, x_recon, reduce=False)
#         loss = attr_jaccard_loss(x, x_recon, max_shift=self.motif_size // 3)

#         # Compute soft-DTW loss in reverse complement direction
#         x_recon_rc = torch.flip(x_recon, dims=[1, 2])
#         # loss_rc = F.mse_loss(x, x_recon_rc, reduce=False)
#         loss_rc = attr_jaccard_loss(x, x_recon_rc, max_shift=self.motif_size // 3)

#         # Take the minimum of both losses
#         final_loss = torch.minimum(loss, loss_rc).mean()
#         return final_loss


# def call_seqlets(proj_attr, attr, seqlet_size=24, min_seqlet_len=4):
#     """
#     Call seqlets from the projection attribute and select seqlet attr matrix.

#     Parameters
#     ----------
#     proj_attr : torch.Tensor
#         Projection attribute matrix, shape (n_region, 1, length).
#     attr : torch.Tensor
#         Full attribute matrix, shape (n_region, 4, length).
#     seqlet_size : int
#         Size of the seqlet, shorter ones will be extended to this size.
#     min_seqlet_len : int
#         Minimum length of the seqlet when calling seqlets.

#     Returns
#     -------
#     seqlet_attr : torch.Tensor
#         Selected seqlet attribute matrix, shape (n_seqlet, 4, seqlet_size).
#     """
#     # find seqlet regions in proj_attr using the fast algorithm
#     result = recursive_seqlets(
#         proj_attr.astype("float32"),
#         threshold=0.01,
#         min_seqlet_len=min_seqlet_len,
#         max_seqlet_len=seqlet_size,
#         additional_flanks=0,
#     )
#     # Standard length to seqlet_size
#     center = (result["end"] + result["start"]) // 2
#     result["fix_start"] = center - seqlet_size // 2
#     result["fix_end"] = center + seqlet_size // 2
#     regions = result.loc[
#         ((result["fix_start"] > 0) & (result["fix_end"] < proj_attr.shape[-1])),
#         ["example_idx", "fix_start", "fix_end"],
#     ].values
#     regions = torch.from_numpy(regions)
#     # Select seqlet regions from full attr
#     batch_idx = regions[:, 0]  # (n_region,)
#     start = regions[:, 1]  # (n_region,)
#     region_length = regions[0, 2] - regions[0, 1]
#     offset = torch.arange(region_length)  # (region_length,)
#     seqlet_attr = attr[batch_idx.unsqueeze(1), :, start.unsqueeze(1) + offset].swapaxes(
#         1, 2
#     )  # (n_region, 4, region_length)
#     return seqlet_attr


# def tanh_weight(bs, seq_len, alpha=2):
#     """
#     Generates a weight vector with size, flat and 1 in the middle,
#     smoothly decrease to 0 on both sides.
#     """
#     x = torch.linspace(0, 100, seq_len)
#     weights = 0.5 * (torch.tanh((x - 20) / alpha) - torch.tanh((x - 80) / alpha))
#     weights = (weights - weights.min()) / (weights.max() - weights.min())
#     return weights.unsqueeze(0).repeat(bs, 1)
