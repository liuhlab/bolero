from collections import namedtuple

import torch.nn as nn
from torch.nn import functional as F
from vector_quantize_pytorch import VectorQuantize

from bolero.tl.generic.train import GenericModel

LossBreakdown = namedtuple(
    "LossBreakdown",
    [
        "commitment",
        "codebook_diversity",
        "orthogonal_reg",
        "inplace_optimize",
        "reconstruction",
    ],
)


class BaseVQVAE(GenericModel):
    default_config = {
        "input_dim": "estimate",
        "vq_dim": 128,
        "vq_kwargs": {
            "codebook_size": 4096,
            "decay": 0.8,
            "commitment_weight": 1,
            "kmeans_init": False,
            "use_cosine_sim": False,
            "threshold_ema_dead_code": 2,
            "layernorm_after_project_in": False,
            "heads": 1,  # number of heads to vector quantize, codebook shared across all heads
            "separate_codebook_per_head": True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
            "orthogonal_reg_weight": 0,  # in paper, they recommended a value of 10
            "orthogonal_reg_max_codes": 128,
            "orthogonal_reg_active_codes_only": False,
            "codebook_diversity_loss_weight": 0.0,
            "codebook_diversity_temperature": 100.0,
            "stochastic_sample_codes": False,
            "sample_codebook_temp": 1.0,
            "straight_through": False,
            "reinmax": False,  # using reinmax for improved straight-through, assuming straight through helps at all
        },
    }

    def __init__(self, input_dim, vq_dim, vq_kwargs):
        super().__init__()

        self.input_dim = input_dim
        hidden_dim = vq_dim * 4

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vq_dim),
            nn.BatchNorm1d(vq_dim),
        )

        self.vq = VectorQuantize(dim=vq_dim, **vq_kwargs)

        self.decoder = nn.Sequential(
            nn.Linear(vq_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        """VQ-VAE forward pass"""
        # Encoding
        z_e = self.encoder(x)

        # Vector quantization
        z_q, embed_ind, loss, loss_breakdown = self.vq(z_e, return_loss_breakdown=True)

        # Decoding
        x2 = self.decoder(z_q)

        # add reconstruction loss
        recon_loss = F.mse_loss(x2, x)
        loss = loss + recon_loss

        new_loss_breakdown = LossBreakdown(
            commitment=loss_breakdown.commitment.item(),
            codebook_diversity=loss_breakdown.codebook_diversity.item(),
            orthogonal_reg=loss_breakdown.orthogonal_reg.item(),
            inplace_optimize=loss_breakdown.inplace_optimize.item(),
            reconstruction=recon_loss.item(),
        )

        return x2, z_q, embed_ind, loss, new_loss_breakdown


# Under construction...

# class MLP(nn.Module):
#     """
#     Encoder using a linear layer followed by a GELU activation
#     """

#     def __init__(
#         self,
#         input_dim: int = 1,
#         hidden_dim: int = 512,
#         dropout: float = 0.1,
#         max_value: int = 512,
#     ):
#         super().__init__()
#         self.max_value = max_value
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim),
#             nn.Dropout(p=dropout),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#             x: Tensor, shape [batch_size, seq_len]

#         Returns
#         -------
#             x: Tensor, shape [batch_size, seq_len, hidden_dim]
#         """
#         if x.dim() == 2:
#             x = x.unsqueeze(-1)
#         x = torch.clamp(x, max=self.max_value)
#         x = self.model(x)
#         return x


# class DualEmbeddingVQVAE(BaseVQVAE):
#     def __init__(self, input_dim, hidden_dim, gene_embedding: nn.Embedding, **vq_kwargs):
#         self._gene_embedding = gene_embedding
#         embedding_dim = gene_embedding.embedding_dim  # default ESM embedding dim is 5120
#         self._gene_exp_embedding = MLP(1, input_dim=embedding_dim)

#         self.gene_encoder = MLP(embedding_dim, input_dim=input_dim)

#         super().__init__(input_dim=embedding_dim, hidden_dim=hidden_dim, **vq_kwargs)

#     def encoder(self, gene_exp: torch.Tensor):
#         """
#         gene_exp: (batch_size, gene_dim)
#         """
#         # add 0 before the first column, as 0 is the padding index
#         gene_exp = F.pad(gene_exp, (1, 0), value=0)
#         # get none zero positions in each row
#         non_zero_positions = torch.nonzero(gene_exp, as_tuple=True)
#         # padding zeros to the end of each nnz index to make it a fixed length
#         non_zero_positions = nn.utils.rnn.pad_sequence(
#             non_zero_positions, batch_first=True, padding_value=0
#         )
#         # get the gene embedding
#         # (batch_size, gene_dim, hidden_dim)
#         gene_nnz_embedding = self._gene_embedding(non_zero_positions)
#         gene_nnz_exp = gene_exp[non_zero_positions]

#         gene_nnz_exp_embedding = self._gene_exp_embedding(gene_nnz_exp)

#         total_embedding = gene_nnz_embedding + gene_nnz_exp_embedding
#         return total_embedding

#     def _zq_to_gene_emb(self, z_q):
#         """
#         z_q: (batch_size, seq_len, hidden_dim)
#         """
#         # get the index of the nearest embedding

#         gene_embedding = MLP(input_dim=z_q.shape[-1], hidden_dim=self._gene_embedding.embedding_dim)(z_q)
#         return gene_embedding

#     def _zq_to_gene_exp(self, z_q):
#         """
#         z_q: (batch_size, seq_len, hidden_dim)
#         """
#         gene_exp = MLP(input_dim=z_q.shape[-1], hidden_dim=1)(z_q)
#         return gene_exp


#     def decoder(self, z_q):
#         pass


"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualEncoderTextVQVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_embeddings, commitment_cost=0.25):
        super(DualEncoderTextVQVAE, self).__init__()

        # Untrainable Embedding layer for Encoder1
        self.embedding1 = nn.Embedding(vocab_size, embedding_dim)
        self.embedding1.weight.requires_grad = False  # Freeze weights

        # Trainable Embedding layer for Encoder2
        self.embedding2 = nn.Embedding(vocab_size, embedding_dim)

        # Encoder 1 (can be just the embedding if no additional layers are needed)
        self.encoder1 = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Encoder 2
        self.encoder2 = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # VQ-VAE Layers
        self.codebook = nn.Embedding(num_embeddings, hidden_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

        # Decoder 1
        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, vocab_size)
        )

        # Decoder 2
        self.decoder2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, vocab_size)
        )

    def forward(self, x):
        # Convert input to embeddings for both encoders
        x_embed1 = self.embedding1(x)
        x_embed2 = self.embedding2(x)

        # Encode using both encoders
        encoded1 = self.encoder1(x_embed1)
        encoded2 = self.encoder2(x_embed2)

        # Sum the encodings
        encoded_sum = encoded1 + encoded2

        # Quantization
        z_e = encoded_sum.unsqueeze(1)
        distances = (torch.sum(z_e**2, dim=2, keepdim=True)
                     + torch.sum(self.codebook.weight**2, dim=1)
                     - 2 * torch.matmul(z_e, self.codebook.weight.t()))

        min_encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(min_encoding_indices).squeeze(1)

        # VQ Loss
        loss_vq = F.mse_loss(z_q, encoded_sum.detach()) + self.commitment_cost * F.mse_loss(encoded_sum, z_q.detach())

        # Decode using both decoders
        decoded1 = self.decoder1(z_q)
        decoded2 = self.decoder2(z_q)

        return decoded1, decoded2, loss_vq

# Example usage
vocab_size = 10000  # Number of words in vocabulary
embedding_dim = 256  # Dimension of word embeddings
hidden_dim = 512     # Hidden layer size
num_embeddings = 512 # Number of vectors in codebook

model = DualEncoderTextVQVAE(vocab_size, embedding_dim, hidden_dim, num_embeddings)

# Example input (sequence of word indices)
input_text = torch.randint(0, vocab_size, (32, 20))  # Batch size 32, sequence length 20

# Forward pass
output1, output2, loss_vq = model(input_text)

print(f"Output1 Shape: {output1.shape}, Output2 Shape: {output2.shape}, VQ Loss: {loss_vq.item()}")
"""
