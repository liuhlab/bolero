from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, atten_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(atten_dropout)  # attention dropout init

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))  # attention dropout execute.
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAtten(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v, dropout=0.1):
        super(MultiHeadAtten, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_q = d_k
        self.d_k = d_k
        self.d_v = d_v  # typically, d_q = d_k = d_v

        self.query_linear = nn.Linear(
            self.d_model, self.d_q * self.num_heads, bias=False
        )
        self.key_linear = nn.Linear(self.d_model, self.d_k * self.num_heads, bias=False)
        self.value_linear = nn.Linear(
            self.d_model, self.d_v * self.num_heads, bias=False
        )
        self.out_linear = nn.Linear(self.d_v * self.num_heads, self.d_model)

        self.attention = ScaledDotProductAttention(temperature=self.d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_heads = self.d_k, self.d_v, self.num_heads

        # print(q.shape)
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # seperate the output of each head. batch_size * length_q * num_head * dim_v
        q = self.query_linear(q).view(sz_b, len_q, n_heads, d_k)
        k = self.key_linear(k).view(sz_b, len_k, n_heads, d_k)
        v = self.value_linear(v).view(sz_b, len_v, n_heads, d_v)

        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # batch_size * numm_head * length_q * dim_v

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)

        q, attn = self.attention(q, k, v, mask=mask)

        # reshape to combine the last two dims. batch_size * length_q * (num_head * dim_q)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.out_linear(q))
        q += residual  # add

        q = self.layer_norm(q)  # norm

        return q, attn


class CrossAttention(nn.Module):

    def __init__(self, d_model, d_in_query, d_k, d_v, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.d_model = d_model  # dim of current k, v
        self.d_in_query = d_in_query  # dim of input q.

        self.model_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.query_norm = nn.LayerNorm(d_in_query, eps=1e-6)

        self.d_q = d_k
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads

        self.transform_linear = nn.Linear(d_in_query, d_model, bias=False)
        self.query_linear = nn.Linear(d_in_query, self.d_q * self.num_heads, bias=False)
        self.key_linear = nn.Linear(d_model, self.d_k * self.num_heads, bias=False)
        self.value_linear = nn.Linear(d_model, self.d_v * self.num_heads, bias=False)
        self.out_linear = nn.Linear(self.d_v * self.num_heads, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k**0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, x, q_mask=None, mask=None):
        if self.d_in_query != self.d_model:
            residual = self.transform_linear(q)
        else:
            residual = q
        q = self.query_norm(q)
        x = self.model_norm(x)

        d_q, d_k, d_v, num_heads = self.d_q, self.d_k, self.d_v, self.num_heads
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), x.size(1), x.size(1)

        q = self.query_linear(q).view(sz_b, len_q, num_heads, d_q)
        k = self.key_linear(x).view(sz_b, len_k, num_heads, d_k)
        v = self.value_linear(x).view(sz_b, len_k, num_heads, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)

        out, attn = self.attention(q, k, v, mask=mask)
        # print(out.shape)
        out = out.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # print(out.shape)
        out = self.dropout(self.out_linear(out))
        out = out + residual

        return out, attn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_hid, d_k, d_v, dropout=0.2):
        super(TransformerEncoder, self).__init__()
        self.self_attn = MultiHeadAtten(d_model, n_heads, d_k, d_v, dropout=dropout)
        self.pos_ff = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)

    def forward(self, x, mask=None):
        enc_output, enc_self_attn = self.self_attn(x, x, x, mask=mask)
        enc_output = self.pos_ff(enc_output)
        return enc_output, enc_self_attn


class CrossAttenEncoder(nn.Module):
    def __init__(self, d_model, d_in_query, n_heads, d_hid, d_k, d_v, dropout=0.1):
        super(CrossAttenEncoder, self).__init__()
        self.cross_attn = CrossAttention(
            d_model, d_in_query, d_k, d_v, n_heads, dropout=dropout
        )
        self.pos_ff = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)

    def forward(self, x_query, x, mask=None):
        enc_output, enc_cross_attn = self.cross_attn(x_query, x, mask=mask)
        enc_output = self.pos_ff(enc_output)
        return enc_output, enc_cross_attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=128):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # print(self.pos_table[:, :x.size(1)])
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class AttentionPool(nn.Module):
    def __init__(self, d_model, pool_dim=1):
        super(AttentionPool, self).__init__()
        self.query = nn.Parameter(torch.randn(pool_dim, d_model))

    def forward(self, x):
        attn_scores = torch.matmul(self.query, x.transpose(1, 2))
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = torch.matmul(attn_weights, x)
        return pooled


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool1d(2), Conv1DBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = Conv1DBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.GELU(inplace=True),
            nn.Linear(ff_hidden_dim, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerStack(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout, num_layers):
        super(TransformerStack, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, ff_hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
