import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask


class AttentionLayer(nn.Module):

    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):

        B, L, Q, _ = queries.shape
        _, S, P, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, Q, H, -1)
        keys = self.key_projection(keys).view(B, S, P, H, -1)
        values = self.value_projection(values).view(B, S, P, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, Q, -1)

        return self.out_projection(out), attn



class FullyAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullyAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):

        B, L, Q, H, E = queries.shape
        _, S, P, _, _ = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("BLQHE,BSPHE->BSPLQH", queries, keys)

        scaled = (scale * scores).view(B, -1, L, Q, H)
        scaled = torch.softmax(scaled, dim=1).view(B, S, P, L, Q, H)
        Attn = self.dropout(scaled)
        V = torch.einsum("BSPHE,BSPLQH->BLQHE", values, Attn)

        if self.output_attention:
            return V.contiguous(), Attn
        else:
            return V.contiguous(), None


class WhoTalksAttention(nn.Module):
    def __init__(self, mask_flag=False, scale=None, attention_dropout=0.1, output_attention=False):
        super(WhoTalksAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):

        queries = queries[:, :, -1:, :, :]
        B, L, Q, H, E = queries.shape
        _, S, P, _, _ = keys.shape

        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("BLQHE,BSPHE->BSPH", queries, keys)
        scaled = (scale * scores).view(B, -1, H)
        scaled = self.score_normalization(scaled).view(B, S, P, H)
        Attn = self.dropout(scaled)
        V = torch.einsum("BSPHE,BSPH->BSPHE", values, Attn)

        if self.output_attention:
            return V.contiguous(), Attn
        else:
            return V.contiguous(), None

    def score_normalization(self, seq):
        maximum = torch.max(seq, dim=1, keepdim=True).values
        minimum = torch.min(seq, dim=1, keepdim=True).values
        seq = (seq - minimum) / (maximum - minimum)
        return seq


class NoneAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(NoneAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        return values, None
