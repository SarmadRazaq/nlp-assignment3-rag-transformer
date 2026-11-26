"""Transformer building blocks written from scratch (no nn.Transformer / nn.MultiheadAttention)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product self-attention split across n_heads.

    attn_mask: optional (B, T) tensor with 1 for real tokens and 0 for padding.
    causal:    if True, position i may only attend to positions <= i.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, causal=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, C) -> (B, heads, T, d_head)
        q, k, v = (t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, h, T, T)

        allowed = torch.ones(T, T, dtype=torch.bool, device=x.device)
        if causal:
            allowed = torch.tril(allowed)  # lower-triangular: no access to future tokens
        allowed = allowed.view(1, 1, T, T)
        if attn_mask is not None:
            allowed = allowed & attn_mask.bool().view(B, 1, 1, T)  # hide padded keys
        scores = scores.masked_fill(~allowed, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights)  # rows that are fully masked (pad queries) -> 0
        out = self.drop(weights) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block: x + Attn(LN(x)), then x + FFN(LN(x)).
    Pre-LN trains stably without learning-rate warmup tricks on small models."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, causal=False):
        super().__init__()
        self.causal = causal
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.drop(self.attn(self.ln1(x), attn_mask, causal=self.causal))
        return x + self.ff(self.ln2(x))


class Embeddings(nn.Module):
    """Token embedding + learned absolute positional embedding."""

    def __init__(self, vocab_size, d_model, max_len, dropout=0.1, pad_id=0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        # small init (GPT-2 style); PyTorch's default N(0, 1) makes tied output logits huge
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.tok.weight[pad_id].zero_()

    def forward(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        return self.drop(self.tok(ids) + self.pos(pos))
