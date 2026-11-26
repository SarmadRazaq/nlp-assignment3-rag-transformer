"""Encoder-only multi-task model (Part A) and decoder-only language model (Part C)."""
from dataclasses import dataclass

import torch
import torch.nn as nn

from .transformer import Embeddings, TransformerBlock


@dataclass
class EncoderConfig:
    vocab_size: int
    max_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    n_sentiment: int = 3
    n_helpful: int = 2
    pad_id: int = 0


class MultiTaskEncoder(nn.Module):
    """Bidirectional Transformer encoder with two classification heads.

    The review vector is the masked mean of the final hidden states: every real
    token contributes and padding never does. Mean pooling summarises the whole
    review rather than relying on one position, which suits retrieval in Part B."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg.vocab_size, cfg.d_model, cfg.max_len, cfg.dropout, cfg.pad_id)
        self.blocks = nn.ModuleList([TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff,
                                                      cfg.dropout, causal=False)
                                     for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.d_model)

        def head(n_out):
            return nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, n_out))

        self.sentiment_head = head(cfg.n_sentiment)
        self.helpful_head = head(cfg.n_helpful)

    def forward(self, ids, mask):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln(x)
        m = mask.unsqueeze(-1).float()
        emb = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        return {"sentiment_logits": self.sentiment_head(emb),
                "helpful_logits": self.helpful_head(emb),
                "embedding": emb}


@dataclass
class DecoderConfig:
    vocab_size: int
    max_len: int = 320
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    pad_id: int = 0


class DecoderLM(nn.Module):
    """GPT-style decoder: causal self-attention only, so position t sees tokens <= t
    both in training and at inference. Output projection is tied to the token
    embedding, which cuts parameters and helps a small model on limited data."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg.vocab_size, cfg.d_model, cfg.max_len, cfg.dropout, cfg.pad_id)
        self.blocks = nn.ModuleList([TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff,
                                                      cfg.dropout, causal=True)
                                     for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.tok.weight

    def forward(self, ids):
        # Sequences are right-padded; with a causal mask, trailing pads can never
        # influence earlier positions, and their labels are ignored in the loss.
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln(x))


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, eos_id, temperature=0.0, top_k=None,
             device="cpu"):
    """Autoregressive decoding: append one token at a time, re-feeding the whole
    sequence so far (cropped to the model's context window). temperature=0 -> greedy.
    Returns only the newly generated ids (including <eos> if produced)."""
    ids = list(prompt_ids)
    new = []
    for _ in range(max_new_tokens):
        ctx = torch.tensor([ids[-model.cfg.max_len:]], device=device)
        logits = model(ctx)[0, -1]
        if temperature and temperature > 0:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
        else:
            nxt = int(torch.argmax(logits))
        ids.append(nxt)
        new.append(nxt)
        if nxt == eos_id:
            break
    return new
