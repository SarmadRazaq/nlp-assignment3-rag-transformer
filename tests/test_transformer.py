import torch
import torch.nn as nn

from ragpipe.transformer import MultiHeadSelfAttention, TransformerBlock

torch.manual_seed(0)


def test_attention_preserves_shape():
    attn = MultiHeadSelfAttention(d_model=32, n_heads=4, dropout=0.0)
    x = torch.randn(2, 7, 32)
    assert attn(x).shape == (2, 7, 32)


def test_causal_block_cannot_see_future_positions():
    block = TransformerBlock(d_model=32, n_heads=4, d_ff=64, dropout=0.0, causal=True).eval()
    x = torch.randn(1, 6, 32)
    y = x.clone()
    y[:, 4:] = torch.randn(1, 2, 32)  # change only positions 4 and 5
    out_x, out_y = block(x), block(y)
    assert torch.allclose(out_x[:, :4], out_y[:, :4], atol=1e-6)
    assert not torch.allclose(out_x[:, 4:], out_y[:, 4:])


def test_padding_mask_hides_pad_positions_from_real_tokens():
    block = TransformerBlock(d_model=32, n_heads=4, d_ff=64, dropout=0.0, causal=False).eval()
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    x = torch.randn(1, 5, 32)
    y = x.clone()
    y[:, 3:] = torch.randn(1, 2, 32)  # perturb padded positions only
    assert torch.allclose(block(x, mask)[:, :3], block(y, mask)[:, :3], atol=1e-6)


def test_bidirectional_block_does_see_future_positions():
    block = TransformerBlock(d_model=32, n_heads=4, d_ff=64, dropout=0.0, causal=False).eval()
    x = torch.randn(1, 6, 32)
    y = x.clone()
    y[:, 5:] = torch.randn(1, 1, 32)
    assert not torch.allclose(block(x)[:, :1], block(y)[:, :1])


def test_no_builtin_transformer_modules_are_used():
    block = TransformerBlock(d_model=32, n_heads=4, d_ff=64)
    banned = (nn.MultiheadAttention, nn.Transformer, nn.TransformerEncoder,
              nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)
    assert not any(isinstance(m, banned) for m in block.modules())
