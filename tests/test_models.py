import torch

from ragpipe.models import EncoderConfig, MultiTaskEncoder, DecoderConfig, DecoderLM, generate

torch.manual_seed(0)


def test_encoder_returns_sentiment_logits_helpful_logits_and_fixed_size_embedding():
    cfg = EncoderConfig(vocab_size=50, max_len=16, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    enc = MultiTaskEncoder(cfg).eval()
    ids = torch.randint(4, 50, (3, 16))
    mask = torch.ones(3, 16, dtype=torch.long)
    out = enc(ids, mask)
    assert out["sentiment_logits"].shape == (3, 3)
    assert out["helpful_logits"].shape == (3, 2)
    assert out["embedding"].shape == (3, 32)


def test_encoder_embedding_ignores_padding_tokens():
    cfg = EncoderConfig(vocab_size=50, max_len=8, d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0)
    enc = MultiTaskEncoder(cfg).eval()
    a = torch.tensor([[5, 6, 7, 0, 0, 0, 0, 0]])
    b = torch.tensor([[5, 6, 7, 9, 9, 9, 9, 9]])  # same real tokens, junk in padded slots
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]])
    assert torch.allclose(enc(a, mask)["embedding"], enc(b, mask)["embedding"], atol=1e-5)


def _decoder():
    cfg = DecoderConfig(vocab_size=40, max_len=32, d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0)
    return DecoderLM(cfg).eval()


def test_decoder_logits_at_t_do_not_depend_on_later_tokens():
    dec = _decoder()
    x = torch.tensor([[2, 5, 6, 7, 8]])
    y = torch.tensor([[2, 5, 6, 30, 31]])
    assert torch.allclose(dec(x)[:, :3], dec(y)[:, :3], atol=1e-5)


class CountingStub(torch.nn.Module):
    """Predicts (last token + 1) and records every context it is shown."""

    def __init__(self, vocab=20, max_len=64):
        super().__init__()
        self.vocab, self.seen = vocab, []
        self.cfg = type("C", (), {"max_len": max_len})()

    def forward(self, ids):
        self.seen.append(ids[0].tolist())
        logits = torch.full((1, ids.size(1), self.vocab), -1e9)
        logits[0, -1, (ids[0, -1] + 1) % self.vocab] = 0.0
        return logits


def test_generate_is_autoregressive_and_stops_at_eos():
    stub = CountingStub()
    out = generate(stub, [5], max_new_tokens=10, eos_id=8)
    assert out == [6, 7, 8]
    assert stub.seen == [[5], [5, 6], [5, 6, 7]]  # each step conditions on all previous tokens


def test_generate_respects_max_new_tokens():
    out = generate(CountingStub(), [1], max_new_tokens=3, eos_id=19)
    assert out == [2, 3, 4]


def test_generate_with_real_decoder_returns_valid_token_ids():
    dec = _decoder()
    out = generate(dec, [2, 5, 6], max_new_tokens=5, eos_id=3)
    assert 1 <= len(out) <= 5 and all(0 <= t < 40 for t in out)
