"""Text preprocessing implemented from scratch: cleaning, tokenization, vocabulary, encoding."""
import html
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")
# words (with internal apostrophes such as don't / it's), numbers, or single punctuation marks
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def clean_text(text: str) -> str:
    """Lowercase, unescape HTML entities, drop HTML tags and URLs, collapse whitespace."""
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.lower()
    return _SPACES.sub(" ", text).strip()


def tokenize(text: str) -> list:
    """Regex word-level tokenizer; punctuation becomes its own token."""
    return _TOKEN.findall(text)


class Vocab:
    """Token <-> index mapping. Built from training tokens only, so val/test words
    never leak into the vocabulary; anything unseen maps to <unk>."""

    SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = 0, 1, 2, 3

    @classmethod
    def build(cls, token_lists, min_freq=2, max_size=None, extra_specials=()):
        counts = {}
        for toks in token_lists:
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
        specials = cls.SPECIALS + [s for s in extra_specials if s not in cls.SPECIALS]
        words = [w for w, c in counts.items() if c >= min_freq and w not in specials]
        words.sort(key=lambda w: (-counts[w], w))  # most frequent first, ties alphabetical
        if max_size is not None:
            words = words[: max(0, max_size - len(specials))]
        return cls(specials + words)

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_id)

    def encode(self, tokens):
        return [self[t] for t in tokens]

    def encode_fixed(self, tokens, max_len):
        """Truncate to max_len, then right-pad with <pad>; returns (ids, attention_mask)."""
        ids = self.encode(tokens)[:max_len]
        mask = [1] * len(ids)
        pad = max_len - len(ids)
        return ids + [self.pad_id] * pad, mask + [0] * pad

    def decode(self, ids):
        special = set(range(len(self.SPECIALS)))
        return " ".join(self.itos[i] for i in ids if i not in special)

    def to_dict(self):
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d):
        return cls(d["itos"])
