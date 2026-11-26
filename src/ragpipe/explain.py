"""Reference-explanation construction and the decoder's input template.

The Amazon data has no human-written explanations, so reference explanations are
synthesised from signals that are present in each review: its title (summary), the
words in it that are most indicative of the gold sentiment (log-odds lexicon fitted
on the training set), and the gold helpfulness label. Retrieval is never used to
build a reference, so the with/without-retrieval ablation compares the two decoders
against the same targets.
"""
import math
from dataclasses import dataclass

from .data import SENTIMENT_NAMES
from .preprocess import clean_text, tokenize

IGNORE_INDEX = -100
HELPFUL_TOKENS = ["not_helpful", "helpful"]
PROMPT_SPECIALS = ["<title>", "<review>", "<sentiment>", "<helpful>", "<ctx>", "<ex>", "<none>",
                   "<explain>", "not_helpful"]

STOPWORDS = set("""
a an the and or but if then so of to in on at by for with from as is are was were be been
being it its it's this that these those i me my we our you your he she they them their his her
have has had do does did not no can could would should will just very really also than too
there here what which who when where how all any some more most much many one two get got
""".split())


class SentimentLexicon:
    """Per-class smoothed log-odds of each word versus all other classes."""

    def __init__(self, scores):
        self.scores = scores  # list (per class) of {word: log-odds}

    @classmethod
    def fit(cls, token_lists, labels, n_classes=3, alpha=1.0, min_count=1):
        counts = [dict() for _ in range(n_classes)]
        for toks, y in zip(token_lists, labels):
            for t in toks:
                counts[y][t] = counts[y].get(t, 0) + 1
        total = {}
        for c in counts:
            for w, n in c.items():
                total[w] = total.get(w, 0) + n
        vocab = [w for w, n in total.items() if n >= min_count and _is_content(w)]
        V = len(vocab)
        class_n = [sum(c.values()) for c in counts]
        all_n = sum(class_n)
        scores = []
        for y in range(n_classes):
            s = {}
            for w in vocab:
                in_c = counts[y].get(w, 0)
                out_c = total[w] - in_c
                s[w] = (math.log((in_c + alpha) / (class_n[y] + alpha * V))
                        - math.log((out_c + alpha) / (all_n - class_n[y] + alpha * V)))
            scores.append(s)
        return cls(scores)

    def keywords(self, tokens, label, n=2):
        """The n distinct words of `tokens` most indicative of `label` (positive score only)."""
        s = self.scores[label]
        seen = {t for t in tokens if t in s and s[t] > 0}
        return sorted(seen, key=lambda w: (-s[w], w))[:n]


def _is_content(word):
    return word.isalpha() and len(word) > 2 and word not in STOPWORDS


def _join(words):
    if len(words) <= 1:
        return " ".join(words)
    return " , ".join(words[:-1]) + " and " + words[-1]


_FALLBACK = {0: "describes problems with the product",
             1: "describes mixed experiences with the product",
             2: "describes a good experience with the product"}


def reference_explanation(summary, sentiment, helpful, keywords):
    """Two-sentence, already-tokenised reference explanation (tokens joined by spaces)."""
    title = " ".join(tokenize(clean_text(summary)))
    parts = []
    if title:
        parts.append(f'says " {title} "')
    if keywords:
        parts.append(f"mentions {_join(list(keywords))}")
    reason = " and ".join(parts) if parts else _FALLBACK[sentiment]
    first = f"the review is {SENTIMENT_NAMES[sentiment]} because the customer {reason} ."
    second = ("the detailed feedback makes it likely helpful to other shoppers ." if helpful
              else "it is probably not very helpful to other shoppers .")
    return f"{first} {second}"


@dataclass
class PromptConfig:
    max_title: int = 20      # review-title tokens kept in the prompt
    max_review: int = 96     # review tokens kept in the prompt
    max_retrieved: int = 32  # tokens kept per retrieved example


def build_prompt(review_tokens, sentiment, helpful, retrieved, cfg, title_tokens=()):
    """Template (one flat token sequence):
        <bos> <title> t_1..t_m <review> r_1..r_n <sentiment> LABEL <helpful> H
        <ctx> (<ex> LABEL_j e_1..e_m)*k | <none>  <explain>
    `retrieved` is a list of {"sentiment": int, "tokens": [...]}, or None for the baseline."""
    seq = ["<bos>", "<title>", *list(title_tokens)[:cfg.max_title],
           "<review>", *review_tokens[:cfg.max_review],
           "<sentiment>", SENTIMENT_NAMES[sentiment],
           "<helpful>", HELPFUL_TOKENS[helpful], "<ctx>"]
    if retrieved:
        for ex in retrieved:
            seq += ["<ex>", SENTIMENT_NAMES[ex["sentiment"]], *ex["tokens"][:cfg.max_retrieved]]
    else:
        seq.append("<none>")
    seq.append("<explain>")
    return seq


def make_lm_example(prompt_ids, target_ids, eos_id):
    """Next-token-prediction pair. Loss is computed only where the model predicts
    explanation tokens (and the final <eos>); prompt positions get IGNORE_INDEX."""
    full = list(prompt_ids) + list(target_ids) + [eos_id]
    inp, labels = full[:-1], full[1:]
    n_prompt = len(prompt_ids) - 1
    labels = [IGNORE_INDEX] * n_prompt + labels[n_prompt:]
    return inp, labels
