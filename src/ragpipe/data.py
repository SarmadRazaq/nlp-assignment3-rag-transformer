"""Dataset construction: load Amazon 5-core files, derive labels, sample, split."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SENTIMENT_NAMES = ["negative", "neutral", "positive"]
HELPFUL_NAMES = ["not helpful", "helpful"]


def rating_to_sentiment(rating: float) -> int:
    """1-2 stars -> 0 (negative), 3 -> 1 (neutral), 4-5 -> 2 (positive)."""
    if rating <= 2:
        return 0
    if rating == 3:
        return 1
    return 2


def load_category(path, category, min_votes=2):
    """Read one JSON-lines file, keeping reviews with text and >= min_votes helpfulness votes."""
    rows = []
    with open(path, encoding="utf8") as f:
        for line in f:
            d = json.loads(line)
            text = (d.get("reviewText") or "").strip()
            yes, total = d.get("helpful", [0, 0])
            if not text or total < min_votes:
                continue
            rows.append({
                "category": category,
                "asin": d["asin"],
                "text": text,
                "summary": (d.get("summary") or "").strip(),
                "rating": int(d["overall"]),
                "helpful_yes": yes,
                "helpful_total": total,
            })
    df = pd.DataFrame(rows)
    df["sentiment"] = df["rating"].map(rating_to_sentiment)
    df["helpful"] = (df["helpful_yes"] / df["helpful_total"] >= 0.75).astype(int)
    return df


def sample_category(df, n_total, quotas, seed):
    """Sample n_total rows using per-sentiment quotas (fractions); shortfalls are filled
    from the remaining pool so the category always contributes n_total reviews."""
    rng = np.random.RandomState(seed)
    parts, used = [], set()
    for label, frac in quotas.items():
        pool = df[df["sentiment"] == label]
        take = min(len(pool), int(round(n_total * frac)))
        idx = rng.choice(pool.index.values, size=take, replace=False)
        parts.append(idx)
        used.update(idx)
    chosen = np.concatenate(parts)
    short = n_total - len(chosen)
    if short > 0:
        rest = df.index.difference(list(used)).values
        chosen = np.concatenate([chosen, rng.choice(rest, size=short, replace=False)])
    return df.loc[chosen]


def build_dataset(raw_dir, categories, per_category=12000,
                  quotas=None, seed=42, min_votes=2):
    """Build the combined, de-duplicated subset across categories."""
    quotas = quotas or {0: 0.30, 1: 0.25, 2: 0.45}
    frames = []
    for i, (cat, fname) in enumerate(categories.items()):
        df = load_category(Path(raw_dir) / fname, cat, min_votes=min_votes)
        df = df.drop_duplicates(subset="text")
        frames.append(sample_category(df, per_category, quotas, seed + i))
    out = pd.concat(frames, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_dataset(df, train=0.70, val=0.15, seed=42):
    """Stratified 70/15/15 split on (category, sentiment) so every split mirrors the mix."""
    rng = np.random.RandomState(seed)
    splits = {"train": [], "val": [], "test": []}
    for _, g in df.groupby(["category", "sentiment"]):
        idx = rng.permutation(g.index.values)
        n_tr, n_va = int(len(idx) * train), int(len(idx) * val)
        splits["train"].extend(idx[:n_tr])
        splits["val"].extend(idx[n_tr:n_tr + n_va])
        splits["test"].extend(idx[n_tr + n_va:])
    return {k: df.loc[v].sample(frac=1.0, random_state=seed).reset_index(drop=True)
            for k, v in splits.items()}
