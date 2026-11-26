"""Training / evaluation loops for the encoder and decoder, plus text-generation metrics."""
import copy
import math
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from .explain import IGNORE_INDEX


# ----------------------------------------------------------------------------- encoder
class EncodedReviews(Dataset):
    def __init__(self, ids, masks, sentiment, helpful):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.long)
        self.sentiment = torch.tensor(sentiment, dtype=torch.long)
        self.helpful = torch.tensor(helpful, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.masks[i], self.sentiment[i], self.helpful[i]


def _cosine_warmup(optimizer, warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def evaluate_encoder(model, loader, device, sent_w=None, lam=1.0):
    """Returns losses, accuracy / macro-F1 for both tasks, predictions and embeddings."""
    model.eval()
    tot, n = 0.0, 0
    ps, ph, ys, yh, embs = [], [], [], [], []
    for ids, mask, s, h in loader:
        ids, mask, s, h = ids.to(device), mask.to(device), s.to(device), h.to(device)
        out = model(ids, mask)
        loss = (F.cross_entropy(out["sentiment_logits"], s, weight=sent_w)
                + lam * F.cross_entropy(out["helpful_logits"], h))
        tot += loss.item() * len(ids)
        n += len(ids)
        ps.append(out["sentiment_logits"].argmax(-1).cpu())
        ph.append(out["helpful_logits"].argmax(-1).cpu())
        ys.append(s.cpu())
        yh.append(h.cpu())
        embs.append(out["embedding"].float().cpu())
    ps, ph, ys, yh = (torch.cat(t).numpy() for t in (ps, ph, ys, yh))
    return {"loss": tot / n,
            "sent_acc": accuracy_score(ys, ps), "sent_f1": f1_score(ys, ps, average="macro"),
            "help_acc": accuracy_score(yh, ph), "help_f1": f1_score(yh, ph, average="macro"),
            "sent_pred": ps, "help_pred": ph, "sent_true": ys, "help_true": yh,
            "embeddings": torch.cat(embs).numpy()}


def train_encoder(model, train_ds, val_ds, device, epochs=10, lr=3e-4, batch_size=64,
                  weight_decay=0.01, lam=1.0, class_weights=None, patience=3,
                  label_smoothing=0.0, verbose=True):
    """Joint loss L = CE_sentiment (class-weighted) + lam * CE_helpfulness.
    Model selection on validation (sentiment macro-F1 + helpfulness macro-F1)."""
    model.to(device)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=256)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _cosine_warmup(opt, warmup=len(tl), total=epochs * len(tl))
    sent_w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float, device=device)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist, best, best_state, bad = [], -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n, correct_s, correct_h = time.time(), 0.0, 0, 0, 0
        for ids, mask, s, h in tl:
            ids, mask, s, h = ids.to(device), mask.to(device), s.to(device), h.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(ids, mask)
                loss = (F.cross_entropy(out["sentiment_logits"].float(), s, weight=sent_w,
                                        label_smoothing=label_smoothing)
                        + lam * F.cross_entropy(out["helpful_logits"].float(), h,
                                                label_smoothing=label_smoothing))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(ids)
            n += len(ids)
            correct_s += (out["sentiment_logits"].argmax(-1) == s).sum().item()
            correct_h += (out["helpful_logits"].argmax(-1) == h).sum().item()
        v = evaluate_encoder(model, vl, device, sent_w, lam)
        row = {"epoch": ep, "train_loss": tot / n, "train_sent_acc": correct_s / n,
               "train_help_acc": correct_h / n, "val_loss": v["loss"],
               "val_sent_acc": v["sent_acc"], "val_sent_f1": v["sent_f1"],
               "val_help_acc": v["help_acc"], "val_help_f1": v["help_f1"],
               "secs": time.time() - t0}
        hist.append(row)
        if verbose:
            print(f"ep {ep:2d} | train {row['train_loss']:.3f} | val {row['val_loss']:.3f} "
                  f"| sent F1 {row['val_sent_f1']:.3f} | help F1 {row['val_help_f1']:.3f} "
                  f"| {row['secs']:.0f}s")
        score = v["sent_f1"] + v["help_f1"]
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return hist


# ----------------------------------------------------------------------------- decoder
class LMExamples(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs  # list of (input_ids, labels)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def lm_collate(batch, pad_id=0):
    T = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), T), IGNORE_INDEX, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
    return ids, lab


@torch.no_grad()
def lm_loss(model, loader, device):
    """Mean token-level NLL over explanation tokens and the matching perplexity."""
    model.eval()
    nll, ntok = 0.0, 0
    for ids, lab in loader:
        ids, lab = ids.to(device), lab.to(device)
        logits = model(ids).float()
        nll += F.cross_entropy(logits.view(-1, logits.size(-1)), lab.view(-1),
                               ignore_index=IGNORE_INDEX, reduction="sum").item()
        ntok += (lab != IGNORE_INDEX).sum().item()
    return nll / ntok, math.exp(nll / ntok)


def train_decoder(model, train_ds, val_ds, device, epochs=10, lr=5e-4, batch_size=32,
                  weight_decay=0.01, patience=2, verbose=True):
    model.to(device)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lm_collate, drop_last=True)
    vl = DataLoader(val_ds, batch_size=64, collate_fn=lm_collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _cosine_warmup(opt, warmup=len(tl) // 2, total=epochs * len(tl))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist, best, best_state, bad = [], float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, nll, ntok = time.time(), 0.0, 0
        for ids, lab in tl:
            ids, lab = ids.to(device), lab.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(ids)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), lab.view(-1),
                                   ignore_index=IGNORE_INDEX)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            k = (lab != IGNORE_INDEX).sum().item()
            nll += loss.item() * k
            ntok += k
        v_loss, v_ppl = lm_loss(model, vl, device)
        row = {"epoch": ep, "train_loss": nll / ntok, "train_ppl": math.exp(nll / ntok),
               "val_loss": v_loss, "val_ppl": v_ppl, "secs": time.time() - t0}
        hist.append(row)
        if verbose:
            print(f"ep {ep:2d} | train ppl {row['train_ppl']:.2f} | val ppl {v_ppl:.2f} | {row['secs']:.0f}s")
        if v_loss < best:
            best, best_state, bad = v_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return hist


# ----------------------------------------------------------------------------- generation metrics
def rouge_l(hyp, ref):
    """ROUGE-L F1 from the longest common subsequence of two token lists."""
    if not hyp or not ref:
        return 0.0
    dp = [[0] * (len(ref) + 1) for _ in range(len(hyp) + 1)]
    for i, a in enumerate(hyp):
        for j, b in enumerate(ref):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a == b else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(hyp), lcs / len(ref)
    return 2 * p * r / (p + r)


def bleu(hyp, ref, max_n=4):
    """Sentence BLEU with add-one smoothing on n-gram precisions and a brevity penalty."""
    if not hyp:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        h = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        r = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        overlap = sum(min(c, r[g]) for g, c in h.items())
        logs.append(math.log((overlap + 1) / (max(sum(h.values()), 0) + 1)))
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    return bp * math.exp(sum(logs) / max_n)


def distinct_n(texts, n=2):
    """Fraction of unique n-grams across all generations (diversity)."""
    grams = [tuple(t[i:i + n]) for t in texts for i in range(len(t) - n + 1)]
    return len(set(grams)) / max(1, len(grams))
