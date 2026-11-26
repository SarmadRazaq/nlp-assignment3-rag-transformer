# %% [markdown]
# ## 4. Part A: Multi-task encoder
#
# **Architecture.** Token + learned positional embeddings → N pre-LayerNorm Transformer blocks
# (bidirectional multi-head self-attention with a padding mask, plus a GELU feed-forward) → final
# LayerNorm → **masked mean pooling** → the review embedding `e ∈ R^d`. Two small MLP heads sit on
# `e`: sentiment (3-way) and helpfulness (2-way). Mean pooling makes `e` a summary of the whole
# review, which is what retrieval needs, and the padding mask guarantees `<pad>` never
# contributes (unit-tested).
#
# **Loss.** `L = CE_sentiment(class-weighted) + λ · CE_helpfulness`. Sentiment classes are weighted
# by inverse frequency, because neutral is the smallest and hardest class. λ weighs the auxiliary
# task. Sharing the encoder means the embedding has to capture both *polarity* and *how
# informative* a review is.
#
# **Training.** AdamW (weight decay 0.01), 1 epoch of linear warmup then cosine decay, gradient
# clipping at 1.0, fp16 autocast on GPU, and early stopping on validation
# `macro-F1_sentiment + macro-F1_helpful` (patience 3). The best checkpoint is restored.
#
# ### 4.1 Hyperparameter sweep
# Each configuration changes one factor from the base and trains for 4 epochs. It is scored on
# the validation split.

# %%
counts = np.bincount(train_df["sentiment"], minlength=3)
CLASS_WEIGHTS = (counts.sum() / (3 * counts)).tolist()
print("sentiment class weights:", np.round(CLASS_WEIGHTS, 3))

ENC_BASE = dict(d_model=256, n_heads=4, n_layers=4, d_ff=512, dropout=0.1,
                lr=3e-4, batch_size=64, max_len=256, lam=1.0)
ENC_SWEEP = [
    ("base", {}),
    ("lr=1e-4", {"lr": 1e-4}),
    ("lr=1e-3", {"lr": 1e-3}),
    ("layers=2", {"n_layers": 2}),
    ("layers=6", {"n_layers": 6}),
    ("d_model=128,d_ff=256", {"d_model": 128, "d_ff": 256}),
    ("heads=8", {"n_heads": 8}),
    ("dropout=0.3", {"dropout": 0.3}),
    ("batch=32", {"batch_size": 32}),
    ("max_len=128", {"max_len": 128}),
    ("lambda=0.5", {"lam": 0.5}),
]


def truncate_ds(ds, L):
    return EncodedReviews(ds.ids[:, :L].numpy(), ds.masks[:, :L].numpy(), ds.sentiment.numpy(), ds.helpful.numpy())


def run_encoder(hp, epochs, patience=3, verbose=False):
    seed_everything()
    cfg = EncoderConfig(vocab_size=len(enc_vocab), max_len=hp["max_len"], d_model=hp["d_model"],
                        n_heads=hp["n_heads"], n_layers=hp["n_layers"], d_ff=hp["d_ff"], dropout=hp["dropout"])
    model = MultiTaskEncoder(cfg)
    tr, va = enc_data["train"], enc_data["val"]
    if hp["max_len"] != MAX_LEN:
        tr, va = truncate_ds(tr, hp["max_len"]), truncate_ds(va, hp["max_len"])
    hist = train_encoder(model, tr, va, DEVICE, epochs=epochs, lr=hp["lr"], batch_size=hp["batch_size"],
                         lam=hp["lam"], class_weights=CLASS_WEIGHTS, patience=patience, verbose=verbose)
    return model, cfg, pd.DataFrame(hist)


if RUN_TUNING:
    rows = []
    for name, change in ENC_SWEEP:
        hp = {**ENC_BASE, **change}
        _, cfg, h = run_encoder(hp, epochs=4)
        best = h.loc[(h["val_sent_f1"] + h["val_help_f1"]).idxmax()]
        n_params = sum(p.numel() for p in MultiTaskEncoder(cfg).parameters())
        rows.append({"config": name, **change, "params_M": round(n_params / 1e6, 2),
                     "best_epoch": int(best["epoch"]), "val_sent_acc": best["val_sent_acc"],
                     "val_sent_f1": best["val_sent_f1"], "val_help_acc": best["val_help_acc"],
                     "val_help_f1": best["val_help_f1"], "val_loss": best["val_loss"],
                     "sec_per_epoch": h["secs"].mean()})
        print(f"{name:22s} sentF1 {best['val_sent_f1']:.4f}  helpF1 {best['val_help_f1']:.4f}  ({h['secs'].mean():.0f}s/ep)")
    enc_tuning = pd.DataFrame(rows)
    enc_tuning.to_csv(RESULTS_DIR / "encoder_tuning.csv", index=False)
else:
    enc_tuning = pd.read_csv(RESULTS_DIR / "encoder_tuning.csv")
enc_tuning["score"] = enc_tuning["val_sent_f1"] + enc_tuning["val_help_f1"]
display(enc_tuning.round(4))

# %% [markdown]
# ### 4.2 Final encoder
# The final model combines the changes that helped in the sweep (chosen by the cell below from the
# sweep log). It trains for up to 15 epochs with early stopping.

# %%
ENC_FINAL = dict(ENC_BASE)
_lookup = {name: change for name, change in ENC_SWEEP}
_base_score = enc_tuning.loc[enc_tuning["config"] == "base", "score"].item()
for _, r in enc_tuning.sort_values("score", ascending=False).iterrows():
    if r["config"] != "base" and r["score"] > _base_score + 0.005:
        change = _lookup[r["config"]]
        if not any(k in ENC_FINAL and ENC_FINAL[k] != ENC_BASE[k] for k in change):  # one change per factor
            ENC_FINAL.update(change)
print("final encoder hyperparameters:", ENC_FINAL)

enc_ckpt = MODELS_DIR / "encoder.pt"
if TRAIN:
    encoder, enc_cfg, enc_hist = run_encoder(ENC_FINAL, epochs=15, patience=3, verbose=True)
    torch.save({"state_dict": encoder.state_dict(), "config": enc_cfg.__dict__, "hparams": ENC_FINAL,
                "vocab": enc_vocab.to_dict()}, enc_ckpt)
    enc_hist.to_csv(RESULTS_DIR / "encoder_history.csv", index=False)
else:
    ck = torch.load(enc_ckpt, map_location=DEVICE)
    enc_cfg = EncoderConfig(**ck["config"])
    encoder = MultiTaskEncoder(enc_cfg).to(DEVICE)
    encoder.load_state_dict(ck["state_dict"])
    enc_hist = pd.read_csv(RESULTS_DIR / "encoder_history.csv")
print(f"encoder parameters: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(enc_hist["epoch"], enc_hist["train_loss"], "o-", label="train")
ax[0].plot(enc_hist["epoch"], enc_hist["val_loss"], "o-", label="val")
ax[0].set_title("Combined loss")
ax[1].plot(enc_hist["epoch"], enc_hist["train_sent_acc"], "o-", label="train acc")
ax[1].plot(enc_hist["epoch"], enc_hist["val_sent_acc"], "o-", label="val acc")
ax[1].plot(enc_hist["epoch"], enc_hist["val_sent_f1"], "s--", label="val macro-F1")
ax[1].set_title("Sentiment")
ax[2].plot(enc_hist["epoch"], enc_hist["train_help_acc"], "o-", label="train acc")
ax[2].plot(enc_hist["epoch"], enc_hist["val_help_acc"], "o-", label="val acc")
ax[2].plot(enc_hist["epoch"], enc_hist["val_help_f1"], "s--", label="val macro-F1")
ax[2].set_title("Helpfulness")
for a in ax:
    a.set_xlabel("epoch")
    a.legend()
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "encoder_learning_curves.png", dpi=120)
plt.show()

# %% [markdown]
# ### 4.3 Test-set evaluation

# %%
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

ENC_LEN = enc_cfg.max_len
eval_sets = {k: (truncate_ds(v, ENC_LEN) if ENC_LEN != MAX_LEN else v) for k, v in enc_data.items()}
enc_out = {k: evaluate_encoder(encoder, DataLoader(v, batch_size=256), DEVICE) for k, v in eval_sets.items()}
te = enc_out["test"]
print(f"TEST sentiment  acc {te['sent_acc']:.4f}  macro-F1 {te['sent_f1']:.4f}")
print(f"TEST helpful    acc {te['help_acc']:.4f}  macro-F1 {te['help_f1']:.4f}")
print(classification_report(te["sent_true"], te["sent_pred"], target_names=SENTIMENT_NAMES, digits=4))
print(classification_report(te["help_true"], te["help_pred"], target_names=HELPFUL_NAMES, digits=4))

maj_sent = np.bincount(train_df["sentiment"]).argmax()
maj_help = np.bincount(train_df["helpful"]).argmax()
enc_metrics = {
    "test_sentiment_accuracy": te["sent_acc"], "test_sentiment_macro_f1": te["sent_f1"],
    "test_helpful_accuracy": te["help_acc"], "test_helpful_macro_f1": te["help_f1"],
    "majority_baseline_sentiment_accuracy": float((te["sent_true"] == maj_sent).mean()),
    "majority_baseline_helpful_accuracy": float((te["help_true"] == maj_help).mean()),
    "sentiment_report": classification_report(te["sent_true"], te["sent_pred"], target_names=SENTIMENT_NAMES, output_dict=True),
    "helpful_report": classification_report(te["help_true"], te["help_pred"], target_names=HELPFUL_NAMES, output_dict=True),
    "per_category_sentiment_acc": {c: float((te["sent_pred"][m] == te["sent_true"][m]).mean())
                                   for c in CATEGORIES for m in [(test_df["category"] == c).values]},
}
json.dump(enc_metrics, open(RESULTS_DIR / "encoder_test_metrics.json", "w"), indent=2)
print("per-category sentiment accuracy:", {k: round(v, 4) for k, v in enc_metrics["per_category_sentiment_acc"].items()})

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.heatmap(confusion_matrix(te["sent_true"], te["sent_pred"], normalize="true"), annot=True, fmt=".2f",
            xticklabels=SENTIMENT_NAMES, yticklabels=SENTIMENT_NAMES, cmap="Blues", ax=ax[0])
ax[0].set(title="Sentiment (row-normalised)", xlabel="predicted", ylabel="true")
sns.heatmap(confusion_matrix(te["help_true"], te["help_pred"], normalize="true"), annot=True, fmt=".2f",
            xticklabels=HELPFUL_NAMES, yticklabels=HELPFUL_NAMES, cmap="Greens", ax=ax[1])
ax[1].set(title="Helpfulness (row-normalised)", xlabel="predicted", ylabel="true")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "encoder_confusion_matrices.png", dpi=120)
plt.show()

# %% [markdown]
# ### 4.4 Save the review embeddings
# Embeddings are saved for the **entire training set** (the retrieval store), plus val/test (the
# queries). Row `i` of `train_embeddings.npy` corresponds to row `i` of `train_index.csv`.

# %%
for k in ("train", "val", "test"):
    np.save(RESULTS_DIR / f"{k}_embeddings.npy", enc_out[k]["embeddings"].astype(np.float32))
train_df[["category", "asin", "rating", "sentiment", "helpful"]].to_csv(RESULTS_DIR / "train_index.csv", index_label="row")
print("train embeddings:", enc_out["train"]["embeddings"].shape)
