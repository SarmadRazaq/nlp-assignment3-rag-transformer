# %% [markdown]
# ## 6. Part C: Decoder for explanation generation
#
# ### 6.1 Reference explanations
# The dataset has no human-written explanations, so **reference explanations are synthesised**
# from signals that are all **visible in the decoder's input**:
# * the review **title** (`summary`), usually a compressed verdict ("great case", "stopped working"),
# * the two words of the (truncated) review **most indicative of the sentiment label**, using
#   smoothed log-odds of word vs. other classes fitted on the training split (stopwords removed,
#   min count 20),
# * the **helpfulness** label.
#
# Template: `the review is <sentiment> because the customer says " <title> " and mentions <w1> and
# <w2> . <helpfulness sentence> .` For training and validation the gold labels are used. On the
# test set the references are built from the **encoder's predicted labels**, because the brief asks
# for an explanation of *the predicted sentiment*. The explanation must therefore stay consistent
# with the label in the prompt, even when the encoder is wrong.
#
# **Retrieval is never used to build a reference**, so the RAG decoder and the no-retrieval
# baseline are scored against identical targets. Any gain from retrieval comes from the context,
# not from leakage into the targets. The flip side is that the target is mostly predictable from
# the review itself, so retrieval can only help by supplying evidence (similar titles and
# sentiment vocabulary) that the decoder fails to extract from the review alone. Section 6.5
# measures how much that is worth.
#
# ### 6.2 Input template
# ```
# <bos> <title> t1 … t20 <review> r1 … r96 <sentiment> LABEL <helpful> H <ctx> <ex> LABEL_1 e1 … e32 <ex> LABEL_2 … <ex> LABEL_3 … <explain>  →  explanation <eos>
# ```
# Each retrieved example is its *title + ":" + text*, truncated to 32 tokens, and is prefixed by
# its (gold, training-set) sentiment. The baseline uses `<ctx> <none>` in place of the examples.
# Training uses teacher forcing with gold labels. At test time the encoder's predictions are
# inserted, which is the real pipeline. Loss covers only the explanation tokens (prompt positions
# are masked with -100, unit-tested).

# %%
PCFG = PromptConfig(max_title=20, max_review=96, max_retrieved=32)
lexicon = SentimentLexicon.fit(train_df["tokens"], train_df["sentiment"].values, n_classes=3, min_count=20)
for c, name in enumerate(SENTIMENT_NAMES):
    top = sorted(lexicon.scores[c], key=lambda w: -lexicon.scores[c][w])[:12]
    print(f"{name:9s} top log-odds words: {top}")


def make_references(d, sent_labels, help_labels, cfg=PCFG):
    """References use only what the prompt shows: the clipped title and the clipped review."""
    return [reference_explanation(" ".join(tt[:cfg.max_title]), int(y), int(h),
                                  lexicon.keywords(t[:cfg.max_review], int(y), n=2)).split()
            for tt, t, y, h in zip(d["title_tokens"], d["tokens"], sent_labels, help_labels)]


gold = {k: (d["sentiment"].values, d["helpful"].values) for k, d in [("train", train_df), ("val", val_df), ("test", test_df)]}
pred_test = (enc_out["test"]["sent_pred"], enc_out["test"]["help_pred"])
for k, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    d["ref_tokens"] = make_references(d, *gold[k])
    d["reference"] = d["ref_tokens"].map(" ".join)
test_df["ref_tokens_pred"] = make_references(test_df, *pred_test)
test_df["reference_pred"] = test_df["ref_tokens_pred"].map(" ".join)
display(train_df[["summary", "rating", "helpful", "reference"]].head(5))

dec_vocab = Vocab.build(list(train_df["tokens"]) + list(train_df["title_tokens"]) + list(train_df["ref_tokens"]),
                        min_freq=3, max_size=20000, extra_specials=PROMPT_SPECIALS)
print("decoder vocab size:", len(dec_vocab))
print(f"reference length: mean {train_df['ref_tokens'].map(len).mean():.1f} tokens, "
      f"max {train_df['ref_tokens'].map(len).max()}")

# %%
train_neigh = [{"sentiment": int(s), "tokens": tt + [":"] + t}
               for s, tt, t in zip(train_df["sentiment"], train_df["title_tokens"], train_df["tokens"])]


def make_pairs(d, retrieved_ids, sent_labels, help_labels, ref_col="ref_tokens", cfg=PCFG, max_ref=60):
    pairs = []
    for i in range(len(d)):
        retrieved = None if retrieved_ids is None else [train_neigh[j] for j in retrieved_ids[i]]
        prompt = build_prompt(d["tokens"].iloc[i], int(sent_labels[i]), int(help_labels[i]), retrieved, cfg,
                              title_tokens=d["title_tokens"].iloc[i])
        target = dec_vocab.encode(d[ref_col].iloc[i][:max_ref])
        pairs.append(make_lm_example(dec_vocab.encode(prompt), target, dec_vocab.eos_id))
    return LMExamples(pairs)


def make_lm_sets(cfg=PCFG):
    splits = [("train", train_df), ("val", val_df), ("test", test_df)]
    sets = {"rag": {k: make_pairs(d, ret_ids[k], *gold[k], cfg=cfg) for k, d in splits},
            "base": {k: make_pairs(d, None, *gold[k], cfg=cfg) for k, d in splits}}
    sets["rag"]["test_pred"] = make_pairs(test_df, ret_ids["test"], *pred_test, ref_col="ref_tokens_pred", cfg=cfg)
    sets["base"]["test_pred"] = make_pairs(test_df, None, *pred_test, ref_col="ref_tokens_pred", cfg=cfg)
    rand_ids = np.random.RandomState(SEED).randint(0, len(train_df), ret_ids["test"].shape)
    sets["rag"]["test_random_ctx"] = make_pairs(test_df, rand_ids, *gold["test"], cfg=cfg)
    return sets


lm = make_lm_sets()
DEC_MAX_LEN = max(len(x) for v in lm.values() for ds in v.values() for x, _ in ds.pairs) + 1
print("longest decoder sequence:", DEC_MAX_LEN - 1, "tokens")
x, y = lm["rag"]["train"][0]
print("\nexample RAG input :", dec_vocab.decode(x)[:700])
print("target            :", dec_vocab.decode([t for t in y if t >= 0]))

# %% [markdown]
# ### 6.3 Hyperparameter sweep (RAG decoder, 3 epochs each, validation perplexity)

# %%
DEC_BASE = dict(d_model=256, n_heads=4, n_layers=4, d_ff=1024, dropout=0.1, lr=5e-4, batch_size=32)
DEC_SWEEP = [("base", {}), ("lr=1e-3", {"lr": 1e-3}), ("lr=2e-4", {"lr": 2e-4}),
             ("layers=2", {"n_layers": 2}), ("layers=6", {"n_layers": 6}),
             ("d_model=384,d_ff=1536", {"d_model": 384, "d_ff": 1536}), ("heads=8", {"n_heads": 8}),
             ("dropout=0.2", {"dropout": 0.2}), ("batch=64", {"batch_size": 64}),
             ("max_review=48", {"max_review": 48})]


def run_decoder(hp, data, epochs, patience=2, verbose=False):
    seed_everything()
    cfg = DecoderConfig(vocab_size=len(dec_vocab), max_len=DEC_MAX_LEN, d_model=hp["d_model"], n_heads=hp["n_heads"],
                        n_layers=hp["n_layers"], d_ff=hp["d_ff"], dropout=hp["dropout"])
    model = DecoderLM(cfg)
    hist = train_decoder(model, data["train"], data["val"], DEVICE, epochs=epochs, lr=hp["lr"],
                         batch_size=hp["batch_size"], patience=patience, verbose=verbose)
    return model, cfg, pd.DataFrame(hist)


if RUN_TUNING:
    rows = []
    for name, change in DEC_SWEEP:
        data = lm["rag"]
        if "max_review" in change:  # sequence-length variant: rebuild prompts with a shorter review
            data = make_lm_sets(PromptConfig(max_title=20, max_review=change["max_review"], max_retrieved=32))["rag"]
        _, cfg, h = run_decoder({**DEC_BASE, **change}, data, epochs=3)
        best = h.loc[h["val_ppl"].idxmin()]
        rows.append({"config": name, **change, "params_M": round(sum(p.numel() for p in DecoderLM(cfg).parameters()) / 1e6, 2),
                     "best_epoch": int(best["epoch"]), "train_ppl": best["train_ppl"], "val_ppl": best["val_ppl"],
                     "sec_per_epoch": h["secs"].mean()})
        print(f"{name:12s} val ppl {best['val_ppl']:.3f} ({h['secs'].mean():.0f}s/ep)")
    dec_tuning = pd.DataFrame(rows)
    dec_tuning.to_csv(RESULTS_DIR / "decoder_tuning.csv", index=False)
else:
    dec_tuning = pd.read_csv(RESULTS_DIR / "decoder_tuning.csv")
display(dec_tuning.round(4))

DEC_FINAL = {**DEC_BASE, **dict(DEC_SWEEP)[dec_tuning.loc[dec_tuning["val_ppl"].idxmin(), "config"]]}
print("final decoder hyperparameters (used for BOTH RAG and baseline):", DEC_FINAL)
if "max_review" in DEC_FINAL:  # the sweep preferred a different prompt length: rebuild prompts to match
    PCFG = PromptConfig(max_title=20, max_review=int(DEC_FINAL.pop("max_review")), max_retrieved=32)
    lm = make_lm_sets(PCFG)

# %% [markdown]
# ### 6.4 Train the RAG decoder and the no-retrieval baseline
# Both use identical hyperparameters, data, targets and seeds. The only difference is whether the
# `<ctx>` segment contains the top-k retrieved reviews.

# %%
decoders, dec_hists = {}, {}
for variant in ("rag", "base"):
    ckpt = MODELS_DIR / f"decoder_{variant}.pt"
    if TRAIN:
        print(f"--- {variant} ---")
        m, cfg, h = run_decoder(DEC_FINAL, lm[variant], epochs=12, patience=2, verbose=True)
        torch.save({"state_dict": m.state_dict(), "config": cfg.__dict__, "hparams": DEC_FINAL,
                    "vocab": dec_vocab.to_dict(), "prompt_config": PCFG.__dict__, "top_k": TOP_K if variant == "rag" else 0}, ckpt)
        h.to_csv(RESULTS_DIR / f"decoder_{variant}_history.csv", index=False)
    else:
        ck = torch.load(ckpt, map_location=DEVICE)
        m = DecoderLM(DecoderConfig(**ck["config"])).to(DEVICE)
        m.load_state_dict(ck["state_dict"])
        h = pd.read_csv(RESULTS_DIR / f"decoder_{variant}_history.csv")
    decoders[variant], dec_hists[variant] = m.to(DEVICE).eval(), h

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for v, h in dec_hists.items():
    ax[0].plot(h["epoch"], h["train_loss"], "o-", label=f"{v} train")
    ax[0].plot(h["epoch"], h["val_loss"], "s--", label=f"{v} val")
    ax[1].plot(h["epoch"], h["val_ppl"], "o-", label=f"{v} val")
ax[0].set(title="Decoder token NLL", xlabel="epoch")
ax[1].set(title="Validation perplexity", xlabel="epoch")
for a in ax:
    a.legend()
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "decoder_learning_curves.png", dpi=120)
plt.show()

# %% [markdown]
# ### 6.5 Test perplexity and RAG ablation
# * **gold labels:** the prompt contains the gold sentiment/helpfulness, which isolates the decoder.
# * **predicted labels:** the prompt contains the encoder's predictions (the full pipeline).
# * **random context:** the RAG decoder receives 3 *random* training reviews in place of the
#   retrieved ones. If its perplexity rises, the model is really using the retrieved content.

# %%
def ppl(model, ds):
    return lm_loss(model, DataLoader(ds, batch_size=64, collate_fn=lm_collate), DEVICE)[1]


ppl_rows = [
    {"system": "RAG (top-3 retrieved)", "test ppl (gold labels)": ppl(decoders["rag"], lm["rag"]["test"]),
     "test ppl (predicted labels)": ppl(decoders["rag"], lm["rag"]["test_pred"])},
    {"system": "No retrieval (baseline)", "test ppl (gold labels)": ppl(decoders["base"], lm["base"]["test"]),
     "test ppl (predicted labels)": ppl(decoders["base"], lm["base"]["test_pred"])},
    {"system": "RAG decoder + random context", "test ppl (gold labels)": ppl(decoders["rag"], lm["rag"]["test_random_ctx"]),
     "test ppl (predicted labels)": np.nan},
]
ppl_df = pd.DataFrame(ppl_rows)
ppl_df.to_csv(RESULTS_DIR / "decoder_test_perplexity.csv", index=False)
display(ppl_df.round(3))

# %% [markdown]
# ### 6.6 Generation quality (full pipeline: predicted labels, greedy decoding)
# The metrics below are computed on 500 test reviews. **ROUGE-L / BLEU-4** measure overlap with
# the reference explanation. **Label consistency** is the share of explanations that state the
# predicted sentiment. **Title recall** is the share of reference-title tokens reproduced.
# **Grounding** is the share of generated content words that occur in the review or the retrieved
# context; the rest are unsupported (hallucinated) words. **Distinct-2** measures diversity.

# %%
# fixed words of the reference template; they are excluded when measuring grounding
TEMPLATE_WORDS = set(
    "the review is negative neutral positive because customer says and mentions detailed feedback makes it likely "
    "helpful to other shoppers probably not very describes problems with product mixed experiences a good experience".split())
N_GEN = 500
gen_idx = np.random.RandomState(SEED).choice(len(test_df), N_GEN, replace=False)


def explain_one(variant, i, sent, helpful):
    retrieved = [train_neigh[j] for j in ret_ids["test"][i]] if variant == "rag" else None
    prompt = build_prompt(test_df["tokens"].iloc[i], int(sent), int(helpful), retrieved, PCFG,
                          title_tokens=test_df["title_tokens"].iloc[i])
    out = generate(decoders[variant], dec_vocab.encode(prompt), max_new_tokens=60,
                   eos_id=dec_vocab.eos_id, device=DEVICE)
    return dec_vocab.decode(out).split(), prompt


gen_rows = []
t0 = time.time()
for i in gen_idx:
    s_pred, h_pred = enc_out["test"]["sent_pred"][i], enc_out["test"]["help_pred"][i]
    ref = test_df["ref_tokens_pred"].iloc[i]  # explanation of the *predicted* labels
    title = set(test_df["title_tokens"].iloc[i])
    for variant in ("rag", "base"):
        hyp, prompt = explain_one(variant, i, s_pred, h_pred)
        content = [w for w in hyp if w not in TEMPLATE_WORDS and w.isalpha()]
        support = set(prompt)
        gen_rows.append({"idx": i, "variant": variant, "generated": " ".join(hyp), "reference": " ".join(ref),
                         "rougeL": rouge_l(hyp, ref), "bleu4": bleu(hyp, ref),
                         "label_consistent": f"is {SENTIMENT_NAMES[s_pred]}" in " ".join(hyp),
                         "title_recall": len(title & set(hyp)) / max(1, len(title)),
                         "grounding": np.mean([w in support for w in content]) if content else 1.0,
                         "len": len(hyp)})
print(f"generated {len(gen_rows)} explanations in {time.time() - t0:.0f}s")
gen_df = pd.DataFrame(gen_rows)
gen_df.to_csv(RESULTS_DIR / "test_generations.csv", index=False)
gen_summary = gen_df.groupby("variant").agg(rougeL=("rougeL", "mean"), bleu4=("bleu4", "mean"),
                                           label_consistency=("label_consistent", "mean"),
                                           title_recall=("title_recall", "mean"), grounding=("grounding", "mean"),
                                           avg_len=("len", "mean"))
gen_summary["distinct2"] = [distinct_n([g.split() for g in gen_df[gen_df.variant == v]["generated"]]) for v in gen_summary.index]
gen_summary = gen_summary.rename(index={"rag": "RAG (top-3)", "base": "No retrieval"})
gen_summary.to_csv(RESULTS_DIR / "generation_metrics.csv")
display(gen_summary.round(4))

ablation = ppl_df.set_index("system").iloc[:2].copy()
ablation[["rougeL", "bleu4", "label_consistency", "grounding"]] = gen_summary[["rougeL", "bleu4", "label_consistency", "grounding"]].values
ablation.to_csv(RESULTS_DIR / "rag_ablation.csv")
print("RAG ablation (test set):")
display(ablation.round(4))

# %% [markdown]
# ### 6.7 Generated examples

# %%
def show_example(i):
    r = test_df.iloc[i]
    s_pred, h_pred = enc_out["test"]["sent_pred"][i], enc_out["test"]["help_pred"][i]
    print(f"[{r['category']}] {r['rating']}★ gold={SENTIMENT_NAMES[r['sentiment']]}/{HELPFUL_NAMES[r['helpful']]} "
          f"pred={SENTIMENT_NAMES[s_pred]}/{HELPFUL_NAMES[h_pred]}")
    print(f"  REVIEW   : \"{r['summary']}\": {r['text'][:250]}")
    for j in ret_ids["test"][i]:
        print(f"  RETRIEVED: [{SENTIMENT_NAMES[train_df['sentiment'].iloc[j]]}] \"{train_df['summary'].iloc[j]}\"")
    g = gen_df[gen_df.idx == i].set_index("variant")["generated"]
    print(f"  REFERENCE: {r['reference_pred']}")
    print(f"  RAG      : {g['rag']}")
    print(f"  BASELINE : {g['base']}\n")


sel = list(gen_idx[:4])
wrong = [i for i in gen_idx if enc_out["test"]["sent_pred"][i] != test_df["sentiment"].iloc[i]]
neutral = [i for i in gen_idx if test_df["sentiment"].iloc[i] == 1]
for i in sel + wrong[:1] + neutral[:1]:
    show_example(i)
