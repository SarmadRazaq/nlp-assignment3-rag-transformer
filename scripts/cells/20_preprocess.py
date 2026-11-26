# %% [markdown]
# ## 3. Preprocessing
#
# 1. **Cleaning** (`clean_text`): unescape HTML entities (`&amp;` → `&`), strip HTML tags and
#    URLs, lowercase, and collapse whitespace. Stopwords and punctuation are *kept* because
#    "not", "!" and "?" carry sentiment.
# 2. **Tokenisation** (`tokenize`): a regex word-level tokenizer. Words keep internal apostrophes,
#    so `doesn't` stays a single negation token, and each punctuation mark becomes its own token.
# 3. **Vocabulary** (`Vocab.build`): built from the **training split only**, with a minimum
#    frequency of 3 and at most 25k types. Validation and test words never enter the vocabulary,
#    and anything unseen maps to `<unk>`. Specials: `<pad>`=0, `<unk>`, `<bos>`, `<eos>`.
# 4. **Numericalisation + padding/truncation** (`encode_fixed`): token ids are truncated to
#    `MAX_LEN` and right-padded with `<pad>`, along with an attention mask (1 = real token).
#    `MAX_LEN = 256` covers the full text of ~93% of reviews (see the coverage table below).
#    Truncation keeps the beginning of the review, where the verdict usually appears.

# %%
for d in (train_df, val_df, test_df):
    d["tokens"] = d["text"].map(lambda t: tokenize(clean_text(t)))
    d["title_tokens"] = d["summary"].map(lambda t: tokenize(clean_text(t)))

lens = train_df["tokens"].map(len)
print("token length percentiles:", {p: int(np.percentile(lens, p)) for p in (50, 75, 90, 95, 99)})
print("coverage (reviews not truncated):", {L: f"{(lens <= L).mean():.1%}" for L in (64, 128, 256, 384)})

enc_vocab = Vocab.build(train_df["tokens"], min_freq=3, max_size=25000)
print("encoder vocab size:", len(enc_vocab))
for name, d in [("val", val_df), ("test", test_df)]:
    toks = [t for ts in d["tokens"] for t in ts]
    print(f"{name} OOV rate: {np.mean([enc_vocab[t] == enc_vocab.unk_id for t in toks]):.2%}")

example = train_df.iloc[0]
print("\nraw     :", example["text"][:200])
print("cleaned :", clean_text(example["text"])[:200])
print("tokens  :", example["tokens"][:25])
print("ids     :", enc_vocab.encode(example["tokens"][:25]))

# %%
MAX_LEN = 256


def encode_split(d, max_len=MAX_LEN):
    pairs = [enc_vocab.encode_fixed(t, max_len) for t in d["tokens"]]
    return (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))


enc_data = {}
for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    ids, masks = encode_split(d)
    enc_data[name] = EncodedReviews(ids, masks, d["sentiment"].values, d["helpful"].values)
print({k: tuple(v.ids.shape) for k, v in enc_data.items()})
