# %% [markdown]
# ## 5. Part B: Retrieval module
#
# The **store** is the encoder embedding of every training review (`results/train_embeddings.npy`).
# A **query** is the encoder embedding of a val/test review. Similarity is **cosine**: vectors are
# L2-normalised once, so scoring all 25k training reviews is a single matrix product. Search is
# exact (brute force), which takes milliseconds at this scale, so an approximate index would only
# add recall loss. When the training set itself is used as queries (to build the decoder's
# training prompts), each review's own row is excluded.
#
# ### 5.1 Quantitative retrieval quality
# There is no relevance ground truth, so the analysis uses label agreement as a proxy:
# *precision@k* is the share of the k neighbours that have the **same sentiment** (or the same
# category, or the same helpfulness label) as the query. *Diversity* is the mean pairwise cosine
# distance among the k neighbours. Two baselines are included: random retrieval, and the same
# architecture with **untrained** (random-init) weights.

# %%
train_emb = np.load(RESULTS_DIR / "train_embeddings.npy")
test_emb, val_emb = enc_out["test"]["embeddings"], enc_out["val"]["embeddings"]
index = VectorIndex(train_emb)
tr_sent, tr_cat, tr_help = train_df["sentiment"].to_numpy(), train_df["category"].to_numpy(dtype=object), train_df["helpful"].to_numpy()
te_sent, te_cat, te_help = test_df["sentiment"].to_numpy(), test_df["category"].to_numpy(dtype=object), test_df["helpful"].to_numpy()


def retrieval_quality(ids, store_vecs):
    unit = l2_normalize(store_vecs)
    k = ids.shape[1]
    div = np.nan
    if k > 1:
        v = unit[ids]                                   # (n, k, d)
        sims = np.einsum("nkd,njd->nkj", v, v)
        div = float((1 - (sims.sum((1, 2)) - k) / (k * (k - 1))).mean())
    return {"P@k sentiment": float((tr_sent[ids] == te_sent[:, None]).mean()),
            "P@k category": float((tr_cat[ids] == te_cat[:, None]).mean()),
            "P@k helpful": float((tr_help[ids] == te_help[:, None]).mean()),
            "diversity": div}


seed_everything()
untrained = MultiTaskEncoder(enc_cfg).to(DEVICE)
rand_train = evaluate_encoder(untrained, DataLoader(eval_sets["train"], batch_size=256), DEVICE)["embeddings"]
rand_test = evaluate_encoder(untrained, DataLoader(eval_sets["test"], batch_size=256), DEVICE)["embeddings"]
rand_index = VectorIndex(rand_train)

rows, rng = [], np.random.RandomState(SEED)
for k in (1, 3, 5, 10, 20):
    t0 = time.time()
    ids, scores = index.search(test_emb, k=k)
    ms = (time.time() - t0) * 1000 / len(test_emb)
    rows.append({"method": "trained encoder (cosine)", "k": k, **retrieval_quality(ids, train_emb),
                 "mean_sim": float(scores.mean()), "ms/query": ms})
    ids_r, _ = rand_index.search(rand_test, k=k)
    rows.append({"method": "untrained encoder", "k": k, **retrieval_quality(ids_r, rand_train)})
    rows.append({"method": "random", "k": k, **retrieval_quality(rng.randint(0, len(train_emb), (len(test_emb), k)), train_emb)})
ret_k = pd.DataFrame(rows)
ret_k.to_csv(RESULTS_DIR / "retrieval_quality_vs_k.csv", index=False)
display(ret_k.round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for m, g in ret_k.groupby("method"):
    ax[0].plot(g["k"], g["P@k sentiment"], "o-", label=m)
t = ret_k[ret_k["method"] == "trained encoder (cosine)"]
ax[0].set(title="Sentiment precision@k", xlabel="k", xscale="log")
ax[0].legend()
ax[1].plot(t["k"], t["P@k sentiment"], "o-", label="P@k sentiment")
ax[1].plot(t["k"], t["diversity"], "s-", label="diversity (1 - mean pairwise cos)")
ax[1].set(title="Relevance vs diversity (trained encoder)", xlabel="k", xscale="log")
ax[1].legend()
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "retrieval_k_analysis.png", dpi=120)
plt.show()

# %% [markdown]
# ### 5.2 Effect of the similarity metric
# Cosine is compared with raw dot product and Euclidean (L2) distance on the same embeddings
# (k = 5). Dot product is cosine multiplied by both norms, so it favours long-norm "hub" vectors.
# L2 on unnormalised vectors mixes direction with magnitude.

# %%
def search_metric(metric, q, store, k=5, bs=1024):
    out = []
    for s in range(0, len(q), bs):
        qb = q[s:s + bs]
        if metric == "dot":
            sim = qb @ store.T
        else:  # negative squared L2
            sim = -((qb ** 2).sum(1, keepdims=True) - 2 * qb @ store.T + (store ** 2).sum(1)[None])
        out.append(np.argsort(-sim, axis=1)[:, :k])
    return np.concatenate(out)


metric_rows = [{"metric": "cosine", **retrieval_quality(index.search(test_emb, k=5)[0], train_emb)}]
for m in ("dot", "euclidean"):
    ids_m = search_metric(m, test_emb, train_emb)
    hub = np.bincount(ids_m.ravel(), minlength=len(train_emb))
    metric_rows.append({"metric": m, **retrieval_quality(ids_m, train_emb),
                        "share of hits on top-1% items": hub[np.argsort(-hub)[:len(hub) // 100]].sum() / hub.sum()})
hub = np.bincount(index.search(test_emb, k=5)[0].ravel(), minlength=len(train_emb))
metric_rows[0]["share of hits on top-1% items"] = hub[np.argsort(-hub)[:len(hub) // 100]].sum() / hub.sum()
metric_df = pd.DataFrame(metric_rows)
metric_df.to_csv(RESULTS_DIR / "retrieval_similarity_metrics.csv", index=False)
norms = np.linalg.norm(train_emb, axis=1)
print(f"embedding norms: mean {norms.mean():.2f}, std {norms.std():.2f}, min {norms.min():.2f}, max {norms.max():.2f}")
display(metric_df.round(4))

# %% [markdown]
# ### 5.3 Qualitative examples
# Query reviews (test set) with their top-3 retrieved training reviews.

# %%
def show_neighbours(qi, k=3):
    ids, scores = index.search(test_emb[[qi]], k=k)
    q = test_df.iloc[qi]
    print(f"QUERY [{q['category']} | {SENTIMENT_NAMES[q['sentiment']]} | {HELPFUL_NAMES[q['helpful']]}] "
          f"\"{q['summary']}\": {q['text'][:220]}")
    for j, s in zip(ids[0], scores[0]):
        r = train_df.iloc[j]
        print(f"   {s:.3f} [{r['category']} | {SENTIMENT_NAMES[r['sentiment']]} | {HELPFUL_NAMES[r['helpful']]}] "
              f"\"{r['summary']}\": {r['text'][:160]}")
    print()


rng = np.random.RandomState(7)
example_queries = [rng.choice(np.where(te_sent == s)[0]) for s in (0, 1, 2)] + [int(rng.randint(len(test_df)))]
for qi in example_queries:
    show_neighbours(qi)

# %% [markdown]
# ### 5.4 Choice of k
# The prompt includes **k = 3** retrieved reviews. In the table above, sentiment precision stays
# roughly flat from k = 1 to k = 5 and then falls slowly, while diversity rises. k = 3 gives the
# decoder several agreeing signals plus some variety in wording. Each extra neighbour costs ~33
# tokens of decoder context (attention cost grows quadratically with length), and a larger k
# dilutes the context with less similar reviews. `TOP_K` is configurable below.

# %%
TOP_K = 3
ret_ids = {"train": index.search(train_emb, k=TOP_K, exclude=[[i] for i in range(len(train_emb))])[0],
           "val": index.search(val_emb, k=TOP_K)[0],
           "test": index.search(test_emb, k=TOP_K)[0]}
np.save(RESULTS_DIR / "test_retrieved_ids.npy", ret_ids["test"])
print({k: v.shape for k, v in ret_ids.items()})
