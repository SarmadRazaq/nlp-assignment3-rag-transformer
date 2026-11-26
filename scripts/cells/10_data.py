# %% [markdown]
# ## 2. Dataset construction
#
# * **Categories:** Beauty, Cell Phones & Accessories, and Sports & Outdoors. These are three very
#   different domains, which gives diverse vocabulary and makes the retrieval analysis meaningful.
# * **Pool:** only reviews with **≥ 2 helpfulness votes**, so that the derived label is grounded
#   in real reader feedback. A review with 0 votes is not "unhelpful", just unseen.
# * **Derived task (helpfulness):** `helpful = 1` if `helpful_yes / helpful_total ≥ 0.75`. This
#   is a property of the text (detail, specificity, balance) that readers reward, it is distinct
#   from sentiment, and it is useful downstream: the explanation can say whether the review is
#   worth reading.
# * **Sentiment:** ratings 1-2 → negative, 3 → neutral, 4-5 → positive.
# * **Sampling:** 12,000 reviews per category (36,000 total), with sentiment quotas of 30% neg /
#   25% neu / 45% pos. The raw data is ~70% positive, which would leave the neutral class almost
#   unlearnable. Duplicate texts are dropped.
# * **Split:** 70 / 15 / 15, stratified on (category, sentiment).

# %%
CATEGORIES = {"Beauty": "Beauty_5.json",
              "Cell_Phones": "Cell_Phones_and_Accessories_5.json",
              "Sports": "Sports_and_Outdoors_5.json"}
df = build_dataset(DATA_DIR, CATEGORIES, per_category=12000, seed=SEED)
splits = split_dataset(df, seed=SEED)
train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
print({k: len(v) for k, v in splits.items()}, "total", len(df))
display(pd.crosstab(df["category"], df["sentiment"].map(dict(enumerate(SENTIMENT_NAMES))), margins=True))
display(pd.crosstab(df["sentiment"].map(dict(enumerate(SENTIMENT_NAMES))), df["helpful"], normalize="index")
        .rename(columns={0: "not helpful", 1: "helpful"}).round(3))

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
df["rating"].value_counts().sort_index().plot.bar(ax=ax[0], title="Star rating")
df.groupby("category")["helpful"].mean().plot.bar(ax=ax[1], title="Helpful rate by category")
df["text"].str.split().str.len().clip(upper=600).plot.hist(bins=60, ax=ax[2], title="Words per review (clipped at 600)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "data_overview.png", dpi=120)
plt.show()
