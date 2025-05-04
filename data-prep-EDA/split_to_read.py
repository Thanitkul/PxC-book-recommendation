#!/usr/bin/env python3
"""
split_to_read.py

•  Tag-aware pre-filter: cap each tag’s frequency at the 80-th percentile
   of its global occurrence across wish-lists (using the top-5 tags per book).
•  Users with ≤ MIN_WISH_COUNT remaining items are dropped from consideration.
•  A fixed fraction TEST_FRAC of *all* remaining users is sampled
   for the test split (provided there are enough eligible users).
"""

import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── CONFIG ─────────────────────────────────────────────────────────
DATA_DIR        = "clean"
WISHLIST_CSV    = os.path.join(DATA_DIR, "to_read.csv")
BOOK_TAGS_CSV   = os.path.join(DATA_DIR, "book_tags.csv")

TRAIN_OUT       = os.path.join(DATA_DIR, "to_read_train.csv")
TEST_OUT        = os.path.join(DATA_DIR, "to_read_test.csv")
DIST_PLOT_OUT   = os.path.join(DATA_DIR, "wishlist_distribution.png")

TEST_FRAC       = 0.10      # wanted fraction of all users → test
MIN_WISH_COUNT  = 11        # user needs > this after filtering

RNG_SEED        = 42        # reproducibility
np.random.seed(RNG_SEED)

# ─── 1) LOAD DATA ──────────────────────────────────────────────────
wl_df      = pd.read_csv(WISHLIST_CSV)          # columns: user_id, book_id
book_tags  = pd.read_csv(BOOK_TAGS_CSV)         # columns: book_id, tag_id, count

# ─── 2) BUILD top-5 TAG LIST PER BOOK ──────────────────────────────
top_tags: Dict[int, List[int]] = {}
for bid, grp in book_tags.groupby("book_id"):
    tag_list = grp.sort_values("count", ascending=False).tag_id.tolist()
    top_tags[bid] = (tag_list + [0]*5)[:5]

# ─── 3) GLOBAL TAG COUNTS & CAP AT 80-th PERCENTILE ────────────────
tag_counts = defaultdict(int)
for b in wl_df.book_id:
    for t in top_tags.get(b, []):
        if t:                                   # ignore padding 0
            tag_counts[t] += 1

all_tag_freqs = np.array(list(tag_counts.values()))
tag_cap = int(np.percentile(all_tag_freqs, 90))
print(f"Tag-cap (80-th percentile) = {tag_cap}")

# ─── 4) PREFILTER WISHLISTS WITH TAG CAP ───────────────────────────
# shuffle users once to avoid bias
users_shuffled = wl_df.user_id.unique()
np.random.shuffle(users_shuffled)

tag_run_count  = defaultdict(int)               # running totals after filtering
kept_records   = []                             # tuples (user_id, book_id)

for u in users_shuffled:
    books = wl_df.loc[wl_df.user_id == u, "book_id"].tolist()
    for b in books:
        tags = [t for t in top_tags.get(b, []) if t]
        if any(tag_run_count[t] >= tag_cap for t in tags):
            # at least one tag is already saturated – skip this book
            continue
        kept_records.append((u, b))
        for t in tags:
            tag_run_count[t] += 1

pref_df = pd.DataFrame(kept_records, columns=["user_id", "book_id"])
print(f"After pre-filter: {len(pref_df)} wishlist entries, "
      f"{pref_df.user_id.nunique()} users")

# ─── 5) FILTER USERS BY MIN_WISH_COUNT ─────────────────────────────
user_sizes = pref_df.groupby("user_id").size()
eligible_users = user_sizes[user_sizes > MIN_WISH_COUNT].index
print(f"Eligible users (> {MIN_WISH_COUNT} items) = {len(eligible_users)}")

# ─── 6) SAMPLE TEST USERS (TEST_FRAC OF ALL USERS) ─────────────────
all_users      = pref_df.user_id.unique()
test_size_goal = int(len(all_users) * TEST_FRAC)
if test_size_goal > len(eligible_users):
    print("⚠️  Not enough eligible users to meet desired test fraction – "
          "using all eligible users as test.")
    test_users = np.array(eligible_users)
else:
    test_users = np.random.choice(eligible_users,
                                  size=test_size_goal,
                                  replace=False)

train_users = np.setdiff1d(all_users, test_users)   # remaining users

train_df = pref_df[pref_df.user_id.isin(train_users)]
test_df  = pref_df[pref_df.user_id.isin(test_users)]

# ─── 7) SAVE SPLITS ────────────────────────────────────────────────
train_df.to_csv(TRAIN_OUT, index=False)
test_df.to_csv(TEST_OUT,  index=False)
print(f"Saved {len(train_df)} train rows → {TRAIN_OUT}")
print(f"Saved {len(test_df)}  test rows  → {TEST_OUT}")

# ─── 8) SUMMARY STATS ──────────────────────────────────────────────
def summary(name, df_):
    sizes = df_.groupby("user_id").size()
    print(f"\n{name} summary")
    print(f"  users          : {df_.user_id.nunique()}")
    print(f"  entries        : {len(df_)}")
    print(f"  avg entries/u  : {sizes.mean():.2f}")
    print(f"  median entries : {sizes.median()}")

summary("TRAIN", train_df)
summary("TEST ", test_df)

# ─── 9) PLOT DISTRIBUTION ──────────────────────────────────────────
plt.figure(figsize=(10,6))
bins = range(1, max(train_df.groupby("user_id").size().max(),
                    test_df.groupby("user_id").size().max()) + 2)
plt.hist(train_df.groupby("user_id").size(), bins=bins,
         alpha=0.6, label="Train")
plt.hist(test_df.groupby("user_id").size(), bins=bins,
         alpha=0.6, label="Test")
plt.xlabel("Wishlist entries per user")
plt.ylabel("Number of users")
plt.title("Wishlist-size distribution (train vs test)")
plt.legend()
plt.xticks(bins)
plt.tight_layout()
plt.savefig(DIST_PLOT_OUT)
plt.show()
print(f"Distribution plot saved → {DIST_PLOT_OUT}")
