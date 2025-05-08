#!/usr/bin/env python3
"""
split_to_read_balanced.py      (ᓚᘏᗢ)

•  Book-aware pre-filter: cap how often every **book_id** may appear
   at the P_BOOK-th percentile of its original wishlist frequency.
•  Users with ≤ MIN_WISH_COUNT remaining items are dropped.
•  A fixed TEST_FRAC of all remaining users is sampled for the test split.
•  Prints detailed stats and saves a before/after book-count plot.
"""

import os
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── CONFIG ─────────────────────────────────────────────────────────
DATA_DIR        = "clean"
WISHLIST_CSV    = os.path.join(DATA_DIR, "to_read.csv")   # user_id, book_id

TRAIN_OUT       = os.path.join(DATA_DIR, "to_read_train.csv")
TEST_OUT        = os.path.join(DATA_DIR, "to_read_test.csv")
DIST_PLOT_OUT   = os.path.join(DATA_DIR, "wishlist_distribution.png")
BOOK_PLOT_OUT   = os.path.join(DATA_DIR, "book_count_before_after.png")

P_BOOK          = 80        # percentile for book-cap
TEST_FRAC       = 0.10      # wanted fraction of all users → test
MIN_WISH_COUNT  = 11        # user needs > this after filtering

RNG_SEED        = 42
np.random.seed(RNG_SEED)

# ─── 1) LOAD DATA ──────────────────────────────────────────────────
wl_df = pd.read_csv(WISHLIST_CSV)               # columns: user_id, book_id

# ─── 2) GLOBAL BOOK STATISTICS & CAP ───────────────────────────────
book_counts_all = wl_df.book_id.value_counts().sort_index()
book_cap = int(np.percentile(book_counts_all.values, P_BOOK))
print(f"Book-cap ({P_BOOK}-th pct): {book_cap:>6} "
      f"(min={book_counts_all.min()}, max={book_counts_all.max()})")

# ─── 3) PREFILTER WISHLISTS WITH BOOK CAP ─────────────────────────
users_shuffled  = wl_df.user_id.unique()
np.random.shuffle(users_shuffled)

book_run_count: Dict[int, int] = defaultdict(int)
kept_records = []                                 # (user_id, book_id)

for u in users_shuffled:
    for b in wl_df.loc[wl_df.user_id == u, "book_id"]:
        if book_run_count[b] >= book_cap:
            continue                              # saturated book
        kept_records.append((u, b))
        book_run_count[b] += 1

pref_df = pd.DataFrame(kept_records,
                       columns=["user_id", "book_id"])
print(f"After pre-filter : {len(pref_df):>7} wishlist rows, "
      f"{pref_df.user_id.nunique():>5} users")

# Book stats *after* cap
book_counts_after = pref_df.book_id.value_counts().sort_index()
print(f"Book counts after: min={book_counts_after.min()}, "
      f"max={book_counts_after.max()}")

# ─── 4) FILTER USERS BY MIN_WISH_COUNT ─────────────────────────────
user_sizes      = pref_df.groupby("user_id").size()
eligible_users  = user_sizes[user_sizes > MIN_WISH_COUNT].index
print(f"Eligible users (> {MIN_WISH_COUNT} items) = {len(eligible_users)}")

# ─── 5) SAMPLE TEST USERS (TEST_FRAC OF ALL USERS) ─────────────────
all_users       = pref_df.user_id.unique()
test_size_goal  = int(len(all_users) * TEST_FRAC)

if test_size_goal > len(eligible_users):
    print("⚠️  Not enough eligible users – using all eligible users for test")
    test_users = np.array(eligible_users)
else:
    test_users = np.random.choice(eligible_users,
                                  size=test_size_goal,
                                  replace=False)

train_users = np.setdiff1d(all_users, test_users)
train_df    = pref_df[pref_df.user_id.isin(train_users)]
test_df     = pref_df[pref_df.user_id.isin(test_users)]

# ─── 6) SAVE SPLITS ────────────────────────────────────────────────
train_df.to_csv(TRAIN_OUT, index=False)
test_df.to_csv(TEST_OUT,  index=False)
print(f"Saved {len(train_df):>7} train rows → {TRAIN_OUT}")
print(f"Saved {len(test_df):>7}  test rows → {TEST_OUT}")

# ─── 7) SUMMARY STATS ──────────────────────────────────────────────
def summary(name: str, df_: pd.DataFrame) -> None:
    sizes = df_.groupby("user_id").size()
    print(f"\n{name} summary")
    print(f"  users          : {df_.user_id.nunique()}")
    print(f"  entries        : {len(df_)}")
    print(f"  avg entries/u  : {sizes.mean():.2f}")
    print(f"  median entries : {sizes.median()}")

summary("TRAIN", train_df)
summary("TEST ", test_df)

# ─── 8) PLOT USER-SIZE DISTRIBUTION (TRAIN vs TEST) ────────────────
plt.figure(figsize=(10, 6))
bins = range(1,
             max(train_df.groupby("user_id").size().max(),
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
plt.close()
print(f"User-size distribution plot saved → {DIST_PLOT_OUT}")

# ─── 9) PLOT BOOK-COUNT BEFORE vs AFTER ────────────────────────────
plt.figure(figsize=(12, 6))
book_ids_sorted = book_counts_all.index
plt.bar(book_ids_sorted, book_counts_all.values,
        alpha=0.5, label="Before cap")
plt.bar(book_ids_sorted,
        book_counts_after.reindex(book_ids_sorted).fillna(0).values,
        alpha=0.5, label="After cap", color="red")
plt.xlabel("Book ID")
plt.ylabel("Times wish-listed")
plt.title("Book-frequency distribution (before vs after cap)")
plt.legend()
plt.tight_layout()
plt.savefig(BOOK_PLOT_OUT)
plt.close()
print(f"Book-count plot saved          → {BOOK_PLOT_OUT}")
