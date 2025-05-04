#!/usr/bin/env python3
"""
tag_distribution.py

Reads:
  - One or more wishlist CSVs (user_id, book_id)
  - A book-tags CSV (book_id, tag_id, count)

Concatenates all wishlists, builds each book’s top-5 tags, counts how often
each tag appears across all wish-lists, computes basic statistics (min, max,
avg, std), and saves a bar chart of tag IDs vs. their counts with stats printed
and a horizontal line at the mean.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ─── CONFIG ─────────────────────────────────────────────────────────
# List of paths to wishlist CSVs (each must have columns: user_id, book_id)
WISHLIST_CSVS  = [
    # "clean/to_read_train.csv",
    # "clean/to_read_test.csv",
    "clean/to_read.csv",
    # add more paths as needed...
]
# Path to the book-tags CSV (must have columns: book_id, tag_id, count)
BOOK_TAGS_CSV  = "clean/book_tags.csv"
# Where to save the output histogram
OUTPUT_PATH    = "clean/tag_count_distribution.png"

def main():
    # 1) Load and concatenate all wishlist CSVs
    dfs = []
    for path in WISHLIST_CSVS:
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {path}")
    wl = pd.concat(dfs, ignore_index=True)
    print(f"Total wishlist entries combined: {len(wl)} rows")

    # 2) Load book-tags and build top-5 tags per book
    btag = pd.read_csv(BOOK_TAGS_CSV)
    top5 = {}
    for bid, grp in btag.groupby("book_id"):
        tags = grp.sort_values("count", ascending=False).tag_id.tolist()
        top5[bid] = tags[:5]

    # 3) Count tag occurrences across all wish-lists
    tag_counts = defaultdict(int)
    for book in wl["book_id"]:
        for t in top5.get(book, []):
            if t > 0:
                tag_counts[t] += 1

    if not tag_counts:
        print("⚠️  No tags found in wishlists; nothing to plot.")
        return

    # 4) Prepare data for bar chart
    tag_ids = sorted(tag_counts.keys())
    counts  = np.array([tag_counts[t] for t in tag_ids], dtype=int)

    # 5) Compute statistics
    cnt_min = counts.min()
    cnt_max = counts.max()
    cnt_avg = counts.mean()
    cnt_std = counts.std(ddof=0)
    print("Tag count statistics:")
    print(f"  Min count: {cnt_min}")
    print(f"  Max count: {cnt_max}")
    print(f"  Avg count: {cnt_avg:.2f}")
    print(f"  Std dev : {cnt_std:.2f}")

    # 6) Plot and save
    plt.figure(figsize=(12, 6))
    plt.bar(tag_ids, counts, width=20.0)
    plt.axhline(cnt_avg, color='red', linestyle='--', label=f"Mean = {cnt_avg:.1f}")
    plt.xlabel("Tag ID")
    plt.ylabel("Count in wishlists")
    plt.title("Frequency of Each Tag ID in Wishlists")
    plt.legend()

    stats_text = (
        f"min: {cnt_min}\n"
        f"max: {cnt_max}\n"
        f"avg: {cnt_avg:.1f}\n"
        f"std: {cnt_std:.1f}"
    )
    plt.text(
        0.95, 0.95, stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"✅ Chart saved → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
