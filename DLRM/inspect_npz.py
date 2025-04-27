#!/usr/bin/env python3
# inspect_goodreads_npz_with_target_book.py
# -------------------------------------------------------
# Inspect examples in goodreads_0_reordered.npz
# Save user profiles and top tags + true item (target) feature
# -------------------------------------------------------

import os
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ─────── CONFIG ───────
DATA_ROOT   = "../data-prep-EDA/clean"
NPZ_PATH    = "data/goodreads_0_reordered.npz"
OUT_DIR     = "inspection_outputs"
MAX_EXAMPLES = 500    # 🛠️ <-- CONTROL HOW MANY EXAMPLES (None = all)

os.makedirs(OUT_DIR, exist_ok=True)

# ─────── Load Data ───────
npz = np.load(NPZ_PATH)
X_int = npz["X_int"]
X_cat = npz["X_cat"]
y     = npz["y"]

# ─────── Load Metadata ───────
books = pd.read_csv(os.path.join(DATA_ROOT, "books.csv"))
book_tags = pd.read_csv(os.path.join(DATA_ROOT, "book_tags.csv"))
tags = pd.read_csv(os.path.join(DATA_ROOT, "tags.csv"))

tag_id2name = tags.set_index("tag_id")["tag_name"].to_dict()

top_tags = {
    bid: (grp.sort_values("count",ascending=False).tag_id.tolist()[:5] + [0]*5)[:5]
    for bid, grp in book_tags.groupby("book_id")
}

# ─────── Helper ───────
def get_book_info(bid):
    info = books[books.book_id==bid]
    if info.empty:
        return {
            "book_id": bid,
            "title": f"Unknown Book ID {bid}",
            "ratings_count": "N/A",
            "average_rating": "N/A",
            "language_code": "N/A",
            "authors": "N/A",
            "tags": []
        }
    rec = info.iloc[0]
    tag_ids = top_tags.get(bid, [])[:5]
    tag_names = [tag_id2name.get(tid, "") for tid in tag_ids if tid]
    return {
        "book_id": int(rec.book_id),
        "title": rec.title,
        "ratings_count": int(rec.ratings_count),
        "average_rating": float(rec.average_rating),
        "language_code": rec.language_code if pd.notna(rec.language_code) else "N/A",
        "authors": rec.authors,
        "tags": tag_names
    }

# ─────── Main Loop ───────
all_user_texts = []
all_tags = []

N = len(X_int) if MAX_EXAMPLES is None else min(MAX_EXAMPLES, len(X_int))

for i in tqdm(range(N), desc="Processing examples"):
    rated_bids = X_int[i, :20]
    wish_bids  = X_int[i, 20:40]
    candidate_bid = int(X_int[i, 40])   # <-- the book model predicts yes/no for

    rated_books = [bid for bid in rated_bids if bid != 0]
    wishlist_books = [bid for bid in wish_bids if bid != 0]

    dense_feat = X_cat[i]  # [ratings_count_norm, average_rating_norm]

    label = int(y[i][0])

    rated_lines = []
    for bid in rated_books:
        info = get_book_info(bid)
        all_tags.extend(info["tags"])
        rated_lines.append(
            f"  - {info['title']} (book_id={info['book_id']})\n"
            f"    Tags: {', '.join(info['tags'])}\n"
            f"    Features: ratings_count={info['ratings_count']}, average_rating={info['average_rating']:.2f}, lang={info['language_code']}, authors={info['authors']}"
        )

    wishlist_lines = []
    for bid in wishlist_books:
        info = get_book_info(bid)
        all_tags.extend(info["tags"])
        wishlist_lines.append(
            f"  - {info['title']} (book_id={info['book_id']})\n"
            f"    Tags: {', '.join(info['tags'])}\n"
            f"    Features: ratings_count={info['ratings_count']}, average_rating={info['average_rating']:.2f}, lang={info['language_code']}, authors={info['authors']}"
        )

    # Favorite Genres
    tag_counter = Counter()
    for bid in rated_books + wishlist_books:
        for tid in top_tags.get(bid, []):
            if tid:
                tag_counter[tid] += 1
    fav_tags = [tag_id2name.get(tid, "") for tid, _ in tag_counter.most_common(5)]

    # Inspect target candidate book
    candidate_info = get_book_info(candidate_bid)
    ratings_count_norm = float(dense_feat[0])
    avg_rating_norm    = float(dense_feat[1])

    # Build output
    user_txt = []
    user_txt.append(f"Example {i+1}: (Label: {label})")
    user_txt.append("\nRated Books:")
    user_txt.extend(rated_lines)
    user_txt.append("\nWishlist Books:")
    user_txt.extend(wishlist_lines)
    user_txt.append("\nFavorite Genres:")
    user_txt.append("  " + ", ".join(fav_tags))

    user_txt.append("\nItem Features (Target Book):")
    user_txt.append(f"  - {candidate_info['title']} (book_id={candidate_info['book_id']})")
    user_txt.append(f"    Tags: {', '.join(candidate_info['tags'])}")
    user_txt.append(f"    Original Features: ratings_count={candidate_info['ratings_count']}, average_rating={candidate_info['average_rating']:.2f}, lang={candidate_info['language_code']}, authors={candidate_info['authors']}")
    user_txt.append(f"    Model Input Dense: ratings_count_norm={ratings_count_norm:.4f}, avg_rating_norm={avg_rating_norm:.4f}")

    user_txt.append("\n" + "="*100 + "\n")

    all_user_texts.append("\n".join(user_txt))

# Save all user profiles
user_profile_txt = os.path.join(OUT_DIR, f"user_profile_with_targetbook_{N}_examples.txt")
with open(user_profile_txt, "w", encoding="utf-8") as f:
    f.writelines(all_user_texts)

print(f"✅ User profiles saved to {user_profile_txt}")

# Top-100 tags summary
overall_tag_counter = Counter(all_tags)
top100_tags = overall_tag_counter.most_common(100)

top_tags_txt = os.path.join(OUT_DIR, f"top_tags_summary_{N}_examples.txt")
with open(top_tags_txt, "w", encoding="utf-8") as f:
    for tag, count in top100_tags:
        f.write(f"{tag}: {count}\n")

print(f"✅ Top tags summary saved to {top_tags_txt}")
