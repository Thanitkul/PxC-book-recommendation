#!/usr/bin/env python3
"""
eval.py: Evaluate a trained DLRM model on the test dataset, compute F1 score,
and save detailed tag-based results (history, wishlist, and target book tags),
and also count tag occurrences by label and plot an area chart.
"""
import os
import sys
import math
import json
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import defaultdict

torch_available = True
try:
    import torch
    from torch import nn
    from tqdm import tqdm
    import matplotlib.pyplot as plt
except ImportError:
    torch_available = False

# ─── CONFIG ─────────────────────────────────────────────────────────
DATA_ROOT     = "../data-prep-EDA/clean"
MODEL_FILE    = "trained_dlrm_goodreads_features.pt"
OUTPUT_FILE   = "eval_results2.txt"
TAG_JSON_FILE = "tag_counts_train.json"
TAG_PLOT_FILE = "tag_counts_area.png"

MAX_HIST_LEN  = 20
PAD_BOOK      = 0
PAD_SPARSE_VEC = [0] * 7
PAD_DENSE_VEC  = [0.0] * 3
K_NEG         = 10
BATCH_SIZE    = 2048
CPU_WORKERS   = 20

PATHS = {
    "books":     f"{DATA_ROOT}/books.csv",
    "ratings":   f"{DATA_ROOT}/ratings.csv",
    "wish_test": f"{DATA_ROOT}/to_read_train.csv",
    "book_tags": f"{DATA_ROOT}/book_tags.csv",
    "tags":      f"{DATA_ROOT}/tags.csv",
}

# ─── IMPORT DLRM ──────────────────────────────────────────────────────
if torch_available:
    sys.path.insert(0, "dlrm")
    from dlrm_s_pytorch import DLRM_Net

# ─── LOAD BOOKS & METADATA ────────────────────────────────────────────
books = pd.read_csv(PATHS["books"])
MAX_RATINGS_COUNT = float(books["ratings_count"].max() or 1.0)

author2idx = {a: i+1 for i, a in enumerate(sorted(books.authors.unique()))}
lang2idx   = {l: i+1 for i, l in enumerate(books.language_code.fillna("unk").unique())}

book_author = books.set_index("book_id").authors.map(author2idx).fillna(0).astype(int).to_dict()
book_lang   = books.set_index("book_id").language_code.fillna('unk').map(lang2idx).fillna(0).astype(int).to_dict()
book_dense  = books.set_index("book_id")[['ratings_count','average_rating']].astype(float).to_dict('index')
all_books_np = books.book_id.values.astype(np.int32)

# ─── TAGS ─────────────────────────────────────────────────────────────
tags_df = pd.read_csv(PATHS["tags"])
tag_id_to_name = dict(zip(tags_df.tag_id, tags_df.tag_name))
book_tags_df = pd.read_csv(PATHS["book_tags"])
top_tags = {}
for bid, grp in book_tags_df.groupby("book_id"):
    sorted_tags = grp.sort_values("count", ascending=False).tag_id.tolist()
    top_tags[bid] = (sorted_tags + [0]*5)[:5]

# ─── USER DATA ────────────────────────────────────────────────────────
ratings = pd.read_csv(PATHS["ratings"])
ratings_by_user = ratings.groupby("user_id").book_id.apply(list).to_dict()
ratings_full    = ratings.groupby('user_id').apply(lambda df: dict(zip(df.book_id, df.rating))).to_dict()

wish_test = pd.read_csv(PATHS["wish_test"])
test_wish_by_usr = wish_test.groupby("user_id").book_id.apply(list).to_dict()

# ─── FEATURE FUNCTIONS ────────────────────────────────────────────────
def get_sparse_vec7(bid: int):
    if bid == PAD_BOOK:
        return PAD_SPARSE_VEC
    return [
        book_author.get(bid, 0),
        book_lang.get(bid, 0),
        *top_tags.get(bid, [0]*5)
    ]

def get_dense_vec3(uid: int, bid: int):
    if bid == PAD_BOOK:
        return PAD_DENSE_VEC
    rec = book_dense.get(bid, {'ratings_count': 0.0, 'average_rating': 0.0})
    r_norm = math.log1p(rec['ratings_count']) / math.log1p(MAX_RATINGS_COUNT)
    a_norm = (rec['average_rating'] - 1.0) / 4.0 if rec['average_rating'] >= 1.0 else 0.0
    u_rating = (ratings_full.get(uid, {}).get(bid, 0.0) - 1.0) / 4.0 if ratings_full.get(uid) else 0.0
    return [
        max(0.0, min(1.0, r_norm)),
        max(0.0, min(1.0, a_norm)),
        max(0.0, min(1.0, u_rating))
    ]

# ─── BUILD EVAL ROWS ───────────────────────────────────────────────────
def build_eval_rows(pair):
    uid, wish_list = pair
    history = ratings_by_user.get(uid, [])[:MAX_HIST_LEN]
    wish    = wish_list[:MAX_HIST_LEN]

    hist_s, hist_d = [], []
    for b in history:
        hist_s += get_sparse_vec7(b)
        hist_d += get_dense_vec3(uid, b)
    hpad = MAX_HIST_LEN - len(history)
    hist_s += PAD_SPARSE_VEC * hpad
    hist_d += PAD_DENSE_VEC  * hpad

    wish_s, wish_d = [], []
    for b in wish:
        wish_s += get_sparse_vec7(b)
        wish_d += get_dense_vec3(uid, b)
    wpad = MAX_HIST_LEN - len(wish)
    wish_s += PAD_SPARSE_VEC * wpad
    wish_d += PAD_DENSE_VEC  * wpad

    base_s = hist_s + wish_s
    base_d = hist_d + wish_d

    user_hist_ids = set(history + wish)
    user_auth_ids = {book_author.get(b,0) for b in user_hist_ids if b != PAD_BOOK}
    user_tag_ids  = set(t for b in user_hist_ids for t in top_tags.get(b, []))

    rows = []
    for pos in wish:
        rows.append((base_s + get_sparse_vec7(pos),
                     base_d + get_dense_vec3(uid, pos),
                     1, uid, pos, history, wish))
        negs, attempts = 0, 0
        while negs < K_NEG and attempts < K_NEG*50:
            attempts += 1
            neg = int(np.random.choice(all_books_np))
            if neg in user_hist_ids or neg == PAD_BOOK:
                continue
            if book_author.get(neg,0) in user_auth_ids:
                continue
            if set(top_tags.get(neg,[])) & user_tag_ids:
                continue
            rows.append((base_s + get_sparse_vec7(neg),
                         base_d + get_dense_vec3(uid, neg),
                         0, uid, neg, history, wish))
            negs += 1
    return rows

# ─── MAIN ─────────────────────────────────────────────────────────────
def main():
    if not torch_available:
        print("PyTorch is required for evaluation.")
        sys.exit(1)

    # --- Control parameters ---
    NUM_EVAL_USERS   = None    # limit number of users (set to None for all)
    NUM_EVAL_SAMPLES = None  # limit number of samples (set to None for all)
    RANDOM_SAMPLE    = False # if True, randomly select users

    print(f"Building test samples with {CPU_WORKERS} workers...")
    user_items = list(test_wish_by_usr.items())

    if NUM_EVAL_USERS is not None:
        if RANDOM_SAMPLE:
            print(f"Randomly selecting {NUM_EVAL_USERS} users...")
            user_items = random.sample(user_items, min(NUM_EVAL_USERS, len(user_items)))
        else:
            print(f"Selecting first {NUM_EVAL_USERS} users...")
            user_items = user_items[:NUM_EVAL_USERS]

    with mp.Pool(CPU_WORKERS) as pool:
        results = list(tqdm(pool.imap(build_eval_rows, user_items), total=len(user_items)))

    samples = [row for user_rows in results for row in user_rows]

    if NUM_EVAL_SAMPLES is not None:
        print(f"Limiting to {NUM_EVAL_SAMPLES} samples...")
        samples = samples[:NUM_EVAL_SAMPLES]

    # --- Prepare arrays ---
    Xi     = np.array([r[0] for r in samples], dtype=np.int64)
    Xc     = np.array([r[1] for r in samples], dtype=np.float32)
    y_true = np.array([r[2] for r in samples], dtype=np.int64)
    uids   = [r[3] for r in samples]
    bids   = [r[4] for r in samples]
    hists  = [r[5] for r in samples]
    wishs  = [r[6] for r in samples]
    num_sparse = Xi.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {MODEL_FILE} on {device}...")
    ckpt = torch.load(MODEL_FILE, map_location=device)
    model = DLRM_Net(
        m_spa               = ckpt["embed_dim"],
        ln_emb              = np.array(ckpt["embedding_sizes"]),
        ln_bot              = np.array(ckpt["bottom_mlp"]),
        ln_top              = np.array(ckpt["top_mlp"]),
        arch_interaction_op = "dot",
        sigmoid_bot         = -1,
        sigmoid_top         = len(ckpt["top_mlp"])-2,
        loss_function       = "bce",
        ndevices            = 1 if device.type=='cuda' else -1
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print("Running inference...")
    all_scores = []
    N = Xc.shape[0]
    for i in tqdm(range(0, N, BATCH_SIZE)):
        Xc_b = torch.tensor(Xc[i:i+BATCH_SIZE], device=device)
        Xi_b = torch.tensor(Xi[i:i+BATCH_SIZE], dtype=torch.long, device=device)
        lS_i = [Xi_b[:, j] for j in range(num_sparse)]
        lS_o = [torch.arange(Xi_b.size(0), device=device)] * num_sparse
        with torch.no_grad():
            sc = model(Xc_b, lS_o, lS_i).squeeze(1).cpu().numpy()
        all_scores.extend(sc.tolist())

    preds = (np.array(all_scores) > 0.5).astype(int)
    tp = int(((preds==1) & (y_true==1)).sum())
    fp = int(((preds==1) & (y_true==0)).sum())
    fn = int(((preds==0) & (y_true==1)).sum())
    precision = tp / (tp+fp+1e-8)
    recall    = tp / (tp+fn+1e-8)
    f1        = 2 * precision * recall / (precision+recall+1e-8)
    accuracy  = float((preds == y_true).mean())

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Acc: {accuracy:.4f}")

    # ─── WRITE DETAILED RESULTS ────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fw:
        for i in tqdm(range(N), desc="Writing detailed results"):
            hist_tags   = [[tag_id_to_name.get(t,"") for t in top_tags.get(b,[0]*5)] for b in hists[i]]
            wish_tags   = [[tag_id_to_name.get(t,"") for t in top_tags.get(b,[0]*5)] for b in wishs[i]]
            target_tags = [tag_id_to_name.get(t,"") for t in top_tags.get(bids[i],[0]*5)]
            fw.write(f"User:{uids[i]} Book:{bids[i]} True:{y_true[i]} "
                     f"Score:{all_scores[i]:.4f} Pred:{preds[i]}\n")
            fw.write(f"  Target Tags: {target_tags}\n")
            fw.write(f"  History Tags: {hist_tags}\n")
            fw.write(f"  Wishlist Tags: {wish_tags}\n\n")
    print(f"Detailed results written to {OUTPUT_FILE}")

    # ─── TAG COUNTS & PLOT ────────────────────────────────────────────
    tag_counts = defaultdict(lambda: {"0": 0, "1": 0})
    for _, _, label, _, bid, _, _ in samples:
        for tag_id in top_tags.get(bid, []):
            if tag_id == 0:
                continue
            tag_name = tag_id_to_name.get(tag_id, "")
            tag_counts[tag_name][str(label)] += 1

    # Write JSON
    with open(TAG_JSON_FILE, "w", encoding="utf-8") as jf:
        json.dump(tag_counts, jf, indent=2, ensure_ascii=False)
    print(f"Wrote tag counts to {TAG_JSON_FILE}")

    # Plot area chart
    tags = list(tag_counts.keys())
    counts0 = [tag_counts[t]["0"] for t in tags]
    counts1 = [tag_counts[t]["1"] for t in tags]

    plt.figure(figsize=(12, 6))
    plt.fill_between(tags, counts0, color="red", alpha=0.6, label="label 0")
    plt.fill_between(tags, counts1, color="blue", alpha=0.6, label="label 1")
    plt.xticks(rotation=90)
    plt.xlabel("Tag")
    plt.ylabel("Count")
    plt.title("Tag Appearance by Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(TAG_PLOT_FILE)
    print(f"Saved area chart to {TAG_PLOT_FILE}")


if __name__ == "__main__":
    main()
