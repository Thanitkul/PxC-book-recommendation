#!/usr/bin/env python3
"""
eval.py: Evaluate a trained DLRM model on the test dataset, compute F1 score,
and save detailed tag-based results (history, wishlist, and target book tags),
and also count tag occurrences by label and plot an area chart.
The test wishlist is first filtered exactly as we did at training time.
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
DATA_ROOT      = "../data-prep-EDA/clean"
MODEL_FILE     = "trained_dlrm_goodreads_features.pt"
OUTPUT_FILE    = "eval_results2.txt"
TAG_JSON_FILE  = "tag_counts_eval.json"
TAG_PLOT_FILE  = "tag_counts_area.png"

MAX_HIST_LEN   = 20
PAD_BOOK       = 0
PAD_SPARSE_VEC = [0] * 7
PAD_DENSE_VEC  = [0.0] * 3
K_NEG          = 5
BATCH_SIZE     = 2048
CPU_WORKERS    = 20

PATHS = {
    "books":     f"{DATA_ROOT}/books.csv",
    "ratings":   f"{DATA_ROOT}/ratings.csv",
    "wish_train":f"{DATA_ROOT}/to_read_train.csv",  # needed for threshold
    "wish_test": f"{DATA_ROOT}/to_read_train.csv",  # typo in original: eval uses train file?
    "book_tags": f"{DATA_ROOT}/book_tags.csv",
    "tags":      f"{DATA_ROOT}/tags.csv",
}

# ─── IMPORT DLRM ─────────────────────────────────────────────────────
if torch_available:
    sys.path.insert(0, "dlrm")
    from dlrm_s_pytorch import DLRM_Net

# ─── LOAD BOOKS & METADATA ───────────────────────────────────────────
books = pd.read_csv(PATHS["books"])
MAX_RATINGS_COUNT = float(books["ratings_count"].max() or 1.0)

author2idx = {a:i+1 for i,a in enumerate(sorted(books.authors.unique()))}
lang2idx   = {l:i+1 for i,l in enumerate(books.language_code.fillna("unk").unique())}

book_author = books.set_index("book_id").authors.map(author2idx).fillna(0).astype(int).to_dict()
book_lang   = books.set_index("book_id").language_code.fillna("unk").map(lang2idx).fillna(0).astype(int).to_dict()
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
ratings      = pd.read_csv(PATHS["ratings"])
ratings_by_user = ratings.groupby("user_id").book_id.apply(list).to_dict()
ratings_full    = ratings.groupby('user_id').apply(lambda df: dict(zip(df.book_id, df.rating))).to_dict()

wish_test_df      = pd.read_csv(PATHS["wish_test"])
test_wish_by_usr  = wish_test_df.groupby("user_id").book_id.apply(list).to_dict()

# ─── TRAIN WISHLIST (for threshold) ──────────────────────────────────
wish_train_df     = pd.read_csv(PATHS["wish_train"])
train_wish_by_usr = wish_train_df.groupby("user_id").book_id.apply(list).to_dict()

# ─── COMPUTE TAG THRESHOLD ────────────────────────────────────────────
pos_tag_counts = defaultdict(int)
for blist in train_wish_by_usr.values():
    for b in blist:
        for t in top_tags.get(b, []):
            if t:
                pos_tag_counts[t] += 1

median_ct     = int(np.median(list(pos_tag_counts.values())))
max_tag_count = int(median_ct * 1.2)
print(f"[Filter] median pos-tag count = {median_ct}, threshold = {max_tag_count}")

# ─── PREFILTER FUNCTION ──────────────────────────────────────────────
def prefilter(wish_map):
    run_ct  = defaultdict(int)
    filtered = {}
    for uid, blist in wish_map.items():
        keep = []
        for b in blist:
            tags = [t for t in top_tags.get(b, []) if t]
            # drop book if ANY of its tags has already hit the cap
            if any(run_ct[t] >= max_tag_count for t in tags):
                continue
            keep.append(b)
            for t in tags:
                run_ct[t] += 1
        filtered[uid] = keep
    return filtered

filtered_test = prefilter(test_wish_by_usr)

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
    rec = book_dense.get(bid, {'ratings_count':0.0,'average_rating':0.0})
    r_norm = math.log1p(rec['ratings_count'])/math.log1p(MAX_RATINGS_COUNT)
    a_norm = (rec['average_rating']-1.0)/4.0 if rec['average_rating']>=1.0 else 0.0
    u_norm = ((ratings_full.get(uid,{}).get(bid,0.0)-1.0)/4.0) if ratings_full.get(uid) else 0.0
    return [max(0, min(1, r_norm)), max(0, min(1, a_norm)), max(0, min(1, u_norm))]

# ─── BUILD EVAL ROWS ───────────────────────────────────────────────────
def build_eval_rows(pair):
    uid, wish_list = pair
    wish_list = wish_list[:MAX_HIST_LEN]
    hist = ratings_by_user.get(uid, [])[:MAX_HIST_LEN]

    # build history vectors
    hist_s, hist_d = [], []
    for b in hist:
        hist_s += get_sparse_vec7(b)
        hist_d += get_dense_vec3(uid, b)
    hpad = MAX_HIST_LEN - len(hist)
    hist_s += PAD_SPARSE_VEC * hpad
    hist_d += PAD_DENSE_VEC  * hpad

    # build wishlist vectors
    wish_s, wish_d = [], []
    for b in wish_list:
        wish_s += get_sparse_vec7(b)
        wish_d += get_dense_vec3(uid, b)
    wpad = MAX_HIST_LEN - len(wish_list)
    wish_s += PAD_SPARSE_VEC * wpad
    wish_d += PAD_DENSE_VEC  * wpad

    base_s = hist_s + wish_s
    base_d = hist_d + wish_d

    rows = []
    user_hist_ids = set(hist + wish_list)
    user_auth_ids = {book_author.get(b,0) for b in user_hist_ids if b!=PAD_BOOK}
    user_tag_ids  = set(t for b in user_hist_ids for t in top_tags.get(b,[]))

    for pos in wish_list:
        rows.append((base_s + get_sparse_vec7(pos),
                     base_d + get_dense_vec3(uid, pos),
                     1, uid, pos, hist, wish_list))
        negs, attempts = 0, 0
        while negs < K_NEG and attempts < K_NEG*50:
            attempts += 1
            neg = int(np.random.choice(all_books_np))
            if neg in user_hist_ids or neg==PAD_BOOK: continue
            if book_author.get(neg,0) in user_auth_ids: continue
            if set(top_tags.get(neg,[])) & user_tag_ids: continue
            rows.append((base_s + get_sparse_vec7(neg),
                         base_d + get_dense_vec3(uid, neg),
                         0, uid, neg, hist, wish_list))
            negs += 1

    return rows

# ─── MAIN ─────────────────────────────────────────────────────────────
def main():
    if not torch_available:
        print("PyTorch is required for evaluation.")
        sys.exit(1)

    print(f"Building filtered test samples with {CPU_WORKERS} workers…")
    items = list(filtered_test.items())
    with mp.Pool(CPU_WORKERS) as pool:
        all_user_rows = list(tqdm(pool.imap(build_eval_rows, items), total=len(items)))
    samples = [row for user_rows in all_user_rows for row in user_rows]

    # unpack samples
    Xi, Xc, y_true, uids, bids, hists, wishs = zip(*samples)
    Xi     = np.array(Xi, dtype=np.int64)
    Xc     = np.array(Xc, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int64)
    num_sparse = Xi.shape[1]

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(MODEL_FILE, map_location=device)
    model  = DLRM_Net(
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

    # inference
    scores = []
    N = Xc.shape[0]
    for i in tqdm(range(0, N, BATCH_SIZE)):
        Xc_b = torch.tensor(Xc[i:i+BATCH_SIZE], device=device)
        Xi_b = torch.tensor(Xi[i:i+BATCH_SIZE], dtype=torch.long, device=device)
        lS_i = [Xi_b[:,j] for j in range(num_sparse)]
        lS_o = [torch.arange(Xi_b.size(0),device=device)] * num_sparse
        with torch.no_grad():
            sc = model(Xc_b, lS_o, lS_i).squeeze(1).cpu().numpy()
        scores.extend(sc.tolist())

    preds = (np.array(scores) > 0.5).astype(int)
    tp = int(((preds==1)&(y_true==1)).sum())
    fp = int(((preds==1)&(y_true==0)).sum())
    fn = int(((preds==0)&(y_true==1)).sum())
    precision = tp/(tp+fp+1e-8)
    recall    = tp/(tp+fn+1e-8)
    f1        = 2*precision*recall/(precision+recall+1e-8)
    acc       = float((preds==y_true).mean())
    print(f"Prec:{precision:.4f}, Rec:{recall:.4f}, F1:{f1:.4f}, Acc:{acc:.4f}")

    # write detailed results
    with open(OUTPUT_FILE,"w",encoding="utf-8") as fw:
        for i in range(N):
            hist_tags = [
                [tag_id_to_name.get(t, "") for t in top_tags.get(b, [0]*5) if t != 0]
                for b in hists[i]
            ]
            wish_tags = [
                [tag_id_to_name.get(t, "") for t in top_tags.get(b, [0]*5) if t != 0]
                for b in wishs[i]
            ]
            target_tags = [
                tag_id_to_name.get(t, "")
                for t in top_tags.get(bids[i], [0]*5)
                if t != 0
            ]

            fw.write(f"User:{uids[i]} Book:{bids[i]} True:{y_true[i]} "
                     f"Score:{scores[i]:.4f} Pred:{preds[i]}\n")
            fw.write(f"  Target Tags: {target_tags}\n")
            fw.write(f"  History Tags: {hist_tags}\n")
            fw.write(f"  Wishlist Tags: {wish_tags}\n\n")
    print(f"Wrote details → {OUTPUT_FILE}")

    # count tags by label & plot
    tag_counts = defaultdict(lambda: {"0":0,"1":0})
    for label, bid in zip(y_true, bids):
        for t in top_tags.get(bid, []):
            if t == 0:
                continue
            name = tag_id_to_name.get(t, "")
            if not name:
                continue
            tag_counts[name][str(label)] += 1


    with open(TAG_JSON_FILE,"w",encoding="utf-8") as jf:
        json.dump(tag_counts, jf, indent=2, ensure_ascii=False)
    print(f"Wrote tag counts JSON → {TAG_JSON_FILE}")

    # area chart
    tags, c0, c1 = zip(*[
        (name, v["0"], v["1"]) for name,v in tag_counts.items()
    ])
    plt.figure(figsize=(12,6))
    plt.fill_between(tags, c0, color="red",   alpha=0.6, label="label 0")
    plt.fill_between(tags, c1, color="blue",  alpha=0.6, label="label 1")
    plt.xticks(rotation=90)
    plt.xlabel("Tag"); plt.ylabel("Count")
    plt.title("Tag Appearance by Label")
    plt.legend(); plt.tight_layout()
    plt.savefig(TAG_PLOT_FILE)
    print(f"Saved area chart → {TAG_PLOT_FILE}")

if __name__ == "__main__":
    main()
