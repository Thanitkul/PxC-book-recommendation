#!/usr/bin/env python3
# model_custom_inference.py — Updated for latest preprocessing (2025-04-29)
import os, sys, math
import numpy as np
import pandas as pd
import torch
import random
from typing import List, Tuple, Dict

# ───────── CONFIG ─────────
DATA_ROOT  = "../data-prep-EDA/clean"
CKPT_PATH  = "trained_dlrm_goodreads_features.pt"
DEVICE_ID  = 0
TOP_K      = 10_000
BATCH_SIZE = 8192

# ──────── Helper ─────────────────────────────────────────────
def sys_path_insert_once(p: str):
    if p not in sys.path:
        sys.path.insert(0, p)

def load_artifacts(data_root: str, ckpt_path: str, device: torch.device) -> Dict:
    # --- Load CSVs ---
    csv = lambda name: os.path.join(data_root, name)
    books      = pd.read_csv(csv("books.csv"))
    book_tags  = pd.read_csv(csv("book_tags.csv"))
    tags_df    = pd.read_csv(csv("tags.csv"))
    ratings    = pd.read_csv(csv("ratings.csv"))

    # --- Lookups ---
    author2idx = {a: i+1 for i, a in enumerate(sorted(books.authors.unique()))}
    lang2idx   = {l: i+1 for i, l in enumerate(books.language_code.fillna("unk").unique())}
    tag_id2name = tags_df.set_index("tag_id")["tag_name"].to_dict()

    top_tags = {}
    for bid, grp in book_tags.groupby('book_id'):
        sorted_tags = grp.sort_values('count', ascending=False).tag_id.tolist()
        top_tags[bid] = (sorted_tags + [0]*5)[:5]

    book_author = books.set_index('book_id').authors.map(author2idx).fillna(0).astype(int).to_dict()
    book_lang   = books.set_index('book_id').language_code.fillna('unk').map(lang2idx).fillna(0).astype(int).to_dict()
    book_dense  = books.set_index('book_id')[['ratings_count','average_rating']].astype(float).to_dict('index')
    ratings_map = ratings.groupby('user_id').apply(lambda df: dict(zip(df.book_id, df.rating))).to_dict()
    all_books   = books.book_id.values.astype(np.int64)
    max_rc      = float(books.ratings_count.max() or 1.0)

    sys_path_insert_once("dlrm")
    from dlrm_s_pytorch import DLRM_Net
    ckpt = torch.load(ckpt_path, map_location=device)
    model = DLRM_Net(
        m_spa = ckpt["embed_dim"],
        ln_emb = np.array(ckpt["embedding_sizes"], dtype=np.int64),
        ln_bot = np.array(ckpt["bottom_mlp"], dtype=np.int64),
        ln_top = np.array(ckpt["top_mlp"], dtype=np.int64),
        arch_interaction_op = "dot",
        sigmoid_bot = -1,
        sigmoid_top = len(ckpt["top_mlp"]) - 2,
        ndevices = -1
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return dict(
        model=model, device=device,
        books=books, book_author=book_author, book_lang=book_lang,
        top_tags=top_tags, tag_id2name=tag_id2name,
        book_dense=book_dense, ratings_map=ratings_map,
        all_books=all_books, max_rc=max_rc,
        max_hist_len=ckpt["max_hist_len"],
    )

# ──────── Feature functions ─────────────────────────────────
PAD_BOOK = 0
PAD_SPARSE_VEC7 = [0]*7
PAD_DENSE_VEC3  = [0.0]*3

def get_sparse_vec7(bid: int, art: Dict) -> List[int]:
    if bid == PAD_BOOK:
        return PAD_SPARSE_VEC7
    return [
        art["book_author"].get(bid, 0),
        art["book_lang"].get(bid, 0),
        *art["top_tags"].get(bid, [0]*5)
    ]

def get_dense_vec3(uid: int, bid: int, art: Dict) -> List[float]:
    if bid == PAD_BOOK:
        return PAD_DENSE_VEC3
    dense = art["book_dense"].get(bid, {"ratings_count": 0.0, "average_rating": 0.0})
    ratings_norm = math.log1p(dense["ratings_count"]) / math.log1p(art["max_rc"]) if art["max_rc"] else 0.0
    rating_norm  = (dense["average_rating"] - 1.0) / 4.0
    user_rating  = (art["ratings_map"].get(uid, {}).get(bid, 0.0) - 1.0) / 4.0 if art["ratings_map"].get(uid) else 0.0
    return [
        max(0.0, min(1.0, ratings_norm)),
        max(0.0, min(1.0, rating_norm)),
        max(0.0, min(1.0, user_rating))
    ]

# ──────── Inference ──────────────────────────────────────────
def single_user_inference(
        rated_books: List[Tuple[int, float]],
        wishlist_books: List[int],
        art: Dict):

    model, device = art["model"], art["device"]
    M = art["max_hist_len"]

    rated_ids = [bid for bid, _ in rated_books][:M]
    wish_ids  = wishlist_books[:M]
    rated_ids += [PAD_BOOK]*(M - len(rated_ids))
    wish_ids  += [PAD_BOOK]*(M - len(wish_ids))

    rated_sparse_hist = []
    rated_dense_hist  = []
    for bid in rated_ids:
        rated_sparse_hist.extend(get_sparse_vec7(bid, art))
        rated_dense_hist.extend(get_dense_vec3(0, bid, art))  # user_id=0 for one-shot inference

    wish_sparse_hist = []
    wish_dense_hist  = []
    for bid in wish_ids:
        wish_sparse_hist.extend(get_sparse_vec7(bid, art))
        wish_dense_hist.extend(get_dense_vec3(0, bid, art))

    base_sparse = torch.tensor(rated_sparse_hist + wish_sparse_hist, dtype=torch.long, device=device)
    base_dense  = torch.tensor(rated_dense_hist  + wish_dense_hist,  dtype=torch.float32, device=device)

    NUM_SPARSE = base_sparse.numel() + 7
    NUM_DENSE  = base_dense.numel()  + 3

    top_prob = torch.full((TOP_K,), -1.0, device="cpu")
    top_book = torch.full((TOP_K,), -1, dtype=torch.long, device="cpu")

    books = art["all_books"]
    np.random.shuffle(books)
    for b0 in range(0, len(books), BATCH_SIZE):
        cand = books[b0:b0+BATCH_SIZE]
        C = len(cand)

        cand_sparse = torch.tensor(
            [get_sparse_vec7(int(b), art) for b in cand],
            dtype=torch.long, device=device)
        sparse_all = torch.cat(
            [base_sparse.unsqueeze(0).repeat(C,1), cand_sparse], dim=1)

        lS_i = [sparse_all[:,i] for i in range(NUM_SPARSE)]
        offs = torch.arange(C, dtype=torch.long, device=device)
        lS_o = [offs]*NUM_SPARSE

        cand_dense = torch.tensor(
            [get_dense_vec3(0, int(b), art) for b in cand],
            dtype=torch.float32, device=device)
        dense_all = torch.cat(
            [base_dense.unsqueeze(0).repeat(C,1), cand_dense], dim=1)

        with torch.no_grad():
            prob = model(dense_all, lS_o, lS_i).squeeze(1).cpu()

        merged_p = torch.cat([top_prob, prob])
        merged_idx = torch.cat([top_book, torch.from_numpy(cand)])
        sel = torch.topk(merged_p, TOP_K)
        top_prob, top_book = sel.values, merged_idx[sel.indices]

    # Output
    order = torch.argsort(-top_prob)
    recs  = [(int(b), float(p)) for b, p in zip(top_book[order], top_prob[order])]

    print("\n=== Top-10 Recommendations ===")
    for bid, p in recs[:10]:
        title = art["books"].loc[art["books"].book_id == bid, "title"].iloc[0]
        tags = [art["tag_id2name"].get(t,"") for t in art["top_tags"].get(bid, []) if t][:5]
        print(f"• {title:60s}  (p={p:.4f})  [{', '.join(tags)}]")

    print(f"\n{(top_prob > 0.5).sum().item()} books predicted with >0.5 prob")
    print(f"{(top_prob < 0.5).sum().item()} books predicted with <0.5 prob")

    # Rated and wishlist books
    print("\n=== User Rated Books ===")
    for bid, r in rated_books:
        title = art["books"].loc[art["books"].book_id == bid, "title"].iloc[0]
        tags = [art["tag_id2name"].get(t,"") for t in art["top_tags"].get(bid, []) if t][:5]
        print(f"• {title:60s}  (r={r:.1f})  [{', '.join(tags)}]")

    print("\n=== User Wishlist Books ===")
    for bid in wishlist_books:
        title = art["books"].loc[art["books"].book_id == bid, "title"].iloc[0]
        tags = [art["tag_id2name"].get(t,"") for t in art["top_tags"].get(bid, []) if t][:5]
        print(f"• {title:60s}  [{', '.join(tags)}]")

# ──────── Entrypoint ────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    artifacts = load_artifacts(DATA_ROOT, CKPT_PATH, device)

    # rated_books = [(1, 5.0), (25, 4.5), (9949, 5.0)]   # 🔧 Customize
    # wishlist_books = [5523, 3499]
    rated_books = [(1460, 5.0)]   # 🔧 Customize
    wishlist_books = [1684]

    single_user_inference(rated_books, wishlist_books, artifacts)
