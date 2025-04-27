#!/usr/bin/env python3
# goodreads_dlrm_single_inference.py
# ------------------------------------------------------------
# Inference for a custom user (manual wishlist + rated books)
# ------------------------------------------------------------
import os, sys
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# ─────── CONFIG ───────
DATA_ROOT  = "../data-prep-EDA/clean"
CKPT_PATH  = "trained_dlrm_goodreads.pt"
DEVICE_ID  = 0
TOP_K      = 10000
BATCH_SIZE = 8192

# ─────── Helpers ───────
def sys_path_insert_once(p: str):
    if p not in sys.path:
        sys.path.insert(0, p)

def load_artifacts(data_root: str, ckpt_path: str, device: torch.device) -> Dict:
    csv = lambda name: os.path.join(data_root, name)
    books     = pd.read_csv(csv("books.csv"))
    book_tags = pd.read_csv(csv("book_tags.csv"))
    tags_df   = pd.read_csv(csv("tags.csv"))

    # build mappings
    author2idx = {a: i+1 for i, a in enumerate(sorted(books.authors.unique()))}
    lang2idx   = {l: i+1 for i, l in enumerate(books.language_code.fillna("unk").unique())}

    top_tags = {
        bid: (grp.sort_values("count", ascending=False).tag_id.tolist()[:5] + [0]*5)[:5]
        for bid, grp in book_tags.groupby("book_id")
    }

    tag_id2name = tags_df.set_index("tag_id")["tag_name"].to_dict()

    book_author = books.set_index("book_id").authors.map(author2idx.get) \
                        .fillna(0).astype(int).to_dict()
    book_lang   = books.set_index("book_id").language_code.fillna("unk") \
                        .map(lang2idx.get).fillna(0).astype(int).to_dict()
    book_dense  = books.set_index("book_id")[["ratings_count","average_rating"]] \
                        .astype(float).to_dict("index")

    all_books = books.book_id.values.astype(np.int64)
    max_rc    = float(books.ratings_count.max() or 1)

    # load model
    sys_path_insert_once("dlrm")
    from dlrm_s_pytorch import DLRM_Net

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = DLRM_Net(
        m_spa                = ckpt["embed_dim"],
        ln_emb               = np.asarray(ckpt["embedding_sizes"], dtype=np.int64),
        ln_bot               = np.asarray(ckpt["bottom_mlp"], dtype=np.int64),
        ln_top               = np.asarray(ckpt["top_mlp"], dtype=np.int64),
        arch_interaction_op  = "dot",               # ← use dot
        sigmoid_bot          = -1,
        sigmoid_top          = len(ckpt["top_mlp"]) - 2,
        ndevices             = -1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return dict(
        model=model, device=device,
        books=books,
        tag_id2name=tag_id2name,
        all_books=all_books,
        book_author=book_author,
        book_lang=book_lang,
        book_dense=book_dense,
        top_tags=top_tags,
        max_rc=max_rc,
    )

# ─────── Main inference ───────
def single_user_inference(
    rated_books: List[Tuple[int,float]],
    wishlist_books: List[int],
    artifacts: Dict
):
    model   = artifacts["model"]
    device  = artifacts["device"]
    PAD     = 0

    # pad/truncate to 20
    rated_ids = [bid for bid,_ in rated_books][:20]
    wish_ids  = wishlist_books[:20]

    rated_pad = rated_ids + [PAD]*(20-len(rated_ids))
    wish_pad  = wish_ids  + [PAD]*(20-len(wish_ids))

    # favorite genres from both
    tag_counter = {}
    for bid in rated_ids + wish_ids:
        for tid in artifacts["top_tags"].get(bid, []):
            if tid:
                tag_counter[tid] = tag_counter.get(tid,0) + 1
    fav_tags   = sorted(tag_counter, key=tag_counter.get, reverse=True)[:5]
    fav_genres = [artifacts["tag_id2name"].get(t,"") for t in fav_tags]

    # convert to tensors
    rated_t   = torch.tensor(rated_pad, dtype=torch.long, device=device)
    wish_base = torch.tensor(wish_pad,  dtype=torch.long, device=device)

    # keep top-K across all candidates
    top_prob = torch.full((TOP_K,), -1.0, device="cpu")
    top_book = torch.full((TOP_K,), -1,    dtype=torch.long, device="cpu")

    books = artifacts["all_books"]
    for b_start in range(0, len(books), BATCH_SIZE):
        cand = books[b_start:b_start+BATCH_SIZE]
        C    = len(cand)

        # 1) build sparse-features block: [20 rated | 20 wish | 1 candidate ID | 7 aux]
        rated_block = rated_t.repeat(C,1)           # C×20
        wish_block  = wish_base.repeat(C,1)         # C×20
        cand_id     = torch.tensor(cand, dtype=torch.long, device=device).unsqueeze(1)  # C×1

        # first 40 dims
        sparse40 = torch.cat([rated_block, wish_block], dim=1)           # C×40
        # add the candidate ID
        sparse41 = torch.cat([sparse40, cand_id], dim=1)                 # C×41

        # build the 7-length aux (author, lang, top5 tags)
        aux7 = torch.tensor([
            [
              artifacts["book_author"].get(int(b),0),
              artifacts["book_lang"].get(int(b),0),
              *artifacts["top_tags"].get(int(b), [0]*5)
            ]
            for b in cand
        ], dtype=torch.long, device=device)                                # C×7

        # final sparse48
        sparse48 = torch.cat([sparse41, aux7], dim=1)                    # C×48

        # split into per-feature lists
        lS_i = [sparse48[:, i] for i in range(48)]
        offs = torch.arange(C, dtype=torch.long, device=device)
        lS_o = [offs]*48

        # 2) build dense side (2 dims)
        recs = artifacts["book_dense"]
        dense_np = np.zeros((C,2), dtype=np.float32)
        dense_np[:,0] = np.clip(
            np.log1p([recs[int(b)]["ratings_count"] for b in cand]) /
            np.log1p(artifacts["max_rc"]), 0, 1
        )
        dense_np[:,1] = [(recs[int(b)]["average_rating"] - 1.0)/4.0 for b in cand]
        dense2 = torch.tensor(dense_np, dtype=torch.float32, device=device)

        # 3) forward & keep top-K
        with torch.no_grad():
            prob = model(dense2, lS_o, lS_i).squeeze(1).cpu()

        merged_p   = torch.cat([top_prob, prob])
        merged_idx = torch.cat([top_book, torch.from_numpy(cand)])
        sel        = torch.topk(merged_p, TOP_K)
        top_prob, top_book = sel.values, merged_idx[sel.indices]

    # ── Print results ────────────────────────────────────────────────
    print("\n📚 Custom user profile:\n")
    print("Rated Books:")
    for bid, rating in rated_books:
        title = artifacts["books"].loc[artifacts["books"].book_id==bid, "title"].iloc[0]
        tags  = [artifacts["tag_id2name"].get(t,"") for t in artifacts["top_tags"].get(bid,[]) if t][:5]
        print(f"  - {title} [Rating: {rating:.1f}] | Tags: {', '.join(tags)}")

    print("\nWishlist Books:")
    for bid in wishlist_books:
        title = artifacts["books"].loc[artifacts["books"].book_id==bid, "title"].iloc[0]
        tags  = [artifacts["tag_id2name"].get(t,"") for t in artifacts["top_tags"].get(bid,[]) if t][:5]
        print(f"  - {title} | Tags: {', '.join(tags)}")

    print("\nFavorite Genres:", ", ".join(fav_genres))

    with open("recommendations_and_distribution.txt", "w") as f:
        f.write("\nTop-10 Recommendations:\n")
        order = torch.argsort(-top_prob)
        for b, p in zip(top_book[order].tolist(), top_prob[order].tolist()):
            rec_title = artifacts["books"].loc[artifacts["books"].book_id==b, "title"].iloc[0]
            rec_tags  = [artifacts["tag_id2name"].get(t,"") for t in artifacts["top_tags"].get(b,[]) if t][:5]
            f.write(f"  - {rec_title} (prob={p:.4f}) | Tags: {', '.join(rec_tags)}\n")
        
        f.write("\nPrediction distribution across all candidates:\n")
        all_preds_above_05 = (top_prob > 0.5).sum().item()
        all_preds_below_05 = (top_prob <= 0.5).sum().item()
        f.write(f"  Predictions > 0.5: {all_preds_above_05}\n")
        f.write(f"  Predictions ≤ 0.5: {all_preds_below_05}\n")

if __name__ == "__main__":
    device    = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    artifacts = load_artifacts(DATA_ROOT, CKPT_PATH, device)

    # 🛠️ ← your custom input
    # rated_books   = [(8934, 5.0), (546, 4.5), (5120, 5.0)]
    # wishlist_books = [8688, 4908]
    rated_books   = [(18, 5.0), (21, 4.5), (23, 5.0)]
    wishlist_books = [24, 25]

    single_user_inference(rated_books, wishlist_books, artifacts)
