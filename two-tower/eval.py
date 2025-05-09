#!/usr/bin/env python3
"""
two_tower_dcg_eval_with_limited_labels.py

Evaluate your trained two‐tower recommender by average DCG@top_n,
using only K_LABEL items from each user’s test wishlist as labels
and the remaining wishlist items (up to MAX_WISH_LEN) as wishlist-history features.
Shows intermediate metrics every 10 users in the tqdm bar,
including the average rank index of all found labels.
Applies an 85th-percentile tag‐cap prefilter to the test wishlist.
"""
import os
import math
import random
from collections import defaultdict
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────
DATA_ROOT       = "../data-prep-EDA/clean"
CKPT_PATH       = "two_tower_pointwise_bce2.pt"
TRAIN_WISH      = os.path.join(DATA_ROOT, "to_read_train.csv")
TEST_WISHLIST   = os.path.join(DATA_ROOT, "to_read_test.csv")
DEVICE          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TOP_N           = 100        # number of items to score & consider
BATCH_SIZE      = 10000
PAD             = 0

# these must match the values used during training
EMBED_DIM       = 128
DENSE_HIDS      = [64]
USER_HIDS       = [512, 256, 128, 64]
MAX_HIST_LEN    = 20
MAX_WISH_LEN    = 20
K_LABEL         = 8         # number of wishlist items to hold out as labels

# ─── MODEL DEFINITION ───────────────────────────────────────────────
class TwoTower(nn.Module):
    def __init__(
        self,
        num_books:    int,
        num_authors:  int,
        num_langs:    int,
        num_tags:     int,
        embed_dim:    int,
        user_hids:    List[int],
        max_hist_len: int,
        max_wish_len: int
    ):
        super().__init__()
        self.max_hist_len = max_hist_len
        self.max_wish_len = max_wish_len

        # sparse embeddings
        self.book_emb = nn.Embedding(num_books+1, embed_dim, padding_idx=0)
        self.auth_emb = nn.Embedding(num_authors+1, embed_dim, padding_idx=0)
        self.lang_emb = nn.Embedding(num_langs+1, embed_dim, padding_idx=0)
        self.tag_emb  = nn.Embedding(num_tags+1, embed_dim, padding_idx=0)

        # dense MLP for item features
        layers = []
        prev = 3
        for h in DENSE_HIDS:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, embed_dim)]
        self.dense_mlp = nn.Sequential(*layers)

        # user MLP after combining rated-history & wishlist-history
        u_layers = []
        prev = embed_dim
        for h in user_hids:
            u_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        u_layers += [nn.Linear(prev, embed_dim)]
        self.user_mlp = nn.Sequential(*u_layers)

    def forward(
        self,
        hist_ids: torch.LongTensor,   # [B, H]
        wish_ids: torch.LongTensor,   # [B, W]
        bid:      torch.LongTensor,   # [B]
        auth:     torch.LongTensor,   # [B]
        lang:     torch.LongTensor,   # [B]
        tags:     torch.LongTensor,   # [B,5]
        dense:    torch.FloatTensor    # [B,3]
    ) -> torch.Tensor:               # [B,1]
        # embed and pool rated-history
        u_h   = self.book_emb(hist_ids).mean(dim=1)   # [B, E]
        # embed and pool wishlist-history
        u_w   = self.book_emb(wish_ids).mean(dim=1)   # [B, E]
        # combine
        u0    = u_h + u_w                             # [B, E]
        u_emb = self.user_mlp(u0)                     # [B, E]

        # item tower
        b_e   = self.book_emb(bid)                    # [B, E]
        a_e   = self.auth_emb(auth)                   # [B, E]
        l_e   = self.lang_emb(lang)                   # [B, E]
        t_e   = self.tag_emb(tags).mean(dim=1)        # [B, E]
        d_e   = self.dense_mlp(dense)                 # [B, E]
        i_emb = b_e + a_e + l_e + t_e + d_e           # [B, E]

        # dot product → logits
        return (u_emb * i_emb).sum(dim=1, keepdim=True)  # [B,1]


# ─── HELPERS ────────────────────────────────────────────────────────
def load_artifacts(
    data_root: str,
    ckpt_path: str,
    device: torch.device
) -> Dict:
    # 1) Load CSVs
    books       = pd.read_csv(os.path.join(data_root, "books.csv"))
    tags_df     = pd.read_csv(os.path.join(data_root, "tags.csv"))
    book_tags   = pd.read_csv(os.path.join(data_root, "book_tags.csv"))
    ratings     = pd.read_csv(os.path.join(data_root, "ratings.csv"))
    train_wish  = pd.read_csv(TRAIN_WISH)
    test_wish   = pd.read_csv(TEST_WISHLIST)

    # lookups
    author2idx = {a:i+1 for i,a in enumerate(sorted(books.authors.unique()))}
    lang2idx   = {l:i+1 for i,l in enumerate(books.language_code.fillna("unk").unique())}

    book_author = books.set_index("book_id") \
                       .authors.map(author2idx).fillna(0).astype(int).to_dict()
    book_lang   = books.set_index("book_id") \
                       .language_code.fillna("unk").map(lang2idx).fillna(0).astype(int).to_dict()
    book_dense  = books.set_index("book_id")[["ratings_count", "average_rating"]].to_dict("index")

    ratings_map = ratings.groupby("user_id") \
                         .apply(lambda df: list(df.book_id)).to_dict()
    train_map   = train_wish.groupby("user_id").book_id.apply(list).to_dict()
    test_map    = test_wish.groupby("user_id").book_id.apply(list).to_dict()

    all_books = books.book_id.values.astype(np.int64)
    max_rc    = float(books.ratings_count.max() or 1.0)

    # top-5 tags per book
    top_tags = {}
    for bid, grp in book_tags.groupby("book_id"):
        lst = grp.sort_values("count", ascending=False).tag_id.tolist()
        top_tags[bid] = (lst + [0]*5)[:5]

    # ─── PREFILTER ─────────────────────────────────────────────────
    # count tag appearances in train_map
    tag_counts = defaultdict(int)
    for bl in train_map.values():
        for b in bl:
            for t in top_tags.get(b, []):
                if t>0:
                    tag_counts[t] += 1
    # 85th percentile cap
    cap = int(np.percentile(list(tag_counts.values()), 100))
    print(f"⎯ Prefilter tag cap = {cap}")

    def prefilter(wmap):
        seen = defaultdict(int)
        out  = {}
        for u, bl in wmap.items():
            keep = []
            for b in bl:
                ts = [t for t in top_tags.get(b, []) if t>0]
                if any(seen[t] >= cap for t in ts):
                    continue
                keep.append(b)
                for t in ts:
                    seen[t] += 1
            out[u] = keep
        return out

    train_map = prefilter(train_map)
    test_map  = prefilter(test_map)

    # rebuild model
    ckpt = torch.load(ckpt_path, map_location=device)
    num_books    = ckpt.get("num_books", int(books.book_id.max()))
    num_authors  = ckpt.get("num_authors", max(author2idx.values()))
    num_langs    = ckpt.get("num_langs", max(lang2idx.values()))
    num_tags     = ckpt.get("num_tags", int(tags_df.tag_id.max())+1)
    embed_dim    = ckpt.get("embed_dim", EMBED_DIM)
    user_hids    = ckpt.get("user_hids", USER_HIDS)
    max_hist_len = ckpt.get("max_hist_len", MAX_HIST_LEN)
    max_wish_len = ckpt.get("max_wish_len", MAX_WISH_LEN)

    model = TwoTower(
        num_books, num_authors, num_langs, num_tags,
        embed_dim, user_hids, max_hist_len, max_wish_len
    ).to(device)

    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model":         model,
        "device":        device,
        "ratings_map":   ratings_map,
        "train_wish_map":train_map,
        "test_wish_map": test_map,
        "book_author":   book_author,
        "book_lang":     book_lang,
        "book_dense":    book_dense,
        "top_tags":      top_tags,
        "all_books":     all_books,
        "max_rc":        max_rc,
        "max_hist_len":  max_hist_len,
        "max_wish_len":  max_wish_len,
    }


def recommend_top_n(
    uid: int,
    art: Dict,
    top_n: int = TOP_N,
    batch_size: int = BATCH_SIZE
) -> (List[int], List[int]):
    model        = art["model"]
    device       = art["device"]
    ratings_map  = art["ratings_map"]
    test_wish    = art["test_wish_map"].get(uid, [])
    all_books    = art["all_books"]
    max_rc       = art["max_rc"]
    mh, mw       = art["max_hist_len"], art["max_wish_len"]
    book_author  = art["book_author"]
    book_lang    = art["book_lang"]
    top_tags     = art["top_tags"]

    # split labels & history
    labels       = test_wish[:K_LABEL]
    hist_full    = test_wish[K_LABEL:]
    wish_hist    = hist_full[:mw]
    wish_ids     = wish_hist + [PAD]*(mw - len(wish_hist))

    # rated-history
    hist         = ratings_map.get(uid, [])[:mh]
    hist_ids     = hist + [PAD]*(mh - len(hist))

    def make_dense(bs):
        out=[]
        for b in bs:
            if b==PAD:
                out += [0.0,0.0,0.0]
            else:
                rd = art["book_dense"][b]
                out += [
                    math.log1p(rd["ratings_count"])/math.log1p(max_rc),
                    (rd["average_rating"]-1.0)/4.0,
                    1.0 if b in ratings_map.get(uid,[]) else 0.0
                ]
        return out

    h_t = torch.tensor(hist_ids, dtype=torch.long, device=device).unsqueeze(0)
    w_t = torch.tensor(wish_ids, dtype=torch.long, device=device).unsqueeze(0)

    top_scores = torch.full((top_n,), -1e9)
    top_books  = torch.full((top_n,), -1, dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_books), batch_size):
            batch = all_books[i:i+batch_size]
            C     = len(batch)
            bid_t  = torch.tensor(batch, device=device)
            auth_t = torch.tensor([book_author[b] for b in batch], device=device)
            lang_t = torch.tensor([book_lang[b]   for b in batch], device=device)
            tags_t = torch.tensor([top_tags[b]    for b in batch], device=device)
            den_t  = torch.tensor(make_dense(batch), device=device).view(C,3)

            h_b = h_t.repeat(C,1)
            w_b = w_t.repeat(C,1)
            scores = model(h_b, w_b, bid_t, auth_t, lang_t, tags_t, den_t)\
                     .squeeze(1).cpu()

            all_s = torch.cat([top_scores, scores])
            all_b = torch.cat([top_books, torch.tensor(batch)])
            vals, idxs = all_s.topk(top_n)
            top_scores, top_books = vals, all_b[idxs]

    ordered = top_books[torch.argsort(-top_scores)].tolist()
    return ordered, labels


def evaluate_dcg_two_tower(
    art: Dict,
    top_n: int = TOP_N
) -> float:
    total_dcg       = 0.0
    total_hits      = 0
    total_rank_sum  = 0.0
    total_label_hits= 0
    count           = 0
    users           = list(art["test_wish_map"].keys())

    pbar = tqdm(users, desc="Evaluating DCG")
    for uid in pbar:
        rec_ids, labels = recommend_top_n(uid, art, top_n=top_n)
        if not labels:
            continue

        # collect rank indices for all found labels
        for bid in labels:
            if bid in rec_ids:
                idx = rec_ids.index(bid)
                total_rank_sum   += idx
                total_label_hits += 1

        # DCG per user
        dcg = 0.0; hits=0
        for rank, bid in enumerate(rec_ids):
            if bid in labels:
                dcg   += 1.0/math.log2(rank+2)
                hits  += 1

        total_dcg  += dcg
        total_hits += hits
        count      += 1

        if count % 10 == 0:
            avg_ndcg = total_dcg / count
            avg_hits = total_hits / count
            avg_rank = (total_rank_sum/total_label_hits) if total_label_hits else float('nan')
            pbar.set_postfix({
                "avg_nDCG": f"{avg_ndcg:.4f}",
                "avg_hits": f"{avg_hits:.2f}",
                "avg_rank": f"{avg_rank:.1f}"
            })

    return (total_dcg / count) if count>0 else 0.0


if __name__ == "__main__":
    print("Loading model and artifacts…")
    art = load_artifacts(DATA_ROOT, CKPT_PATH, DEVICE)

    print(f"Evaluating DCG@{TOP_N} on test wishlist (K_LABEL={K_LABEL})…")
    avg_dcg = evaluate_dcg_two_tower(art, top_n=TOP_N)
    print(f"▶︎ Average DCG@{TOP_N} (two-tower with {K_LABEL} labels): {avg_dcg:.4f}")
