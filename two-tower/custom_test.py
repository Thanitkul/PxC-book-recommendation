#!/usr/bin/env python3
"""
custom_two_tower_inference.py — Two-tower inference with wishlist-history features

– Loads a two-tower checkpoint (`two_tower_pointwise_bce_prefilter.pt`), whether it’s a bare state_dict
  or a dict with metadata
– Reconstructs the TwoTower model used at training, falling back to sensible defaults
– Uses both rated-history (MAX_HIST_LEN) and wishlist-history (MAX_WISH_LEN) as user features
– Scores every candidate book by dot(u_emb, i_emb)
– Prints the top-10 recommendations, plus for the user’s rated & wishlist books their top-5 tags
"""
import os
import math
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────
DATA_ROOT    = "../data-prep-EDA/clean"
CKPT_PATH    = "two_tower_pointwise_bce2.pt"
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOP_K        = 10_000
BATCH_SIZE   = 8192
PAD          = 0

MAX_HIST_LEN = 20
MAX_WISH_LEN = 20  # wishlist-history length

# defaults if missing from checkpoint
EMBED_DIM    = 128
DENSE_HIDS   = [64]
USER_HIDS    = [512, 256, 128, 64]

# ─── MODEL DEFINITION ────────────────────────────────────────────────
class TwoTower(nn.Module):
    def __init__(
        self,
        num_books:    int,
        num_authors:  int,
        num_langs:    int,
        num_tags:     int,
        embed_dim:    int,
        dense_hids:   List[int],
        user_hids:    List[int],
        max_hist_len: int,
        max_wish_len: int
    ):
        super().__init__()
        self.max_hist_len = max_hist_len
        self.max_wish_len = max_wish_len

        # embeddings
        self.book_emb = nn.Embedding(num_books+1, embed_dim, padding_idx=0)
        self.auth_emb = nn.Embedding(num_authors+1, embed_dim, padding_idx=0)
        self.lang_emb = nn.Embedding(num_langs+1, embed_dim, padding_idx=0)
        self.tag_emb  = nn.Embedding(num_tags+1, embed_dim, padding_idx=0)

        # dense MLP for item features
        layers = []
        prev = 3
        for h in dense_hids:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, embed_dim)]
        self.dense_mlp = nn.Sequential(*layers)

        # user MLP after combining histories
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
        # user tower
        u_h   = self.book_emb(hist_ids).mean(dim=1)   # [B, E]
        u_w   = self.book_emb(wish_ids).mean(dim=1)   # [B, E]
        u_emb = self.user_mlp(u_h + u_w)              # [B, E]

        # item tower
        b_e   = self.book_emb(bid)                    # [B, E]
        a_e   = self.auth_emb(auth)                   # [B, E]
        l_e   = self.lang_emb(lang)                   # [B, E]
        t_e   = self.tag_emb(tags).mean(dim=1)        # [B, E]
        d_e   = self.dense_mlp(dense)                 # [B, E]
        i_emb = b_e + a_e + l_e + t_e + d_e           # [B, E]

        # dot product → logits
        return (u_emb * i_emb).sum(dim=1, keepdim=True)  # [B,1]


# ─── LOAD ARTIFACTS & CHECKPOINT ────────────────────────────────────
def load_artifacts(data_root: str, ckpt_path: str, device: torch.device):
    # 1) Read CSVs
    books     = pd.read_csv(os.path.join(data_root, "books.csv"))
    book_tags = pd.read_csv(os.path.join(data_root, "book_tags.csv"))
    tags_df   = pd.read_csv(os.path.join(data_root, "tags.csv"))
    ratings   = pd.read_csv(os.path.join(data_root, "ratings.csv"))
    test_wish = pd.read_csv(os.path.join(data_root, "to_read_test.csv"))

    # 2) Lookups
    author2idx = {a:i+1 for i,a in enumerate(sorted(books.authors.unique()))}
    lang2idx   = {l:i+1 for i,l in enumerate(books.language_code.fillna("unk").unique())}

    book_author = books.set_index("book_id").authors.map(author2idx).fillna(0).astype(int).to_dict()
    book_lang   = books.set_index("book_id").language_code.fillna("unk").map(lang2idx).fillna(0).astype(int).to_dict()
    book_dense  = books.set_index("book_id")[["ratings_count","average_rating"]].to_dict("index")
    ratings_map = ratings.groupby("user_id").apply(lambda df: dict(zip(df.book_id,df.rating))).to_dict()
    all_books   = books.book_id.values.astype(np.int64)
    max_rc      = float(books.ratings_count.max() or 1.0)

    # top-5 tags per book
    top_tags: Dict[int,List[int]] = {}
    for bid, grp in book_tags.groupby("book_id"):
        lst = grp.sort_values("count", ascending=False).tag_id.tolist()
        top_tags[bid] = (lst + [0]*5)[:5]

    # tag id → name
    tag_id2name = dict(zip(tags_df.tag_id, tags_df.tag_name))

    # 3) Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    # 4) Handle both bare state_dict and metadata dict
    if "state_dict" in ckpt:
        sd           = ckpt["state_dict"]
        num_books    = ckpt.get("num_books",    int(books.book_id.max()))
        num_authors  = ckpt.get("num_authors",  max(author2idx.values()))
        num_langs    = ckpt.get("num_langs",    max(lang2idx.values()))
        num_tags     = ckpt.get("num_tags",     int(tags_df.tag_id.max())+1)
        embed_dim    = ckpt.get("embed_dim",    EMBED_DIM)
        dense_hids   = ckpt.get("dense_hids",   DENSE_HIDS)
        user_hids    = ckpt.get("user_hids",    USER_HIDS)
        max_hist_len = ckpt.get("max_hist_len", MAX_HIST_LEN)
        max_wish_len = ckpt.get("max_wish_len", MAX_WISH_LEN)
    else:
        sd           = ckpt
        num_books    = int(books.book_id.max())
        num_authors  = max(author2idx.values())
        num_langs    = max(lang2idx.values())
        num_tags     = int(tags_df.tag_id.max())+1
        embed_dim    = EMBED_DIM
        dense_hids   = DENSE_HIDS
        user_hids    = USER_HIDS
        max_hist_len = MAX_HIST_LEN
        max_wish_len = MAX_WISH_LEN

    # 5) Build and load model
    model = TwoTower(
        num_books, num_authors, num_langs, num_tags,
        embed_dim, dense_hids, user_hids,
        max_hist_len, max_wish_len
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    return {
        "model":        model,
        "device":       device,
        "books":        books,
        "book_author":  book_author,
        "book_lang":    book_lang,
        "book_dense":   book_dense,
        "top_tags":     top_tags,
        "tag_id2name":  tag_id2name,
        "ratings_map":  ratings_map,
        "all_books":    all_books,
        "max_rc":       max_rc,
        "max_hist_len": max_hist_len,
        "max_wish_len": max_wish_len,
    }


# ─── SINGLE-USER INFERENCE ───────────────────────────────────────────
def single_user_inference(
    uid:       int,
    rated:     List[Tuple[int,float]],
    wishlist:  List[int],
    art:       Dict
):
    model       = art["model"]
    device      = art["device"]
    mh, mw      = art["max_hist_len"], art["max_wish_len"]
    book_dense  = art["book_dense"]
    max_rc      = art["max_rc"]
    ratings_map = art["ratings_map"]
    all_books   = art["all_books"]

    book_author = art["book_author"]
    book_lang   = art["book_lang"]
    top_tags    = art["top_tags"]
    tag_names   = art["tag_id2name"]
    books_df    = art["books"]

    # pad histories
    hist_ids = [b for b,_ in rated][:mh] + [PAD]*(mh-len(rated))
    wish_ids = wishlist[:mw]             + [PAD]*(mw-len(wishlist))

    def make_dense(u, ids):
        out=[]
        for bid in ids:
            if bid==PAD:
                out += [0.0,0.0,0.0]
            else:
                rd = book_dense[bid]
                r_n = math.log1p(rd["ratings_count"])/math.log1p(max_rc)
                a_n = (rd["average_rating"]-1.0)/4.0
                u_n = (ratings_map.get(u,{}).get(bid,0.0)-1.0)/4.0
                out += [r_n, a_n, u_n]
        return out

    h_id_t = torch.tensor(hist_ids, dtype=torch.long, device=device).unsqueeze(0)
    w_id_t = torch.tensor(wish_ids, dtype=torch.long, device=device).unsqueeze(0)

    top_scores = torch.full((TOP_K,), -1e6, device="cpu")
    top_books  = torch.full((TOP_K,), -1,    dtype=torch.int64, device="cpu")

    # score all candidates
    books = all_books.copy()
    np.random.shuffle(books)
    for i in range(0, len(books), BATCH_SIZE):
        batch = books[i:i+BATCH_SIZE]
        C     = len(batch)
        bid_t  = torch.tensor(batch, dtype=torch.long, device=device)
        auth_t = torch.tensor([book_author.get(b,0) for b in batch], device=device)
        lang_t = torch.tensor([book_lang.get(b,0)   for b in batch], device=device)
        tags_t = torch.tensor([top_tags.get(b,[0]*5)for b in batch], device=device)
        den_i  = torch.tensor(make_dense(uid, batch), device=device).view(C,3)

        with torch.no_grad():
            scores = model(
                h_id_t.repeat(C,1),
                w_id_t.repeat(C,1),
                bid_t, auth_t, lang_t, tags_t, den_i
            ).squeeze(1).cpu()

        all_s = torch.cat([top_scores, scores])
        all_b = torch.cat([top_books, torch.tensor(batch)])
        vals, idxs = all_s.topk(TOP_K)
        top_scores, top_books = vals, all_b[idxs]

    # show top-10
    order = torch.argsort(-top_scores)[:100]
    print("\n=== Top-10 Recommendations ===")
    for idx in order:
        b   = int(top_books[idx].item())
        p   = float(top_scores[idx].item())
        title = books_df.loc[books_df.book_id==b, "title"].iloc[0]
        tags  = [tag_names[t] for t in top_tags[b] if t]
        print(f"• {title:60s} (p={p:.4f})  tags={tags}")

    print(f"\n>0.5: {(top_scores>0.5).sum().item()}   <0.5: {(top_scores<=0.5).sum().item()}")

    # show user rated books with their top-5 tags
    print("\n=== User Rated Books ===")
    for b,r in rated:
        title = books_df.loc[books_df.book_id==b, "title"].iloc[0]
        tags  = [tag_names[t] for t in top_tags.get(b,[]) if t]
        tags  = tags[:5]
        print(f"• {title:60s} r={r:.1f}  tags={tags}")

    # show user wishlist books with their top-5 tags
    print("\n=== User Wishlist Books ===")
    for b in wishlist:
        title = books_df.loc[books_df.book_id==b, "title"].iloc[0]
        tags  = [tag_names[t] for t in top_tags.get(b,[]) if t]
        tags  = tags[:5]
        print(f"• {title:60s}  tags={tags}")



# ─── ENTRYPOINT ─────────────────────────────────────────────────────
if __name__ == "__main__":
    art = load_artifacts(DATA_ROOT, CKPT_PATH, DEVICE)

    # Example user (replace with real IDs)
    user_id        = 125
    rated_books    = [(497, 5.0), (720, 4.0), (803, 5.0)]
    wishlist_books = [5523, 3499, 1684]
    # rated_books = [(1, 5.0), (25, 4.5), (9949, 5.0)]   # 🔧 Customize
    # wishlist_books = [28, 17]

    single_user_inference(user_id, rated_books, wishlist_books, art)
