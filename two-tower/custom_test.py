#!/usr/bin/env python3
"""
custom_two_tower_inference.py — Two-tower inference for Goodreads recommender

– Loads a two-tower checkpoint (`two_tower_model.pt`)
– Reconstructs the same TwoTower model used at training (with exact embedding sizes
  and MLP hidden layers)
– Scores every candidate book by dot(u_emb, i_emb)
– Keeps a top-K heap and prints the top-10 recommendations
"""
import os, math, random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────
DATA_ROOT   = "../data-prep-EDA/clean"
CKPT_PATH   = "two_tower_model.pt"
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOP_K       = 10_000
BATCH_SIZE  = 8192
PAD         = 0

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
        max_hist_len: int
    ):
        super().__init__()
        self.max_hist_len = max_hist_len

        # embeddings
        self.book_emb = nn.Embedding(num_books+1, embed_dim, padding_idx=0)
        self.auth_emb = nn.Embedding(num_authors+1, embed_dim, padding_idx=0)
        self.lang_emb = nn.Embedding(num_langs+1, embed_dim, padding_idx=0)
        self.tag_emb  = nn.Embedding(num_tags+1, embed_dim, padding_idx=0)

        # item tower: small MLP over dense features → embed_dim
        layers = []
        prev = 3
        for h in dense_hids:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, embed_dim)]
        self.dense_mlp = nn.Sequential(*layers)

        # user tower: MLP from averaged history embed → embed_dim
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
        bid:      torch.LongTensor,   # [B]
        auth:     torch.LongTensor,   # [B]
        lang:     torch.LongTensor,   # [B]
        tags:     torch.LongTensor,   # [B,5]
        dense:    torch.FloatTensor   # [B,3]
    ) -> torch.Tensor:               # returns [B,1]
        B, H = hist_ids.size()

        # user tower
        h_emb = self.book_emb(hist_ids)     # [B,H,E]
        u0    = h_emb.mean(dim=1)           # [B,E]
        u_emb = self.user_mlp(u0)           # [B,E]

        # item tower
        i_b = self.book_emb(bid)            # [B,E]
        i_a = self.auth_emb(auth)           # [B,E]
        i_l = self.lang_emb(lang)           # [B,E]
        t_e = self.tag_emb(tags)            # [B,5,E]
        i_t = t_e.mean(dim=1)               # [B,E]
        d_e = self.dense_mlp(dense)         # [B,E]
        i_emb = i_b + i_a + i_l + i_t + d_e  # [B,E]

        # dot product
        return (u_emb * i_emb).sum(dim=1, keepdim=True)  # [B,1]


# ─── LOAD ARTIFACTS & CHECKPOINT ────────────────────────────────────
def load_artifacts(data_root: str, ckpt_path: str, device: torch.device):
    # 1) Read CSVs
    books     = pd.read_csv(os.path.join(data_root, "books.csv"))
    book_tags = pd.read_csv(os.path.join(data_root, "book_tags.csv"))
    tags_df   = pd.read_csv(os.path.join(data_root, "tags.csv"))
    ratings   = pd.read_csv(os.path.join(data_root, "ratings.csv"))

    # 2) Lookups
    author2idx = {a:i+1 for i,a in enumerate(sorted(books.authors.unique()))}
    lang2idx   = {l:i+1 for i,l in enumerate(books.language_code.fillna("unk").unique())}

    book_author = books.set_index("book_id").authors.map(author2idx).fillna(0).astype(int).to_dict()
    book_lang   = books.set_index("book_id").language_code.fillna("unk").map(lang2idx).fillna(0).astype(int).to_dict()
    book_dense  = books.set_index("book_id")[["ratings_count","average_rating"]].to_dict("index")
    ratings_map = ratings.groupby("user_id").apply(lambda df: dict(zip(df.book_id,df.rating))).to_dict()
    all_books   = books.book_id.values.astype(np.int64)
    max_rc      = float(books.ratings_count.max() or 1.0)

    top_tags: Dict[int, List[int]] = {}
    for bid, grp in book_tags.groupby("book_id"):
        lst = grp.sort_values("count", ascending=False).tag_id.tolist()
        top_tags[bid] = (lst + [0]*5)[:5]

    # 3) Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    # 4) Extract saved hyper-params
    num_books    = ckpt["num_books"]
    num_authors  = ckpt["num_authors"]
    num_langs    = ckpt["num_langs"]
    num_tags     = ckpt["num_tags"]
    embed_dim    = ckpt["embed_dim"]
    dense_hids   = ckpt["dense_hids"]
    user_hids    = ckpt["user_hids"]
    max_hist_len = ckpt["max_hist_len"]

    # 5) Build model
    model = TwoTower(
        num_books    = num_books,
        num_authors  = num_authors,
        num_langs    = num_langs,
        num_tags     = num_tags,
        embed_dim    = embed_dim,
        dense_hids   = dense_hids,
        user_hids    = user_hids,
        max_hist_len = max_hist_len
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return {
        "model":        model,
        "device":       device,
        "books":        books,
        "book_author":  book_author,
        "book_lang":    book_lang,
        "book_dense":   book_dense,
        "top_tags":     top_tags,
        "tag_id2name":  dict(zip(tags_df.tag_id, tags_df.tag_name)),
        "ratings_map":  ratings_map,
        "all_books":    all_books,
        "max_rc":       max_rc,
        "max_hist_len": max_hist_len,
    }


# ─── SINGLE-USER INFERENCE ───────────────────────────────────────────
def single_user_inference(
    rated:    List[Tuple[int,float]],
    wishlist: List[int],
    art:      Dict
):
    model, device = art["model"], art["device"]
    M             = art["max_hist_len"]

    # pad/truncate history + wishlist
    hist_ids = [b for b,_ in rated][:M] + [PAD]*(M-len(rated))
    wish_ids = wishlist[:M]            + [PAD]*(M-len(wishlist))

    def make_dense(u, ids):
        out = []
        for bid in ids:
            if bid==PAD:
                out += [0.0]*3
            else:
                rec = art["book_dense"][bid]
                r_n = math.log1p(rec["ratings_count"])/math.log1p(art["max_rc"])
                a_n = (rec["average_rating"]-1.0)/4.0
                u_n = (art["ratings_map"].get(u,{}).get(bid,0.0)-1.0)/4.0
                out += [r_n, a_n, u_n]
        return out

    hist_den = make_dense(0, hist_ids)
    wish_den = make_dense(0, wish_ids)

    # broadcast user inputs
    h_id_t  = torch.tensor(hist_ids, dtype=torch.long, device=device).unsqueeze(0)
    w_id_t  = torch.tensor(wish_ids, dtype=torch.long, device=device).unsqueeze(0)
    h_den_t = torch.tensor(hist_den, dtype=torch.float32, device=device).view(1,M,3)
    w_den_t = torch.tensor(wish_den, dtype=torch.float32, device=device).view(1,M,3)

    # CPU-side top-K
    top_scores = torch.full((TOP_K,), -1e6)
    top_books  = torch.full((TOP_K,), -1, dtype=torch.int64)

    # batch over all candidates
    books = art["all_books"].copy()
    np.random.shuffle(books)
    for i in range(0, len(books), BATCH_SIZE):
        batch = books[i:i+BATCH_SIZE]
        C     = len(batch)

        bid_t  = torch.tensor(batch, dtype=torch.long, device=device)
        auth_t = torch.tensor([art["book_author"].get(b,0) for b in batch],
                              dtype=torch.long, device=device)
        lang_t = torch.tensor([art["book_lang"].get(b,0) for b in batch],
                              dtype=torch.long, device=device)
        tags_t = torch.tensor([art["top_tags"].get(b,[0]*5) for b in batch],
                              dtype=torch.long, device=device)
        den_t  = torch.tensor(make_dense(0, batch),
                              dtype=torch.float32, device=device).view(C,3)

        with torch.no_grad():
            scores = model(
                h_id_t.repeat(C,1),
                bid_t, auth_t, lang_t, tags_t,
                den_t
            ).squeeze(1).cpu()  # → [C]

        all_s = torch.cat([top_scores, scores])
        all_b = torch.cat([top_books, bid_t.cpu()])
        vals, idxs = all_s.topk(TOP_K)
        top_scores, top_books = vals, all_b[idxs]

    # print top-10
    order = torch.argsort(-top_scores)[:10]
    print("\n=== Top-10 Recommendations ===")
    for idx in order:
        b = int(top_books[idx].item())
        p = float(top_scores[idx].item())
        title = art["books"].loc[art["books"].book_id==b, "title"].iloc[0]
        tags  = [art["tag_id2name"][t] for t in art["top_tags"][b] if t]
        print(f"• {title:60s} (p={p:.4f}) tags={tags}")

    print(f"\n>0.5: {(top_scores>0.5).sum().item()}   <0.5: {(top_scores<=0.5).sum().item()}")

    print("\n=== User Rated Books ===")
    for b,r in rated:
        t = art["books"].loc[art["books"].book_id==b, "title"].iloc[0]
        print(f"• {t:60s} r={r:.1f}")

    print("\n=== User Wishlist Books ===")
    for b in wishlist:
        t = art["books"].loc[art["books"].book_id==b, "title"].iloc[0]
        print(f"• {t}")


# ─── ENTRYPOINT ─────────────────────────────────────────────────────
if __name__ == "__main__":
    art = load_artifacts(DATA_ROOT, CKPT_PATH, DEVICE)

    # 🔧 Example user (customize as desired):
    rated_books    = [(1460, 5.0), (25, 4.0), (9949, 5.0)]
    wishlist_books = [5523, 3499, 1684]

    single_user_inference(rated_books, wishlist_books, art)
