import os
import math
import torch
import numpy as np
from typing import List, Tuple

from torch import nn
from recsys.data.loader import (
    load_static_book_features,
    get_user_seen_books,
    get_user_history
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
CKPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "two_tower_pointwise_bce.pt")
)
PAD = 0
BATCH_SIZE = 8192
TOP_K = 10000

_two_tower_model = None
_static = None

MAX_HIST_LEN = 20
MAX_WISH_LEN = 20

# -----------------------------
# MODEL DEFINITION
# -----------------------------
class TwoTower(nn.Module):
    def __init__(self, num_books, num_authors, num_langs, num_tags, embed_dim, dense_hids, user_hids, max_hist_len, max_wish_len):
        super().__init__()
        self.max_hist_len = max_hist_len
        self.max_wish_len = max_wish_len

        self.book_emb = nn.Embedding(num_books+1, embed_dim, padding_idx=0)
        self.auth_emb = nn.Embedding(num_authors+1, embed_dim, padding_idx=0)
        self.lang_emb = nn.Embedding(num_langs+1, embed_dim, padding_idx=0)
        self.tag_emb  = nn.Embedding(num_tags+1, embed_dim, padding_idx=0)

        dense_layers = []
        prev = 3
        for h in dense_hids:
            dense_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dense_layers += [nn.Linear(prev, embed_dim)]
        self.dense_mlp = nn.Sequential(*dense_layers)

        user_layers = []
        prev = embed_dim
        for h in user_hids:
            user_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        user_layers += [nn.Linear(prev, embed_dim)]
        self.user_mlp = nn.Sequential(*user_layers)

    def forward(self, hist_ids, wish_ids, bid, auth, lang, tags, dense):
        u_h   = self.book_emb(hist_ids).mean(dim=1)
        u_w   = self.book_emb(wish_ids).mean(dim=1)
        u_emb = self.user_mlp(u_h + u_w)

        b_e = self.book_emb(bid)
        a_e = self.auth_emb(auth)
        l_e = self.lang_emb(lang)
        t_e = self.tag_emb(tags).mean(dim=1)
        d_e = self.dense_mlp(dense)
        i_emb = b_e + a_e + l_e + t_e + d_e

        return (u_emb * i_emb).sum(dim=1, keepdim=True)

# -----------------------------
# INITIALIZATION
# -----------------------------
def init_two_tower_model():
    global _two_tower_model, _static

    print("Loading TwoTower model...")
    _static = load_static_book_features()
    print("Static features loaded.")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    meta = ckpt.get("state_dict", ckpt)

    print("Loading static features...")

    # Metadata or fallback values
    num_books   = int(_static["books"]["book_id"].max())
    num_authors = max(_static["author2idx"].values())
    num_langs   = max(_static["lang2idx"].values())
    num_tags = max(_static["tag_id2name"].keys()) + 1


    embed_dim   = ckpt.get("embed_dim", 128)
    dense_hids  = ckpt.get("dense_hids", [64])
    user_hids   = ckpt.get("user_hids", [512, 256, 128, 64])

    print(f"Initializing TwoTower model with: {num_books=}, {num_authors=}, {num_langs=}, {num_tags=}")


    model = TwoTower(
        num_books, num_authors, num_langs, num_tags,
        embed_dim, dense_hids, user_hids,
        MAX_HIST_LEN, MAX_WISH_LEN
    ).to(DEVICE)
    model.load_state_dict(meta if "state_dict" not in ckpt else ckpt["state_dict"])
    model.eval()

    _two_tower_model = model

# -----------------------------
# INFERENCE
# -----------------------------
def recommend_two_tower(user_id: int, top_n: int = 70) -> List[int]:
    assert _two_tower_model and _static, "Model not initialized. Call init_two_tower_model() first."

    rated_books, wishlist_books = get_user_history(user_id)
    seen = get_user_seen_books(user_id)

    hist_ids = [b for b,_ in rated_books][:MAX_HIST_LEN]
    wish_ids = wishlist_books[:MAX_WISH_LEN]
    hist_ids += [PAD] * (MAX_HIST_LEN - len(hist_ids))
    wish_ids += [PAD] * (MAX_WISH_LEN - len(wish_ids))

    def make_dense(bid):
        if bid == PAD:
            return [0.0, 0.0, 0.0]
        dense = _static["book_dense"].get(bid, {"ratings_count": 0, "average_rating": 0})
        r_n = math.log1p(dense["ratings_count"]) / math.log1p(_static["max_rc"])
        a_n = (dense["average_rating"] - 1.0) / 4.0
        u_n = dict(rated_books).get(bid, 0.0)
        u_n = (u_n - 1.0) / 4.0 if u_n else 0.0
        return [r_n, a_n, u_n]

    h_tensor = torch.tensor(hist_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    w_tensor = torch.tensor(wish_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    books = _static["books"]["book_id"].tolist()
    np.random.shuffle(books)

    top_scores = torch.full((TOP_K,), -1e6, device="cpu")
    top_books  = torch.full((TOP_K,), -1, dtype=torch.long, device="cpu")

    for i in range(0, len(books), BATCH_SIZE):
        batch = books[i:i + BATCH_SIZE]
        C = len(batch)

        bid_t  = torch.tensor(batch, device=DEVICE)
        auth_t = torch.tensor([_static["book_author"].get(b, 0) for b in batch], device=DEVICE)
        lang_t = torch.tensor([_static["book_lang"].get(b, 0) for b in batch], device=DEVICE)
        tags_t = torch.tensor([_static["top_tags"].get(b, [0]*5) for b in batch], device=DEVICE)
        dense_t = torch.tensor([make_dense(b) for b in batch], device=DEVICE).view(C, 3)

        with torch.no_grad():
            scores = _two_tower_model(
                h_tensor.repeat(C, 1), w_tensor.repeat(C, 1),
                bid_t, auth_t, lang_t, tags_t, dense_t
            ).squeeze(1).cpu()

        mask = ~torch.tensor([b in seen for b in batch])
        scores[~mask] = -float("inf")

        all_scores = torch.cat([top_scores, scores])
        all_books  = torch.cat([top_books, torch.tensor(batch)])

        vals, idxs = all_scores.topk(TOP_K)
        top_scores, top_books = vals, all_books[idxs]

    return top_books[:top_n].tolist()
