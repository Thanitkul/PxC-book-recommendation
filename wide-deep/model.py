#!/usr/bin/env python3
"""
wide_and_deep_catalog_train_and_eval.py

– Trains a “wide & deep” model over the full catalog:
    • Deep part pools user rated‐history & wishlist‐history, and item embeddings + MLPs
    • Wide part is just a linear model over the raw 3‐d dense features
– After each epoch, evaluates on a held‐out set of users by ranking the full catalog
  and computing true nDCG@TOP_N over their K_LABEL held‐out wishlist items.
– Saves (overwrites) the model after each epoch + training loss curve + eval nDCG curve.
"""
import os
import math
import random
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # select GPU

# ─── CONFIG ─────────────────────────────────────────────────────────
DATA_ROOT      = "../data-prep-EDA/clean"
MODEL_SAVE     = "wide_and_deep_catalog.pt"
LOSS_CURVE     = "loss_curve.png"
EVAL_CURVE     = "eval_ndcg_curve.png"

MAX_HIST_LEN   = 20      # rated-history length
MAX_WISH_LEN   = 20      # wishlist-history length
K_LABEL        = 5       # positives per user
EMBED_DIM      = 64
DENSE_HID      = 32
USER_HID       = [256, 128, 64]
BATCH_SIZE     = 1       # one user per batch (full-catalog)
EPOCHS         = 10
LR             = 1e-3
TAU            = 4.0     # soft‐rank temperature
SEED           = 42
PAD            = 0

TOP_N          = 100     # nDCG@TOP_N
EVAL_USERS     = 100     # number of test users for eval

# ─── REPRODUCIBILITY ────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─── 1) LOAD METADATA ──────────────────────────────────────────────
books       = pd.read_csv(f"{DATA_ROOT}/books.csv")
MAX_RC      = float(books["ratings_count"].max() or 1.0)

author2idx  = {a:i+1 for i,a in enumerate(sorted(books.authors.unique()))}
lang2idx    = {l:i+1 for i,l in enumerate(books.language_code.fillna("unk").unique())}

book_author = books.set_index("book_id")\
                   .authors.map(author2idx).fillna(0).astype(int).to_dict()
book_lang   = books.set_index("book_id")\
                   .language_code.fillna("unk").map(lang2idx).fillna(0).astype(int).to_dict()
book_dense  = books.set_index("book_id")[["ratings_count","average_rating"]].to_dict("index")
all_books   = books.book_id.values.astype(np.int64)

tags_df     = pd.read_csv(f"{DATA_ROOT}/tags.csv")
NUM_TAGS    = int(tags_df.tag_id.max()) + 1
book_tags   = pd.read_csv(f"{DATA_ROOT}/book_tags.csv")
top_tags    = {}
for bid, grp in book_tags.groupby("book_id"):
    lst = grp.sort_values("count", ascending=False).tag_id.tolist()
    top_tags[bid] = (lst + [0]*5)[:5]

ratings         = pd.read_csv(f"{DATA_ROOT}/ratings.csv")
ratings_by_user = ratings.groupby("user_id").book_id.apply(list).to_dict()

wt               = pd.read_csv(f"{DATA_ROOT}/to_read_train.csv")
wish_train_map   = wt.groupby("user_id").book_id.apply(list).to_dict()
we               = pd.read_csv(f"{DATA_ROOT}/to_read_test.csv")
wish_test_map    = we.groupby("user_id").book_id.apply(list).to_dict()

# pick a fixed sample of test users
all_test_users      = list(wish_test_map.keys())
random.shuffle(all_test_users)
selected_test_users = all_test_users[:EVAL_USERS]


# ─── 2) CATALOG DATASET ────────────────────────────────────────────
class CatalogDataset(Dataset):
    def __init__(self,
                 users: List[int],
                 ratings_map: Dict[int,List[int]],
                 wish_map:    Dict[int,List[int]],
                 all_books:   np.ndarray):
        self.users       = users
        self.ratings_map = ratings_map
        self.wish_map    = wish_map
        self.all_books   = all_books

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        # 1) rated history
        hist = self.ratings_map.get(u, [])[:MAX_HIST_LEN]
        h    = hist + [PAD]*(MAX_HIST_LEN - len(hist))
        # 2) wishlist → split positives vs. history
        full_wish  = self.wish_map.get(u, [])
        pos_labels = full_wish[:K_LABEL]
        wish_hist  = full_wish[K_LABEL:K_LABEL+MAX_WISH_LEN]
        w          = wish_hist + [PAD]*(MAX_WISH_LEN - len(wish_hist))
        # 3) catalog candidates
        cands = list(self.all_books)
        # 4) sparse & dense features per candidate
        auth  = [book_author.get(b,0)   for b in cands]
        lang  = [book_lang.get(b,0)     for b in cands]
        tags  = [top_tags.get(b,[0]*5)  for b in cands]
        dense = []
        for b in cands:
            if b == PAD:
                dense.append([0.0,0.0,0.0])
            else:
                rd = book_dense[b]
                r_n = math.log1p(rd["ratings_count"])/math.log1p(MAX_RC)
                a_n = max(0.0,(rd["average_rating"]-1.0)/4.0)
                u_r = 1.0 if b in self.ratings_map.get(u, []) else 0.0
                dense.append([r_n, a_n, u_r])
        # 5) binary labels for positives
        labels = [1.0 if b in pos_labels else 0.0 for b in cands]

        return (
            torch.tensor(h,     dtype=torch.long),
            torch.tensor(w,     dtype=torch.long),
            torch.tensor(cands, dtype=torch.long),
            torch.tensor(auth,  dtype=torch.long),
            torch.tensor(lang,  dtype=torch.long),
            torch.tensor(tags,  dtype=torch.long),
            torch.tensor(dense, dtype=torch.float32),
            torch.tensor(labels,dtype=torch.float32),
        )

train_ds = CatalogDataset(
    users       = list(wish_train_map.keys()),
    ratings_map = ratings_by_user,
    wish_map    = wish_train_map,
    all_books   = all_books
)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)

eval_ds = CatalogDataset(
    users       = selected_test_users,
    ratings_map = ratings_by_user,
    wish_map    = wish_test_map,
    all_books   = all_books
)
eval_loader = DataLoader(
    eval_ds,
    batch_size=1,
    shuffle=False,
    num_workers=2,
)


# ─── 3) WIDE & DEEP MODEL ───────────────────────────────────────────
class WideAndDeep(nn.Module):
    def __init__(self,
                 num_books, num_authors, num_langs, num_tags,
                 embed_dim, user_hids,
                 max_hist_len, max_wish_len, candidate_count):
        super().__init__()
        self.C            = candidate_count
        # embedding tables
        self.book_emb   = nn.Embedding(num_books+1, embed_dim, padding_idx=0)
        self.auth_emb   = nn.Embedding(num_authors+1, embed_dim, padding_idx=0)
        self.lang_emb   = nn.Embedding(num_langs+1, embed_dim, padding_idx=0)
        self.tag_emb    = nn.Embedding(num_tags+1, embed_dim, padding_idx=0)

        # deep‐MLP on the 3‐d dense features into embed_dim
        self.dense_mlp  = nn.Sequential(
            nn.Linear(3, DENSE_HID), nn.ReLU(inplace=True),
            nn.Linear(DENSE_HID, embed_dim)
        )

        # user MLP after pooling histories
        layers = []
        prev   = embed_dim
        for h in user_hids:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev    = h
        layers += [nn.Linear(prev, embed_dim)]
        self.user_mlp = nn.Sequential(*layers)

        # **wide** linear layer over raw dense features
        self.wide_linear = nn.Linear(3, 1)

    def forward(self, hist, wish, bid, auth, lang, tags, dense):
        # hist: [B,H], wish: [B,W], bid/auth/lang: [B,C], tags: [B,C,5], dense: [B,C,3]
        B = hist.size(0)

        # --- deep user tower ---
        u_h   = self.book_emb(hist).mean(dim=1)    # [B,E]
        u_w   = self.book_emb(wish).mean(dim=1)    # [B,E]
        u_emb = self.user_mlp(u_h + u_w)           # [B,E]

        # --- deep item tower ---
        b_e   = self.book_emb(bid)                 # [B,C,E]
        a_e   = self.auth_emb(auth)                # [B,C,E]
        l_e   = self.lang_emb(lang)                # [B,C,E]
        t_e   = self.tag_emb(tags).mean(dim=2)     # [B,C,E]
        d_e   = self.dense_mlp(dense)              # [B,C,E]
        i_emb = b_e + a_e + l_e + t_e + d_e        # [B,C,E]

        # expand user to match candidates
        u_exp = u_emb.unsqueeze(1).expand(-1, self.C, -1)  # [B,C,E]

        # deep score = dot(u,i)
        deep_scores = (u_exp * i_emb).sum(dim=2)           # [B,C]

        # wide score = linear on raw dense
        # dense is [B,C,3] → reshape to [B*C,3] → linear → [B*C,1] → reshape back
        wide_scores = self.wide_linear(dense.view(-1,3))  # [B*C,1]
        wide_scores = wide_scores.view(B, self.C)         # [B,C]

        # final
        return deep_scores + wide_scores                  # [B,C]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = WideAndDeep(
    num_books       = int(books.book_id.max()),
    num_authors     = max(author2idx.values()),
    num_langs       = max(lang2idx.values()),
    num_tags        = NUM_TAGS,
    embed_dim       = EMBED_DIM,
    user_hids       = USER_HID,
    max_hist_len    = MAX_HIST_LEN,
    max_wish_len    = MAX_WISH_LEN,
    candidate_count = len(all_books),
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# ─── 4) SOFT‐NDCG LOSS ──────────────────────────────────────────────
def soft_ndcg_loss(scores: torch.Tensor, labels: torch.Tensor, tau=TAU) -> torch.Tensor:
    # scores/labels: [B,C]
    diff  = scores.unsqueeze(2) - scores.unsqueeze(1)      # [B,C,C]
    P     = torch.sigmoid(diff / tau)                      # [B,C,C]
    ranks = 1.0 + P.sum(dim=2)                             # [B,C]
    gains = labels / torch.log2(ranks + 1.0)               # [B,C]
    dcg   = gains.sum(dim=1)                               # [B]
    return (1.0 - dcg).mean()


# ideal DCG for normalization
IDCG = sum(1.0 / math.log2(i+2) for i in range(min(K_LABEL, TOP_N)))


# ─── 5) TRAIN + EVAL LOOP ──────────────────────────────────────────
loss_history      = []
eval_ndcg_history = []

for ep in range(1, EPOCHS+1):
    # -- train --
    model.train()
    run_loss = 0.0
    for hist, wish, cands, auth, lang, tags, dense, labels in tqdm(train_loader, desc=f"Train Ep{ep}"):
        # to device
        hist, wish = hist.to(device), wish.to(device)
        cands, auth, lang = cands.to(device), auth.to(device), lang.to(device)
        tags, dense, labels = tags.to(device), dense.to(device), labels.to(device)

        scores = model(hist, wish, cands, auth, lang, tags, dense)  # [B,C]
        loss   = soft_ndcg_loss(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run_loss += loss.item()

    avg_loss = run_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f">>> Epoch {ep} — Soft‐NDCG Loss: {avg_loss:.4f}")

    # save checkpoint (overwrite)
    torch.save({
        "state_dict":   model.state_dict(),
        "num_books":    int(books.book_id.max()),
        "num_authors":  max(author2idx.values()),
        "num_langs":    max(lang2idx.values()),
        "num_tags":     NUM_TAGS,
        "embed_dim":    EMBED_DIM,
        "user_hids":    USER_HID,
        "max_hist_len": MAX_HIST_LEN,
        "max_wish_len": MAX_WISH_LEN,
    }, MODEL_SAVE)
    print(f"✅ Checkpoint saved → {MODEL_SAVE}")

    # -- eval nDCG@TOP_N --
    model.eval()
    ndcg_vals = []
    with torch.no_grad():
        for hist, wish, cands, auth, lang, tags, dense, labels in tqdm(eval_loader, desc=f"Eval Ep{ep}"):
            hist, wish = hist.to(device), wish.to(device)
            cands, auth, lang = cands.to(device), auth.to(device), lang.to(device)
            tags, dense = tags.to(device), dense.to(device)
            lbls = labels.cpu().numpy()[0]

            scores_np = model(hist, wish, cands, auth, lang, tags, dense).cpu().numpy()[0]
            top_idxs  = np.argsort(-scores_np)[:TOP_N]

            dcg = 0.0
            for pos in np.where(lbls==1.0)[0]:
                if pos in top_idxs:
                    rank_pos = int(np.where(top_idxs==pos)[0][0])
                    dcg += 1.0 / math.log2(rank_pos+2)
            ndcg_vals.append(dcg / IDCG)

    avg_ndcg = float(np.mean(ndcg_vals))
    eval_ndcg_history.append(avg_ndcg)
    print(f"▶︎ Epoch {ep} — Eval nDCG@{TOP_N}: {avg_ndcg:.4f}")

# ─── 6) PLOT CURVES ────────────────────────────────────────────────
plt.figure()
plt.plot(range(1, EPOCHS+1), loss_history,      marker='o', label='Train Loss')
plt.xlabel("Epoch"); plt.ylabel("Soft‐NDCG Loss"); plt.legend()
plt.tight_layout(); plt.savefig(LOSS_CURVE)

plt.figure()
plt.plot(range(1, EPOCHS+1), eval_ndcg_history, marker='o', label=f'Eval nDCG@{TOP_N}')
plt.xlabel("Epoch"); plt.ylabel(f"nDCG@{TOP_N}"); plt.legend()
plt.tight_layout(); plt.savefig(EVAL_CURVE)

print("Done.")
