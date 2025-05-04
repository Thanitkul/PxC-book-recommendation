#!/usr/bin/env python3
"""
two_tower_pointwise_train_and_eval_bce_with_prefilter.py

– Prefilters wishlists so no tag appears more than the 85th‐percentile number of times.
– Builds pointwise samples with an adjustable POS:NEG ratio.
– Trains a two‐tower model with BCEWithLogitsLoss.
– After each epoch, evaluates on a held-out set by ranking the full catalog and computing true nDCG@TOP_N.
– Saves (overwrites) the model after each epoch + training loss curve + eval nDCG curve.
"""
import os
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─── CONFIG ─────────────────────────────────────────────────────────
DATA_ROOT      = "../data-prep-EDA/clean"
MODEL_SAVE     = "two_tower_pointwise_bce_prefilter_80.pt"
LOSS_CURVE     = "loss_curve.png"
EVAL_CURVE     = "eval_ndcg_curve.png"

MAX_HIST_LEN   = 20       # how many past ratings to keep
MAX_WISH_LEN   = 20       # how many past wishlist items to keep as history
K_LABEL        = 5        # how many wishlist items to treat as positives
NEG_RATIO      = 20       # number of negatives per positive
EMBED_DIM      = 128
DENSE_HID      = 64
USER_HID       = [512, 256, 128, 64]
BATCH_SIZE     = 2048
EPOCHS         = 10
LR             = 3e-4
SEED           = 42
PAD            = 0        # padding book_id

TOP_N          = 100      # nDCG@TOP_N at eval time
EVAL_USERS     = 100      # how many test users to sample for eval

# ─── REPRODUCIBILITY ────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─── 1) LOAD & BUILD METADATA ──────────────────────────────────────
books      = pd.read_csv(f"{DATA_ROOT}/books.csv")
MAX_RC     = float(books["ratings_count"].max() or 1.0)

author2idx = {a:i+1 for i,a in enumerate(sorted(books.authors.unique()))}
lang2idx   = {l:i+1 for i,l in enumerate(books.language_code.fillna("unk").unique())}

book_author = books.set_index("book_id")\
                   .authors.map(author2idx).fillna(0).astype(int).to_dict()
book_lang   = books.set_index("book_id")\
                   .language_code.fillna("unk").map(lang2idx).fillna(0).astype(int).to_dict()
book_dense  = books.set_index("book_id")[["ratings_count","average_rating"]].to_dict("index")
all_books   = books.book_id.values.astype(np.int64)

tags_df     = pd.read_csv(f"{DATA_ROOT}/tags.csv")
NUM_TAGS    = int(tags_df.tag_id.max()) + 1
book_tags   = pd.read_csv(f"{DATA_ROOT}/book_tags.csv")
top_tags    = {
    bid: (grp.sort_values("count", ascending=False)
             .tag_id.tolist() + [0]*5)[:5]
    for bid,grp in book_tags.groupby("book_id")
}

ratings       = pd.read_csv(f"{DATA_ROOT}/ratings.csv")
ratings_map   = ratings.groupby("user_id").book_id.apply(list).to_dict()

wt          = pd.read_csv(f"{DATA_ROOT}/to_read_train.csv")
wish_train  = wt.groupby("user_id").book_id.apply(list).to_dict()
we          = pd.read_csv(f"{DATA_ROOT}/to_read_test.csv")
wish_test   = we.groupby("user_id").book_id.apply(list).to_dict()

# ─── 2) PREFILTER BY TAG COUNT ──────────────────────────────────────
# Count how often each tag appears in training wishlists
pos_tag_counts = defaultdict(int)
for bl in wish_train.values():
    for b in bl:
        for t in top_tags.get(b, []):
            if t > 0:
                pos_tag_counts[t] += 1

# Compute 85th percentile and use that as cap
tag_cap = int(np.percentile(list(pos_tag_counts.values()), 98))
print(f"⎯ Tag‐cap = 85th percentile = {tag_cap}")

def prefilter(wmap: Dict[int,List[int]]) -> Dict[int,List[int]]:
    run = defaultdict(int)
    out = {}
    for u, bl in wmap.items():
        keep = []
        for b in bl:
            ts = [t for t in top_tags.get(b, []) if t > 0]
            # if any tag already at cap, skip
            if any(run[t] >= tag_cap for t in ts):
                continue
            keep.append(b)
            for t in ts:
                run[t] += 1
        out[u] = keep
    return out

wish_train = prefilter(wish_train)
wish_test  = prefilter(wish_test)

# pick EVAL_USERS random test users (after prefilter)
all_test_users = [u for u,lst in wish_test.items() if lst]
random.shuffle(all_test_users)
eval_users = all_test_users[:EVAL_USERS]

# ─── 3) BUILD POINTWISE TRAIN SAMPLES ──────────────────────────────
print(f"Building train samples (1 pos : {NEG_RATIO} neg)…")
train_samples: List[Tuple[int,List[int],List[int],int,int]] = []
for u, full_wish in tqdm(wish_train.items(), desc="Users"):
    hist = ratings_map.get(u, [])[:MAX_HIST_LEN]
    wish_hist = full_wish[K_LABEL:K_LABEL+MAX_WISH_LEN]
    excluded = set(hist) | set(full_wish)
    for pos in full_wish[:K_LABEL]:
        # positive
        train_samples.append((u, hist, wish_hist, pos, 1))
        # NEG_RATIO negatives
        for _ in range(NEG_RATIO):
            neg = None
            while neg is None:
                cand = int(np.random.choice(all_books))
                if cand not in excluded:
                    neg = cand
            train_samples.append((u, hist, wish_hist, neg, 0))

print(f"→ Total train samples: {len(train_samples)}")

# ─── 4) POINTWISE DATASET & DATALOADER ────────────────────────────
class PointwiseDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        u, hist, wish_hist, bid, lbl = self.samples[i]
        h = hist[:MAX_HIST_LEN] + [PAD]*(MAX_HIST_LEN-len(hist))
        w = wish_hist[:MAX_WISH_LEN] + [PAD]*(MAX_WISH_LEN-len(wish_hist))

        auth = book_author.get(bid, 0)
        lang = book_lang.get(bid, 0)
        tags = top_tags.get(bid, [0]*5)
        rd   = book_dense.get(bid, {"ratings_count":0,"average_rating":0.0})
        r_n  = math.log1p(rd["ratings_count"]) / math.log1p(MAX_RC)
        a_n  = max(0.0,(rd["average_rating"]-1.0)/4.0)
        u_n  = 1.0 if bid in ratings_map.get(u,[]) else 0.0
        dense = [r_n, a_n, u_n]

        return (
            torch.tensor(h,     dtype=torch.long),
            torch.tensor(w,     dtype=torch.long),
            torch.tensor(bid,   dtype=torch.long),
            torch.tensor(auth,  dtype=torch.long),
            torch.tensor(lang,  dtype=torch.long),
            torch.tensor(tags,  dtype=torch.long),
            torch.tensor(dense, dtype=torch.float32),
            torch.tensor(lbl,   dtype=torch.float32),
        )

train_ds = PointwiseDataset(train_samples)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# ─── 5) TWO‐TOWER MODEL ─────────────────────────────────────────────
class TwoTower(nn.Module):
    def __init__(self,
                 num_books, num_authors, num_langs, num_tags,
                 embed_dim, user_hids, max_hist_len, max_wish_len):
        super().__init__()
        self.book_emb = nn.Embedding(num_books+1, embed_dim, padding_idx=0)
        self.auth_emb = nn.Embedding(num_authors+1, embed_dim, padding_idx=0)
        self.lang_emb = nn.Embedding(num_langs+1, embed_dim, padding_idx=0)
        self.tag_emb  = nn.Embedding(num_tags+1, embed_dim, padding_idx=0)

        self.dense_mlp = nn.Sequential(
            nn.Linear(3, DENSE_HID),
            nn.ReLU(inplace=True),
            nn.Linear(DENSE_HID, embed_dim)
        )

        layers, prev = [], embed_dim
        for h in user_hids:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, embed_dim)]
        self.user_mlp = nn.Sequential(*layers)

    def forward(self,
                hist_ids, wish_ids,
                bid, auth, lang, tags, dense):
        h_emb = self.book_emb(hist_ids).mean(dim=1)
        w_emb = self.book_emb(wish_ids).mean(dim=1)
        u0    = h_emb + w_emb
        u_emb = self.user_mlp(u0)

        b_e   = self.book_emb(bid)
        a_e   = self.auth_emb(auth)
        l_e   = self.lang_emb(lang)
        t_e   = self.tag_emb(tags).mean(dim=1)
        d_e   = self.dense_mlp(dense)

        i_emb = b_e + a_e + l_e + t_e + d_e
        return (u_emb * i_emb).sum(dim=1, keepdim=True)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = TwoTower(
    num_books     = int(books.book_id.max()),
    num_authors   = max(author2idx.values()),
    num_langs     = max(lang2idx.values()),
    num_tags      = NUM_TAGS,
    embed_dim     = EMBED_DIM,
    user_hids     = USER_HID,
    max_hist_len  = MAX_HIST_LEN,
    max_wish_len  = MAX_WISH_LEN,
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ─── 6) RECOMMENDER FOR EVAL ────────────────────────────────────────
def recommend_top_n(uid, model, ratings_map, wish_map, top_n=TOP_N, batch_size=8192):
    hist      = ratings_map.get(uid, [])[:MAX_HIST_LEN]
    hist_ids  = hist + [PAD]*(MAX_HIST_LEN-len(hist))
    full_wish = wish_map.get(uid, [])
    labels    = full_wish[:K_LABEL]
    wish_hist = full_wish[K_LABEL:K_LABEL+MAX_WISH_LEN]
    wish_ids  = wish_hist + [PAD]*(MAX_WISH_LEN-len(wish_hist))

    def make_dense(lst):
        out=[]
        for b in lst:
            if b==PAD:
                out += [0.0,0.0,0.0]
            else:
                rd = book_dense[b]
                out += [
                    math.log1p(rd["ratings_count"])/math.log1p(MAX_RC),
                    max(0.0,(rd["average_rating"]-1.0)/4.0),
                    1.0 if b in ratings_map.get(uid,[]) else 0.0
                ]
        return out

    h_t = torch.tensor(hist_ids, dtype=torch.long, device=device).unsqueeze(0)
    w_t = torch.tensor(wish_ids, dtype=torch.long, device=device).unsqueeze(0)

    top_scores = torch.full((top_n,), -1e9)
    top_books  = torch.full((top_n,), -1, dtype=torch.int64)

    model.eval()
    books_list = all_books.copy()
    np.random.shuffle(books_list)
    with torch.no_grad():
        for i in range(0, len(books_list), batch_size):
            batch = books_list[i:i+batch_size]
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
            vals, idx = all_s.topk(top_n)
            top_scores, top_books = vals, all_b[idx]

    ordered = top_books[torch.argsort(-top_scores)].tolist()
    return ordered, labels

IDCG = sum(1.0/math.log2(i+2) for i in range(min(K_LABEL, TOP_N)))

# ─── 7) TRAIN + EVAL LOOP ──────────────────────────────────────────
loss_history      = []
eval_ndcg_history = []

for ep in range(1, EPOCHS+1):
    # — train —
    model.train()
    total_loss = 0.0
    for h, w, b, a, l, t, d, y in tqdm(train_loader, desc=f"Epoch {ep} Train"):
        h, w, b, a, l, t, d, y = [x.to(device) for x in (h, w, b, a, l, t, d, y)]
        logits = model(h, w, b, a, l, t, d).squeeze(1)
        loss   = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f">>> Epoch {ep}: Train BCE Loss = {avg_loss:.4f}")

    # overwrite checkpoint
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

    # — eval nDCG@TOP_N —
    ndcg_vals = []
    for uid in tqdm(eval_users, desc=f"Epoch {ep} Eval"):
        recs, labels = recommend_top_n(uid, model, ratings_map, wish_test, TOP_N)
        dcg = 0.0
        for pos in labels:
            if pos in recs:
                idx = recs.index(pos)
                dcg += 1.0/math.log2(idx+2)
        ndcg_vals.append(dcg/IDCG)

    avg_ndcg = float(np.mean(ndcg_vals))
    eval_ndcg_history.append(avg_ndcg)
    print(f"▶︎ Epoch {ep}: Eval nDCG@{TOP_N} = {avg_ndcg:.4f}")

# ─── 8) PLOT ────────────────────────────────────────────────────────
plt.figure()
plt.plot(range(1, EPOCHS+1), loss_history, marker='o', label='Train BCE Loss')
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout(); plt.savefig(LOSS_CURVE)

plt.figure()
plt.plot(range(1, EPOCHS+1), eval_ndcg_history, marker='o', label=f"Eval nDCG@{TOP_N}")
plt.xlabel("Epoch"); plt.ylabel(f"nDCG@{TOP_N}"); plt.legend(); plt.tight_layout(); plt.savefig(EVAL_CURVE)

print("All done.")
