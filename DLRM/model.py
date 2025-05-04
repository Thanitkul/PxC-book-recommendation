#!/usr/bin/env python3
"""
dlrm_catalog_train_and_eval.py

– Trains a DLRM‐style model:
    • Embeds user history slots + wishlist‐history slots + item categorical slots
    • Applies MLP to dense features + bottom MLP to each embedding
    • Combines user & item vectors and runs a top MLP to score the full catalog
– After each epoch, evaluates on a held‐out set of users by ranking the full catalog
  and computing true nDCG@TOP_N over their K_LABEL held‐out wishlist items.
– Saves model (overwriting) after each epoch + training loss curve + eval nDCG curve.
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ─── CONFIG ─────────────────────────────────────────────────────────
DATA_ROOT      = "../data-prep-EDA/clean"
MODEL_SAVE     = "dlrm_catalog.pt"
LOSS_CURVE     = "loss_curve.png"
EVAL_CURVE     = "eval_ndcg_curve.png"

MAX_HIST_LEN   = 20       # rated-history slots
MAX_WISH_LEN   = 20       # wishlist-history slots
K_LABEL        = 5        # positives per user
EMBED_DIM      = 32       # embedding dim for all sparse features
DENSE_HID      = 32       # bottom‐MLP hidden size
USER_HID       = [256,128,64]  # top‐MLP layers
BATCH_SIZE     = 1        # one user per batch
EPOCHS         = 10
LR             = 1e-3
TAU            = 4.0      # soft‐rank temperature
SEED           = 42
PAD            = 0

TOP_N          = 100       # nDCG@TOP_N
EVAL_USERS     = 100      # test users to sample for eval

# ─── REPRODUCIBILITY ────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─── 1) LOAD METADATA ──────────────────────────────────────────────
books       = pd.read_csv(f"{DATA_ROOT}/books.csv")
MAX_RC      = float(books["ratings_count"].max() or 1.0)

# categorical indices
author2idx  = {a:i+1 for i,a in enumerate(sorted(books.authors.unique()))}
lang2idx    = {l:i+1 for i,l in enumerate(books.language_code.fillna("unk").unique())}

book_author = books.set_index("book_id").authors.map(author2idx).fillna(0).astype(int).to_dict()
book_lang   = books.set_index("book_id").language_code.fillna("unk").map(lang2idx).fillna(0).astype(int).to_dict()
book_dense  = books.set_index("book_id")[["ratings_count","average_rating"]].to_dict("index")
all_books   = books.book_id.values.astype(np.int64)

tags_df     = pd.read_csv(f"{DATA_ROOT}/tags.csv")
NUM_TAGS    = int(tags_df.tag_id.max()) + 1
book_tags   = pd.read_csv(f"{DATA_ROOT}/book_tags.csv")
top_tags    = {
    bid: (grp.sort_values("count",ascending=False).tag_id.tolist() + [0]*5)[:5]
    for bid,grp in book_tags.groupby("book_id")
}

ratings     = pd.read_csv(f"{DATA_ROOT}/ratings.csv")
ratings_by_user = ratings.groupby("user_id").book_id.apply(list).to_dict()

wt          = pd.read_csv(f"{DATA_ROOT}/to_read_train.csv")
wish_train_map = wt.groupby("user_id").book_id.apply(list).to_dict()
we          = pd.read_csv(f"{DATA_ROOT}/to_read_test.csv")
wish_test_map  = we.groupby("user_id").book_id.apply(list).to_dict()

# sample test users
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

    def __len__(self): return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        # 1) history slots
        hist   = self.ratings_map.get(u, [])[:MAX_HIST_LEN]
        h_slots= hist + [PAD]*(MAX_HIST_LEN-len(hist))
        # 2) wishlist split
        full_w  = self.wish_map.get(u, [])
        pos_w   = set(full_w[:K_LABEL])
        wish_h  = full_w[K_LABEL:K_LABEL+MAX_WISH_LEN]
        w_slots = wish_h + [PAD]*(MAX_WISH_LEN-len(wish_h))
        # 3) catalog
        cands   = list(self.all_books)
        # 4) categorical features
        auth = [book_author[b] for b in cands]
        lang = [book_lang[b]   for b in cands]
        tags = [top_tags[b]    for b in cands]
        # 5) dense
        dense=[]
        for b in cands:
            if b==PAD:
                dense.append([0.,0.,0.])
            else:
                rd = book_dense[b]
                r_n = math.log1p(rd["ratings_count"])/math.log1p(MAX_RC)
                a_n = max(0.,(rd["average_rating"]-1.)/4.)
                u_r = 1.0 if b in self.ratings_map.get(u,[]) else 0.0
                dense.append([r_n,a_n,u_r])
        # 6) labels
        labels = [1.0 if b in pos_w else 0.0 for b in cands]

        return (
            torch.LongTensor(h_slots),      # [H]
            torch.LongTensor(w_slots),      # [W]
            torch.LongTensor(cands),        # [C]
            torch.LongTensor(auth),         # [C]
            torch.LongTensor(lang),         # [C]
            torch.LongTensor(tags),         # [C,5]
            torch.FloatTensor(dense),       # [C,3]
            torch.FloatTensor(labels),      # [C]
        )

train_ds = CatalogDataset(
    users=list(wish_train_map.keys()),
    ratings_map=ratings_by_user,
    wish_map=wish_train_map,
    all_books=all_books
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4)

eval_ds = CatalogDataset(
    users=selected_test_users,
    ratings_map=ratings_by_user,
    wish_map=wish_test_map,
    all_books=all_books
)
eval_loader = DataLoader(eval_ds, batch_size=1,
                         shuffle=False, num_workers=2)

# ─── 3) DLRM‐STYLE MODEL ────────────────────────────────────────────
class DLRM(nn.Module):
    def __init__(self):
        super().__init__()
        # bottom MLP
        self.bottom = nn.Sequential(
            nn.Linear(3, DENSE_HID),
            nn.ReLU(inplace=True),
            nn.Linear(DENSE_HID, EMBED_DIM),
        )
        # embeddings
        V = int(books.book_id.max())+1
        self.emb_hist = nn.Embedding(V, EMBED_DIM, padding_idx=0)
        self.emb_wish = nn.Embedding(V, EMBED_DIM, padding_idx=0)
        self.emb_cand = nn.Embedding(V, EMBED_DIM, padding_idx=0)
        self.emb_auth = nn.Embedding(len(author2idx)+1, EMBED_DIM, padding_idx=0)
        self.emb_lang = nn.Embedding(len(lang2idx)+1,   EMBED_DIM, padding_idx=0)
        self.emb_tags = nn.Embedding(NUM_TAGS, EMBED_DIM, padding_idx=0)
        # top MLP
        layers=[]
        in_dim = EMBED_DIM*2
        for h in USER_HID:
            layers += [nn.Linear(in_dim,h), nn.ReLU(inplace=True)]
            in_dim = h
        layers += [nn.Linear(in_dim,1)]
        self.top = nn.Sequential(*layers)

    def forward(self, hist, wish, cand, auth, lang, tags, dense):
        B,C = cand.size(0), cand.size(1)
        # user tower
        u_h = self.emb_hist(hist).mean(1)
        u_w = self.emb_wish(wish).mean(1)
        u_vec = u_h + u_w                              # [B,E]
        # item tower
        e_c = self.emb_cand(cand)
        e_a = self.emb_auth(auth)
        e_l = self.emb_lang(lang)
        # split 5 tag‐slots along last dim:
        t1,t2,t3,t4,t5 = tags.unbind(2)
        e_t = (self.emb_tags(t1)+self.emb_tags(t2)
             +self.emb_tags(t3)+self.emb_tags(t4)
             +self.emb_tags(t5)) / 5.0             # [B,C,E]
        i_vec = e_c + e_a + e_l + e_t                 # [B,C,E]
        # dense bottom
        d_vec = self.bottom(dense.view(-1,3)).view(B,C,EMBED_DIM)
        i_final = i_vec + d_vec                       # [B,C,E]
        # combine
        u_exp = u_vec.unsqueeze(1).expand(-1,C,-1)    # [B,C,E]
        x     = torch.cat([u_exp, i_final],dim=2)     # [B,C,2E]
        scores= self.top(x).squeeze(2)                # [B,C]
        return scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = DLRM().to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=LR)

# ideal DCG@TOP_N
IDCG = sum(1.0/math.log2(i+2) for i in range(min(K_LABEL,TOP_N)))

def soft_ndcg_loss(scores, labels, tau=TAU):
    diff  = scores.unsqueeze(2) - scores.unsqueeze(1)
    P     = torch.sigmoid(diff/tau)
    ranks = 1.0 + P.sum(2)
    gains = labels / torch.log2(ranks+1.0)
    dcg   = gains.sum(1)
    return (1.0 - dcg).mean()

# ─── 4) TRAIN + EVAL ───────────────────────────────────────────────
loss_history=[]
eval_history=[]

for ep in range(1,EPOCHS+1):
    # training
    model.train()
    running_loss = 0.0
    for hist,wish,cand,auth,lang,tags,dense,labels in tqdm(train_loader, desc=f"Train Ep{ep}"):
        hist,wish = hist.to(device), wish.to(device)
        cand,auth,lang = cand.to(device), auth.to(device), lang.to(device)
        tags,dense,labels = tags.to(device), dense.to(device), labels.to(device)
        scores = model(hist,wish,cand,auth,lang,tags,dense)
        loss   = soft_ndcg_loss(scores,labels)
        opt.zero_grad(); loss.backward(); opt.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f">>> Ep {ep} — Soft‐NDCG Loss: {avg_loss:.4f}")

    # save full artifact dict
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

    # evaluation
    model.eval()
    ndcgs = []
    with torch.no_grad():
        for hist,wish,cand,auth,lang,tags,dense,labels in tqdm(eval_loader, desc=f"Eval Ep{ep}"):
            hist,wish = hist.to(device), wish.to(device)
            cand,auth,lang = cand.to(device), auth.to(device), lang.to(device)
            tags,dense = tags.to(device), dense.to(device)
            lbls = labels.cpu().numpy()[0]

            scores_np = model(hist,wish,cand,auth,lang,tags,dense).cpu().numpy()[0]
            topk   = np.argsort(-scores_np)[:TOP_N]

            dcg=0.0
            for pos in np.where(lbls==1.0)[0]:
                if pos in topk:
                    r = int(np.where(topk==pos)[0][0])
                    dcg += 1.0/math.log2(r+2)
            ndcgs.append(dcg/IDCG)

    avg_ndcg = float(np.mean(ndcgs))
    eval_history.append(avg_ndcg)
    print(f"▶︎ Ep {ep} — Eval nDCG@{TOP_N}: {avg_ndcg:.4f}")

# final plots
plt.figure()
plt.plot(range(1,EPOCHS+1), loss_history, marker='o', label='Train Loss')
plt.xlabel("Epoch"); plt.ylabel("Soft‐NDCG Loss"); plt.legend(); plt.tight_layout()
plt.savefig(LOSS_CURVE)

plt.figure()
plt.plot(range(1,EPOCHS+1), eval_history, marker='o', label=f"Eval nDCG@{TOP_N}")
plt.xlabel("Epoch"); plt.ylabel(f"nDCG@{TOP_N}"); plt.legend(); plt.tight_layout()
plt.savefig(EVAL_CURVE)

print("Done.")
