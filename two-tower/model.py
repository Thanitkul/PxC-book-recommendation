#!/usr/bin/env python3
"""
two_tower_train_eval.py

A full PyTorch two-tower recommender:
 - User tower: averages book embeddings over user history, then a small MLP
 - Item tower: sums book, author, language, and tag embeddings + dense MLP
 - Trains with BCEWithLogitsLoss on pointwise (user, item) pairs
 - After each epoch, evaluates on a held-out test set,
   computing Precision/Recall/F1 and printing the results.
 - At the end, writes out tag-by-label counts, and saves:
     - loss_curve.png
     - metrics_curve.png
     - tag_label_counts_two_tower.json
     - tag_label_area_two_tower.png
"""
import os, math, json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# ─── CONFIG ─────────────────────────────────────────────────────────
DATA_ROOT       = "../data-prep-EDA/clean"
MODEL_SAVE      = "two_tower_model.pt"
TAG_COUNT_JSON  = "tag_label_counts_two_tower.json"
TAG_PLOT        = "tag_label_area_two_tower.png"
LOSS_CURVE      = "loss_curve.png"
METRICS_CURVE   = "metrics_curve.png"

MAX_HIST_LEN    = 20    # how many books per user's history
K_NEG           = 2     # negatives per positive
EMBED_DIM       = 64    # increased embedding size
DENSE_HIDS      = [64, 32]
USER_HIDS       = [128, 64]
BATCH_SIZE      = 1024
EPOCHS          = 500
LR              = 3e-4
CPU_WORKERS     = 8
SEED            = 42

PATHS = {
    "books":      f"{DATA_ROOT}/books.csv",
    "ratings":    f"{DATA_ROOT}/ratings.csv",
    "wish_train": f"{DATA_ROOT}/to_read_train.csv",
    "wish_test":  f"{DATA_ROOT}/to_read_test.csv",
    "book_tags":  f"{DATA_ROOT}/book_tags.csv",
    "tags":       f"{DATA_ROOT}/tags.csv",
}

# ─── REPRODUCIBILITY ────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─── 1) LOAD & BUILD METADATA ──────────────────────────────────────
print("Loading metadata…")
books = pd.read_csv(PATHS["books"])
MAX_RC = float(books["ratings_count"].max() or 1.0)

author2idx = {a: i+1 for i,a in enumerate(sorted(books.authors.unique()))}
lang2idx   = {l: i+1 for i,l in enumerate(books.language_code.fillna("unk").unique())}

book_author = books.set_index("book_id").authors.map(author2idx).fillna(0).astype(int).to_dict()
book_lang   = books.set_index("book_id").language_code.fillna("unk").map(lang2idx).fillna(0).astype(int).to_dict()
book_dense  = books.set_index("book_id")[["ratings_count","average_rating"]].to_dict("index")
all_books   = books.book_id.values.astype(np.int64)

tags_df     = pd.read_csv(PATHS["tags"])
tag_id_to_name = dict(zip(tags_df.tag_id, tags_df.tag_name))
NUM_TAGS    = int(tags_df.tag_id.max()) + 1

bt = pd.read_csv(PATHS["book_tags"])
top_tags = {}
books_by_tag = defaultdict(list)
for bid, grp in bt.groupby("book_id"):
    t5 = grp.sort_values("count",ascending=False).tag_id.tolist()[:5]
    t5 += [0]*(5-len(t5))
    top_tags[bid] = t5
    for t in t5:
        if t>0:
            books_by_tag[t].append(bid)

ratings = pd.read_csv(PATHS["ratings"])
ratings_by_user = ratings.groupby("user_id").book_id.apply(list).to_dict()
ratings_full    = ratings.groupby("user_id").apply(lambda d: dict(zip(d.book_id,d.rating))).to_dict()

wt = pd.read_csv(PATHS["wish_train"])
wish_train_map = wt.groupby("user_id").book_id.apply(list).to_dict()
we = pd.read_csv(PATHS["wish_test"])
wish_test_map  = we.groupby("user_id").book_id.apply(list).to_dict()

# ─── 2) TAG‐CAP & FILTER ────────────────────────────────────────────
print("Computing tag‐cap threshold…")
pos_tag_counts = defaultdict(int)
for bl in wish_train_map.values():
    for b in bl:
        for t in top_tags.get(b,[]):
            if t>0:
                pos_tag_counts[t] += 1

median_ct = int(np.median(list(pos_tag_counts.values())))
tag_cap   = int(median_ct * 1.2)
print(f" median pos‐tag count={median_ct}, cap={tag_cap}")

def prefilter(wmap):
    run = defaultdict(int)
    out = {}
    for u, bl in wmap.items():
        keep=[]
        for b in bl:
            ts = [t for t in top_tags.get(b,[]) if t>0]
            if any(run[t]>=tag_cap for t in ts):
                continue
            keep.append(b)
            for t in ts:
                run[t] += 1
        out[u] = keep
    return out

filtered_train = prefilter(wish_train_map)
filtered_test  = prefilter(wish_test_map)

# ─── 3) NEGATIVE SAMPLER ────────────────────────────────────────────
def sample_neg(uid, pos_bid):
    ts = [t for t in top_tags.get(pos_bid,[]) if t>0]
    random.shuffle(ts)
    for t in ts:
        c = random.choice(books_by_tag[t])
        if c not in ratings_by_user.get(uid,[]) and c!=0:
            return c
    return int(np.random.choice(all_books))

# ─── 4) DATASET ─────────────────────────────────────────────────────
PAD = 0
class TwoTowerDataset(Dataset):
    def __init__(self, wish_map):
        self.samples = []
        for u, bl in wish_map.items():
            hist = ratings_by_user.get(u,[])[:MAX_HIST_LEN]
            for pos in bl[:MAX_HIST_LEN]:
                self.samples.append((u, hist, pos, 1))
                negs, atts = 0, 0
                while negs < K_NEG and atts < 20*K_NEG:
                    atts+=1
                    c = sample_neg(u,pos)
                    if c in hist or c in bl or c==0: continue
                    self.samples.append((u, hist, c, 0))
                    negs+=1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, hist, bid, lbl = self.samples[idx]
        h = hist[:MAX_HIST_LEN] + [PAD]*(MAX_HIST_LEN-len(hist))
        auth = book_author.get(bid,0)
        lang = book_lang.get(bid,0)
        tags = top_tags.get(bid,[0]*5)
        rec  = book_dense.get(bid,{"ratings_count":0,"average_rating":0})
        r_n  = math.log1p(rec["ratings_count"])/math.log1p(MAX_RC)
        a_n  = max(0.0,(rec["average_rating"]-1.0)/4.0)
        u_r  = 0.0
        if u in ratings_full:
            u_r = max(0.0,(ratings_full[u].get(bid,1.0)-1.0)/4.0)
        dense = [r_n, a_n, u_r]
        return (
            torch.tensor(h, dtype=torch.long),
            torch.tensor(bid, dtype=torch.long),
            torch.tensor(auth,dtype=torch.long),
            torch.tensor(lang,dtype=torch.long),
            torch.tensor(tags,dtype=torch.long),
            torch.tensor(dense,dtype=torch.float32),
            torch.tensor(lbl,dtype=torch.float32),
        )

train_ds = TwoTowerDataset(filtered_train)
test_ds  = TwoTowerDataset(filtered_test)
train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=CPU_WORKERS)
test_ld  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=CPU_WORKERS)

# ─── 5) TWO‐TOWER MODEL ──────────────────────────────────────────────
class TwoTower(nn.Module):
    def __init__(self,
                 num_books, num_authors, num_langs, num_tags,
                 embed_dim, dense_hids, user_hids, max_hist_len):
        super().__init__()
        self.max_hist_len = max_hist_len

        # embeddings
        self.book_emb = nn.Embedding(num_books+1, embed_dim, padding_idx=0)
        self.auth_emb = nn.Embedding(num_authors+1, embed_dim, padding_idx=0)
        self.lang_emb = nn.Embedding(num_langs+1, embed_dim, padding_idx=0)
        self.tag_emb  = nn.Embedding(num_tags+1, embed_dim, padding_idx=0)

        # dense‐MLP (item tower)
        layers = []
        prev = 3
        for h in dense_hids:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, embed_dim)]
        self.dense_mlp = nn.Sequential(*layers)

        # user‐MLP (after averaging history)
        u_layers = []
        prev = embed_dim
        for h in user_hids:
            u_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        u_layers += [nn.Linear(prev, embed_dim)]
        self.user_mlp = nn.Sequential(*u_layers)

    def forward(self, hist, bid, auth, lang, tags, dense):
        # user tower
        h_emb = self.book_emb(hist)             # B,H,E
        u0    = h_emb.mean(dim=1)               # B,E
        u_emb = self.user_mlp(u0)               # B,E

        # item tower
        b_e = self.book_emb(bid)                # B,E
        a_e = self.auth_emb(auth)               # B,E
        l_e = self.lang_emb(lang)               # B,E
        t_e = self.tag_emb(tags).mean(dim=1)    # B,E
        d_e = self.dense_mlp(dense)             # B,E
        i_emb = b_e + a_e + l_e + t_e + d_e     # B,E

        # dot
        return (u_emb * i_emb).sum(dim=1)       # B,

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = TwoTower(
    num_books    = int(books.book_id.max()),
    num_authors  = max(author2idx.values()),
    num_langs    = max(lang2idx.values()),
    num_tags     = NUM_TAGS,
    embed_dim    = EMBED_DIM,
    dense_hids   = DENSE_HIDS,
    user_hids    = USER_HIDS,
    max_hist_len = MAX_HIST_LEN
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ─── 6) TRAIN + EVAL EVERY EPOCH ───────────────────────────────────
train_losses, precisions, recalls, f1s = [], [], [], []

for ep in range(1, EPOCHS+1):
    # — training —
    model.train()
    tot_loss = 0.0
    pbar = tqdm(train_ld, desc=f"Epoch {ep:3d} Train")
    for h,b,a,l,t,d,y in pbar:
        h,b,a,l,t,d,y = [x.to(device) for x in (h,b,a,l,t,d,y)]
        logits = model(h,b,a,l,t,d)
        loss   = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tot_loss += loss.item()
        pbar.set_postfix(train_loss=tot_loss/(pbar.n+1))
    avg_loss = tot_loss / len(train_ld)
    train_losses.append(avg_loss)

    # — evaluation —
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for h,b,a,l,t,d,y in tqdm(test_ld, desc=f"Epoch {ep:3d} Eval "):
            h,b,a,l,t,d,y = [x.to(device) for x in (h,b,a,l,t,d,y)]
            logits = model(h,b,a,l,t,d)
            probs  = torch.sigmoid(logits)
            all_p.append(probs.cpu().numpy())
            all_y.append(y.cpu().numpy())

    p_np = np.concatenate(all_p)
    y_np = np.concatenate(all_y)
    b_np = (p_np>0.5).astype(int)

    tp = ((b_np==1)&(y_np==1)).sum()
    fp = ((b_np==1)&(y_np==0)).sum()
    fn = ((b_np==0)&(y_np==1)).sum()

    prec = tp/(tp+fp+1e-8)
    rec  = tp/(tp+fn+1e-8)
    f1   = 2*prec*rec/(prec+rec+1e-8)

    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

    print(f">>> Ep {ep:3d} — Loss {avg_loss:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | F1 {f1:.4f}")

# ─── 7) SAVE FINAL CHECKPOINT ────────────────────────────────────────
print("Saving final checkpoint…")
torch.save({
    "state_dict":   model.state_dict(),
    "num_books":    int(books.book_id.max()),
    "num_authors":  max(author2idx.values()),
    "num_langs":    max(lang2idx.values()),
    "num_tags":     NUM_TAGS,
    "embed_dim":    EMBED_DIM,
    "dense_hids":   DENSE_HIDS,
    "user_hids":    USER_HIDS,
    "max_hist_len": MAX_HIST_LEN,
}, MODEL_SAVE)
print(f"✅ Model & config saved to {MODEL_SAVE}")

# ─── 8) PLOT LOSS & METRICS ─────────────────────────────────────────
# Loss curve
plt.figure()
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='Train Loss')
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss"); plt.legend()
plt.savefig(LOSS_CURVE)
print(f"Saved loss curve → {LOSS_CURVE}")

# Metrics curve
plt.figure()
plt.plot(range(1, EPOCHS+1), precisions, marker='o', label='Precision')
plt.plot(range(1, EPOCHS+1), recalls,    marker='o', label='Recall')
plt.plot(range(1, EPOCHS+1), f1s,        marker='o', label='F1 Score')
plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Eval Metrics"); plt.legend()
plt.savefig(METRICS_CURVE)
print(f"Saved metrics curve → {METRICS_CURVE}")

# ─── 9) TAG‐BY‐LABEL COUNTS & AREA PLOT ─────────────────────────────
print("Counting tags by label…")
tag_counts = defaultdict(lambda: {"0":0,"1":0})
for u,hist_list,bid,lbl in test_ds.samples:
    for t in top_tags.get(bid, []):
        if t==0: continue
        name = tag_id_to_name.get(t,"")
        tag_counts[name][str(lbl)] += 1

with open(TAG_COUNT_JSON, "w", encoding="utf-8") as f:
    json.dump(tag_counts, f, indent=2, ensure_ascii=False)

names = list(tag_counts.keys())
c0    = [tag_counts[n]["0"] for n in names]
c1    = [tag_counts[n]["1"] for n in names]
plt.figure(figsize=(12,6))
plt.fill_between(names, c0, color="red",   alpha=0.5, label="neg")
plt.fill_between(names, c1, color="blue",  alpha=0.5, label="pos")
plt.xticks(rotation=90); plt.legend(); plt.tight_layout()
plt.savefig(TAG_PLOT)
print(f"Wrote tag plot → {TAG_PLOT} and counts → {TAG_COUNT_JSON}")
