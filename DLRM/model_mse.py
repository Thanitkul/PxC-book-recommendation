#!/usr/bin/env python3
import os
import sys
import json
import math
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from tqdm import tqdm
import gc  # Import garbage collector

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ══════════════════ CONFIG ══════════════════════════════════════════
DATA_ROOT    = "../data-prep-EDA/clean"  # Adjust if your data is elsewhere
K_NEG        = 5   # negatives per positive
EMBED_DIM    = 32
BATCH_SIZE   = 2048 # Reduced batch size due to increased feature dimensionality
EPOCHS       = 8
LR           = 3e-4
WEIGHT_DECAY = 1e-6
NUM_GPUS     = 1
CPU_WORKERS  = 40   # Adjusted for potentially higher memory use per worker
LOG_EVERY    = 50
REGEN_NPZ    = True      # force rebuild if True
MAX_HIST_LEN = 20         # Max length for rated/wishlist history
TAG_JSON     = "tag_counts.json"     # where we’ll store the positive-only counts
PATHS = {
    "books":      f"{DATA_ROOT}/books.csv",
    "ratings":    f"{DATA_ROOT}/ratings.csv",
    "wish_train": f"{DATA_ROOT}/to_read_train.csv",
    "wish_test":  f"{DATA_ROOT}/to_read_test.csv",
    "book_tags":  f"{DATA_ROOT}/book_tags.csv",
    "tags":       f"{DATA_ROOT}/tags.csv",
    "train_npz":  "data/goodreads_features_mse_train.npz",
    "test_npz":   "data/goodreads_features_mse_test.npz",
    "tag_json":   "data/tag_label_counts.json",
}

# ═══════════════  bring FB DLRM  ════════════════════════════════════
sys.path.insert(0, "dlrm")
from dlrm_s_pytorch import DLRM_Net

# ──  derive dataset-level stats once ────────────────────────────────
try:
    _books_df = pd.read_csv(PATHS["books"])
    MAX_RATINGS_COUNT = float(_books_df["ratings_count"].max() or 1.0)
    del _books_df
    gc.collect()  # Clean up memory
except FileNotFoundError:
    print(f"Error: Books CSV not found at {PATHS['books']}")
    sys.exit(1)

# ── global pads and placeholders ────────────────────────────────────
PAD_BOOK       = 0
PAD_SPARSE_VEC = [0] * 7   # Padding for sparse features of a book
PAD_DENSE_VEC  = [0.0] * 3 # Padding for dense + user-rating features (3 dims)

# derived input dimensions for guards
SPARSE_INPUT_DIM = len(PAD_SPARSE_VEC) * (2 * MAX_HIST_LEN + 1)
DENSE_INPUT_DIM  = len(PAD_DENSE_VEC)  * (2 * MAX_HIST_LEN + 1)

# globals hydrated during NPZ generation
global book_author, book_lang, book_dense, top_tags
global ratings_by_user, ratings_full, all_books_np, books_by_tag
global max_tag_count, pos_tag_counts_pre
book_author = book_lang = book_dense = top_tags = None
ratings_by_user = ratings_full = all_books_np = books_by_tag = None
pos_tag_counts_pre = {}
max_tag_count = 0

# ══════════════════ helper functions ═══════════════════════════════
def get_sparse_vec7(bid: int):
    if bid == PAD_BOOK:
        return PAD_SPARSE_VEC
    return [
        book_author.get(bid, 0),
        book_lang.get(bid, 0),
        *top_tags.get(bid, [0] * 5)
    ]

def get_dense_vec3(uid: int, bid: int, max_rc: float):
    if bid == PAD_BOOK:
        return PAD_DENSE_VEC
    rec = book_dense.get(bid, {'ratings_count': 0.0, 'average_rating': 0.0})
    r_norm = math.log1p(rec['ratings_count']) / math.log1p(max_rc)
    a_norm = (rec['average_rating'] - 1.0) / 4.0
    u_norm = (ratings_full.get(uid, {}).get(bid, 0.0) - 1.0) / 4.0 if ratings_full.get(uid) else 0.0
    return [
        max(0, min(1, r_norm)),
        max(0, min(1, a_norm)),
        max(0, min(1, u_norm))
    ]

# ── multiprocessing helpers ────────────────────────────────────────
def _init_pool(glob_dict):
    global book_author, book_lang, book_dense, top_tags
    global ratings_by_user, ratings_full, all_books_np, books_by_tag
    global K_NEG, MAX_HIST_LEN, PAD_BOOK, PAD_SPARSE_VEC, PAD_DENSE_VEC
    global pos_tag_counts_pre, max_tag_count
    book_author        = glob_dict['book_author']
    book_lang          = glob_dict['book_lang']
    book_dense         = glob_dict['book_dense']
    top_tags           = glob_dict['top_tags']
    ratings_by_user    = glob_dict['ratings_by_user']
    ratings_full       = glob_dict['ratings_full']
    all_books_np       = glob_dict['all_books_np']
    books_by_tag       = glob_dict['books_by_tag']
    K_NEG              = glob_dict['K_NEG']
    MAX_HIST_LEN       = glob_dict['MAX_HIST_LEN']
    PAD_BOOK           = glob_dict['PAD_BOOK']
    PAD_SPARSE_VEC     = glob_dict['PAD_SPARSE_VEC']
    PAD_DENSE_VEC      = glob_dict['PAD_DENSE_VEC']
    pos_tag_counts_pre = glob_dict['pos_tag_counts_pre']
    max_tag_count      = glob_dict['max_tag_count']

# ── tag-aware negative sampling ────────────────────────────────────
def sample_tag_aware_negative(uid, pos_bid):
    tags = [t for t in top_tags.get(pos_bid, []) if t]
    random.shuffle(tags)
    for t in tags:
        cand = random.choice(books_by_tag[t])
        if cand != PAD_BOOK and cand not in ratings_by_user.get(uid, []):
            return cand
    return int(np.random.choice(all_books_np))

def build_rows(pair):
    uid, wish_list = pair
    wish_list = wish_list[:MAX_HIST_LEN]
    # **NO** internal filtering here – wish_list is already pre-filtered
    hist = ratings_by_user.get(uid, [])[:MAX_HIST_LEN]
    hist_s, hist_d = [], []
    for b in hist:
        hist_s += get_sparse_vec7(b)
        hist_d += get_dense_vec3(uid, b, MAX_RATINGS_COUNT)
    pad = MAX_HIST_LEN - len(hist)
    hist_s += PAD_SPARSE_VEC * pad
    hist_d += PAD_DENSE_VEC * pad

    # build wishlist block
    wish_s, wish_d = [], []
    for b in wish_list:
        wish_s += get_sparse_vec7(b)
        wish_d += get_dense_vec3(uid, b, MAX_RATINGS_COUNT)
    pad = MAX_HIST_LEN - len(wish_list)
    wish_s += PAD_SPARSE_VEC * pad
    wish_d += PAD_DENSE_VEC * pad

    base_s = hist_s + wish_s
    base_d = hist_d + wish_d

    Xi, Xc, Y = [], [], []
    for pos in wish_list:
        # positive sample
        Xi.append(np.array(base_s + get_sparse_vec7(pos), dtype=np.int32))
        Xc.append(np.array(base_d + get_dense_vec3(uid, pos, MAX_RATINGS_COUNT), dtype=np.float32))
        Y.append([1])
        # negatives
        negs = 0
        attempts = 0
        while negs < K_NEG and attempts < 20 * K_NEG:
            attempts += 1
            neg = sample_tag_aware_negative(uid, pos)
            if neg in hist or neg in wish_list or neg == PAD_BOOK:
                continue
            Xi.append(np.array(base_s + get_sparse_vec7(neg), dtype=np.int32))
            Xc.append(np.array(base_d + get_dense_vec3(uid, neg, MAX_RATINGS_COUNT), dtype=np.float32))
            Y.append([0])
            negs += 1

    # guard against empty
    if not Xi:
        Xi_arr = np.empty((0, SPARSE_INPUT_DIM), dtype=np.int32)
        Xc_arr = np.empty((0, DENSE_INPUT_DIM), dtype=np.float32)
        y_arr  = np.empty((0, 1), dtype=np.float32)
        return Xi_arr, Xc_arr, y_arr

    return np.vstack(Xi), np.vstack(Xc), np.vstack(Y)

def regenerate_npz():
    global book_author, book_lang, book_dense, top_tags
    global ratings_by_user, ratings_full, all_books_np, books_by_tag
    global max_tag_count, pos_tag_counts_pre

    # ── load CSVs ─────────────────────────────────────────
    books      = pd.read_csv(PATHS["books"])
    ratings_df = pd.read_csv(PATHS["ratings"])
    wish_tr    = pd.read_csv(PATHS["wish_train"])
    wish_te    = pd.read_csv(PATHS["wish_test"])
    book_tags  = pd.read_csv(PATHS["book_tags"])
    tags_df    = pd.read_csv(PATHS["tags"])  # <-- we'll need this at the end

    # ── build maps ───────────────────────────────────────
    author2idx = {a: i+1 for i,a in enumerate(sorted(books.authors.unique()))}
    lang2idx   = {l: i+1 for i,l in enumerate(books.language_code.fillna('unk').unique())}
    book_author = books.set_index('book_id').authors.map(author2idx).fillna(0).astype(int).to_dict()
    book_lang   = books.set_index('book_id').language_code.fillna('unk').map(lang2idx).fillna(0).astype(int).to_dict()
    book_dense  = books.set_index('book_id')[['ratings_count','average_rating']].to_dict('index')
    all_books_np= books.book_id.values.astype(np.int32)

    top_tags = {}
    books_by_tag = defaultdict(list)
    for bid, grp in book_tags.groupby('book_id'):
        tg = (grp.sort_values('count', ascending=False).tag_id.tolist() + [0]*5)[:5]
        top_tags[bid] = tg
        for t in tg:
            if t:
                books_by_tag[t].append(bid)

    ratings_full    = ratings_df.groupby('user_id').apply(lambda d: dict(zip(d.book_id,d.rating))).to_dict()
    ratings_by_user = ratings_df.groupby('user_id').book_id.apply(list).to_dict()
    wish_train      = wish_tr.groupby('user_id').book_id.apply(list).to_dict()
    wish_test       = wish_te.groupby('user_id').book_id.apply(list).to_dict()

    # ── compute raw positive counts & median threshold ───────
    pos_tag_counts_pre = defaultdict(int)
    for blist in wish_train.values():
        for b in blist:
            for t in top_tags.get(b, []):
                if t:
                    pos_tag_counts_pre[t] += 1

    median_ct = int(np.median(list(pos_tag_counts_pre.values())))
    max_tag_count = int(median_ct * 1.2)
    print(f"[Data] computed threshold = {max_tag_count}")

    # ── dynamic pre-filter wishlist ──────────────────────────
    def prefilter(wish_map):
        run_ct = defaultdict(int)
        filtered = {}
        for uid, blist in wish_map.items():
            keep = []
            for b in blist:
                tags = [t for t in top_tags.get(b, []) if t]
                if any(run_ct[t] >= max_tag_count for t in tags):
                    continue
                keep.append(b)
                for t in tags:
                    run_ct[t] += 1
            filtered[uid] = keep
        return filtered

    filtered_train = prefilter(wish_train)
    filtered_test  = prefilter(wish_test)

    # ── prepare embedding‐size counts for DLRM ───────────────
    num_authors = len(author2idx) + 1
    num_langs   = len(lang2idx)   + 1
    num_tags    = int(tags_df.tag_id.max()) + 1
    single_counts = [num_authors, num_langs] + [num_tags]*5
    counts = np.array(single_counts * (2*MAX_HIST_LEN + 1), dtype=np.int64)

    # ── helper to build NPZ ───────────────────────────────────
    glob = dict(
        book_author=book_author, book_lang=book_lang, book_dense=book_dense,
        top_tags=top_tags, ratings_by_user=ratings_by_user, ratings_full=ratings_full,
        all_books_np=all_books_np, books_by_tag=books_by_tag,
        K_NEG=K_NEG, MAX_HIST_LEN=MAX_HIST_LEN,
        PAD_BOOK=PAD_BOOK, PAD_SPARSE_VEC=PAD_SPARSE_VEC, PAD_DENSE_VEC=PAD_DENSE_VEC,
        pos_tag_counts_pre=pos_tag_counts_pre, max_tag_count=max_tag_count
    )

    def _build(wmap, label):
        Xi_list, Xc_list, Y_list = [], [], []
        with mp.Pool(CPU_WORKERS, initializer=_init_pool, initargs=(glob,)) as pool:
            for xi, xc, y in tqdm(pool.imap_unordered(build_rows, wmap.items()),
                                   total=len(wmap), desc=f"Building {label}"):
                if xi.size == 0:
                    continue
                Xi_list.append(xi); Xc_list.append(xc); Y_list.append(y)
        if not Xi_list:
            # no data → empty arrays
            return (np.empty((0, SPARSE_INPUT_DIM), dtype=np.int32),
                    np.empty((0, DENSE_INPUT_DIM), dtype=np.float32),
                    np.empty((0, 1), dtype=np.float32))
        return np.vstack(Xi_list), np.vstack(Xc_list), np.vstack(Y_list)

    # ── build & save NPZ ──────────────────────────────────────
    X_int_tr, X_den_tr, y_tr = _build(filtered_train, 'train')
    X_int_te, X_den_te, y_te = _build(filtered_test,  'test')

    os.makedirs(os.path.dirname(PATHS["train_npz"]), exist_ok=True)
    np.savez_compressed(PATHS["train_npz"], X_int=X_int_tr, X_dense=X_den_tr, y=y_tr, counts=counts)
    np.savez_compressed(PATHS["test_npz"],  X_int=X_int_te, X_dense=X_den_te, y=y_te, counts=counts)
    print(f"[Data] wrote NPZ files")

    # ── now tally both positive & negative tag counts ─────────
    train_npz = np.load(PATHS["train_npz"])
    Xi_full = train_npz["X_int"]   # shape = (N_samples, SPARSE_INPUT_DIM)
    y_full  = train_npz["y"].reshape(-1)  # (N_samples,)

    # reshape to (N, 2*MAX_HIST_LEN+1, 7) and pick last block
    N = Xi_full.shape[0]
    feat_len = 2*MAX_HIST_LEN + 1   # e.g. 41
    tag_blocks = Xi_full.reshape(N, feat_len, 7)[:, -1, 2:]  # shape (N,5)

    pos_counts = defaultdict(int)
    neg_counts = defaultdict(int)
    for i, tags in enumerate(tag_blocks):
        label = int(y_full[i])
        for t in tags:
            if t == 0: 
                continue
            if label == 1:
                pos_counts[int(t)] += 1
            else:
                neg_counts[int(t)] += 1

    # map tag_id -> tag_name
    tagid2name = dict(zip(tags_df.tag_id, tags_df.tag_name))

    # build final JSON
    out = {}
    for t_id, name in tagid2name.items():
        if t_id not in pos_counts and t_id not in neg_counts:
            continue
        out[name] = {
            "0": neg_counts.get(t_id, 0),
            "1": pos_counts.get(t_id, 0)
        }

    # write it
    os.makedirs(os.path.dirname(PATHS["tag_json"]), exist_ok=True)
    with open(PATHS["tag_json"], 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"[Data] wrote tag‐label counts by name → {PATHS['tag_json']}")


# ═══════════════ Dataset / Loader ═══════════════════════════════════
class GoodreadsNPZ(Dataset):
    def __init__(self, path, counts_ref=None):
        f = np.load(path)
        self.X_int = f["X_int"]
        self.X_dense = f["X_dense"]
        self.y = f["y"]
        self.counts = f["counts"] if "counts" in f.files else counts_ref
        self.num_sparse_features = self.X_int.shape[1]
        self.num_dense_features  = self.X_dense.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_int[idx].astype(np.int64),
            self.X_dense[idx].astype(np.float32),
            self.y[idx].astype(np.float32),
        )

def collate(batch):
    sp = torch.tensor([b[0] for b in batch], dtype=torch.long)
    de = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    y  = torch.tensor([b[2] for b in batch], dtype=torch.float32).view(-1,1)
    lS_i = [sp[:, i] for i in range(sp.shape[1])]
    lS_o = [torch.arange(de.size(0), dtype=torch.long)] * sp.shape[1]
    return de, lS_o, lS_i, y

# ═══════════════ main execution ═══════════════════════════════════
if __name__ == "__main__":
    if REGEN_NPZ or not (os.path.exists(PATHS["train_npz"]) and os.path.exists(PATHS["test_npz"])):
        regenerate_npz()

    train_ds = GoodreadsNPZ(PATHS["train_npz"])
    test_ds  = GoodreadsNPZ(PATHS["test_npz"], counts_ref=train_ds.counts)
    num_sparse_features = train_ds.num_sparse_features
    num_dense_features  = train_ds.num_dense_features

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate, num_workers=CPU_WORKERS//2)
    test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=CPU_WORKERS//2)

    print("── Dataset shapes ──")
    print(f"train_ds   X_int : {train_ds.X_int.shape}  X_dense : {train_ds.X_dense.shape}  y : {train_ds.y.shape}")
    print(f"test_ds    X_int : {test_ds.X_int.shape}   X_dense : {test_ds.X_dense.shape}   y : {test_ds.y.shape}")

    ln_emb = train_ds.counts
    ln_bot = np.array([train_ds.num_dense_features, 128, 64, EMBED_DIM])
    nfeat  = train_ds.num_sparse_features + 1
    ln_top = np.array([EMBED_DIM + (nfeat*(nfeat-1)//2), 256, 128, 1])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DLRM_Net(
        m_spa=EMBED_DIM,
        ln_emb=ln_emb,
        ln_bot=ln_bot,
        ln_top=ln_top,
        arch_interaction_op="dot",
        sigmoid_bot=-1,
        sigmoid_top=len(ln_top)-2,
        loss_function="mse",
        ndevices=1
    ).to(device)

    # --- Training Loop ---
    bce = nn.BCELoss()
    def weighted_bce(y_pred, y_true):
        weights = torch.where(y_true == 1, 1.0, 1.0 / (K_NEG + 1))
        loss = bce(y_pred, y_true)
        return (loss * weights).mean()
    
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)
    print("[Main] Optimizer: Adagrad")

    def run_epoch(epoch: int, train: bool):
        loader = train_dl if train else test_dl
        model.train(train)
        total_samples = total_loss = total_correct = true_pos = false_pos = false_neg = 0
        pbar = tqdm(loader, desc=f"{'Train' if train else 'Test '} {epoch}", ncols=90, leave=False)
        with torch.set_grad_enabled(train):
            for step, (dense_features, lS_o, lS_i, y_true) in enumerate(pbar):
                dense_features = dense_features.to(device)
                lS_i = [S_i.to(device) for S_i in lS_i]
                lS_o = [S_o.to(device) for S_o in lS_o]
                y_true = y_true.to(device)
                y_pred = model(dense_features, lS_o, lS_i)
                loss   = weighted_bce(y_pred, y_true)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                batch_size    = y_true.size(0)
                total_loss   += loss.item() * batch_size
                total_samples+= batch_size
                preds_binary  = (y_pred > 0.5).float()
                total_correct+= (preds_binary == y_true).sum().item()
                tp = ((preds_binary == 1) & (y_true == 1)).sum().item()
                fp = ((preds_binary == 1) & (y_true == 0)).sum().item()
                fn = ((preds_binary == 0) & (y_true == 1)).sum().item()
                true_pos  += tp
                false_pos += fp
                false_neg += fn
                if step > 0 and step % LOG_EVERY == 0:
                    pbar.set_postfix(
                        loss=f"{(total_loss/total_samples):.4f}",
                        acc =f"{(total_correct/total_samples*100):.2f}%"
                    )
        precision = true_pos / (true_pos + false_pos + 1e-8)
        recall    = true_pos / (true_pos + false_neg + 1e-8)
        f1        = 2 * (precision * recall) / (precision + recall + 1e-8)
        return (total_loss/total_samples, total_correct/total_samples, precision, recall, f1)

    print("\n[Main] Starting Training...")
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = run_epoch(ep, train=True)
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        te_loss, te_acc, te_prec, te_rec, te_f1 = run_epoch(ep, train=False)
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        print(f"\nEpoch {ep}/{EPOCHS}:")
        print(f"  Train | Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.2f}%, Prec: {tr_prec:.4f}, Rec: {tr_rec:.4f}, F1: {tr_f1:.4f}")
        print(f"  Test  | Loss: {te_loss:.4f}, Acc: {te_acc*100:.2f}%, Prec: {te_prec:.4f}, Rec: {te_rec:.4f}, F1: {te_f1:.4f}")

    print("\n[Main] Training complete ✅")
    EXPORT = "trained_dlrm_goodreads_features.pt"
    try:
        torch.save({
            "model_state_dict":    model.state_dict(),
            "embedding_sizes":     ln_emb.tolist(),
            "bottom_mlp":          ln_bot.tolist(),
            "top_mlp":             ln_top.tolist(),
            "embed_dim":           EMBED_DIM,
            "num_sparse_features": num_sparse_features,
            "num_dense_features":  num_dense_features,
            "max_hist_len":        MAX_HIST_LEN,
        }, EXPORT)
        print(f"✅ Model and config saved to {EXPORT}")
    except Exception as e:
        print(f"Error saving model: {e}")
