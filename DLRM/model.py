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
from tqdm import tqdm
import gc # Import garbage collector

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ══════════════════ CONFIG ══════════════════════════════════════════
DATA_ROOT    = "../data-prep-EDA/clean"  # Adjust if your data is elsewhere
K_NEG        = 9   # negatives per positive
EMBED_DIM    = 32
BATCH_SIZE   = 10 # Reduced batch size due to increased feature dimensionality
EPOCHS       = 3
LR           = 3e-4
WEIGHT_DECAY = 1e-6
NUM_GPUS     = 1
CPU_WORKERS  = 40   # Adjusted for potentially higher memory use per worker
LOG_EVERY    = 50
REGEN_NPZ    = True      # force rebuild if True
MAX_HIST_LEN = 20         # Max length for rated/wishlist history

LOSS_TYPE    = "listnet"

PATHS = {
    "books":      f"{DATA_ROOT}/books.csv",
    "ratings":    f"{DATA_ROOT}/ratings.csv",
    "wish_train": f"{DATA_ROOT}/to_read_train.csv",
    "wish_test":  f"{DATA_ROOT}/to_read_test.csv",
    "book_tags":  f"{DATA_ROOT}/book_tags.csv",
    "tags":       f"{DATA_ROOT}/tags.csv",
    "train_npz":  f"data/goodreads_features_train_{LOSS_TYPE}.npz",
    "test_npz":   f"data/goodreads_features_test_{LOSS_TYPE}.npz",
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
PAD_BOOK        = 0
PAD_SPARSE_VEC  = [0] * 7   # Padding for sparse features of a book
PAD_DENSE_VEC   = [0.0] * 3 # Padding for dense + user-rating features (3 dims)

# globals hydrated during NPZ generation
global book_author, book_lang, book_dense, top_tags, ratings_by_user, ratings_full, all_books_np
book_author = book_lang = book_dense = top_tags = None
datings_by_user = all_books_np = None
ratings_full = None

# ══════════════════ helper functions ═══════════════════════════════
def get_sparse_vec7(bid: int):
    """Gets the 7 sparse feature indices for a book ID."""
    if bid == PAD_BOOK or book_author is None:
        return PAD_SPARSE_VEC
    author_idx = book_author.get(bid, 0)
    lang_idx   = book_lang.get(bid, 0)
    tag_indices = top_tags.get(bid, [0]*5)[:5]
    if len(tag_indices) < 5:
        tag_indices.extend([0] * (5 - len(tag_indices)))
    return [author_idx, lang_idx, *tag_indices]


def get_dense_vec2(bid: int):
    """Gets the 2 dense book-level features for a book ID."""
    if bid == PAD_BOOK or book_dense is None:
        return [0.0, 0.0]
    rec = book_dense.get(bid, {"ratings_count": 0.0, "average_rating": 0.0})
    ratings_norm = math.log1p(float(rec.get("ratings_count", 0.0))) / math.log1p(MAX_RATINGS_COUNT)
    rating_norm  = (float(rec.get("average_rating", 0.0)) - 1.0) / 4.0 if float(rec.get("average_rating", 0.0)) >= 1.0 else 0.0
    return [max(0.0, min(1.0, ratings_norm)), max(0.0, min(1.0, rating_norm))]

# ── multiprocessing helpers ────────────────────────────────────────
def _init_pool(glob_dict):
    global book_author, book_lang, book_dense, top_tags, ratings_by_user, ratings_full, all_books_np
    book_author   = glob_dict['book_author']
    book_lang     = glob_dict['book_lang']
    book_dense    = glob_dict['book_dense']
    top_tags      = glob_dict['top_tags']
    ratings_by_user = glob_dict['ratings_by_user']
    ratings_full    = glob_dict['ratings_full']
    all_books_np  = glob_dict['all_books_np']
    global K_NEG, MAX_HIST_LEN, PAD_BOOK, PAD_SPARSE_VEC, PAD_DENSE_VEC
    K_NEG          = glob_dict['K_NEG']
    MAX_HIST_LEN   = glob_dict['MAX_HIST_LEN']
    PAD_BOOK       = glob_dict['PAD_BOOK']
    PAD_SPARSE_VEC = glob_dict['PAD_SPARSE_VEC']
    PAD_DENSE_VEC  = glob_dict['PAD_DENSE_VEC']

def _hist_vectors(uid: int, bids):
    s, d = [], []
    for bid in bids:
        s.extend(get_sparse_vec7(bid))
        r2, ur = get_dense_vec2(bid), ratings_full.get(uid, {}).get(bid, 0.0)
        d.extend(r2 + [(ur - 1) / 4 if ur >= 1 else 0.0])
    pad = MAX_HIST_LEN - len(bids)
    s.extend(PAD_SPARSE_VEC * pad)
    d.extend(PAD_DENSE_VEC  * pad)
    return s, d

def build_rows_listwise(pair):
    """One positive + K_NEG negatives ⇒ slate (list‑wise)"""
    uid, wish_list = pair
    user_hist = ratings_by_user.get(uid, [])[:MAX_HIST_LEN]
    wish_list = wish_list[:MAX_HIST_LEN]

    rated_s, rated_d = _hist_vectors(uid, user_hist)
    wish_s , wish_d  = _hist_vectors(uid, wish_list)

    base_sparse = rated_s + wish_s
    base_dense  = rated_d + wish_d

    Xi_rows, Xc_rows, Y_rows, Q_rows = [], [], [], []
    q_counter = 0
    for pos_bid in wish_list:
        qid = (uid << 16) + q_counter; q_counter += 1
        # positive row
        Xi_rows.append(base_sparse + get_sparse_vec7(pos_bid))
        Xc_rows.append(base_dense  + get_dense_vec2(pos_bid) + [1.0])
        Y_rows.append([1]);  Q_rows.append([qid])
        # K negatives
        negs = 0
        while negs < K_NEG:
            neg = int(np.random.choice(all_books_np))
            if neg in user_hist or neg in wish_list or neg == PAD_BOOK: continue
            Xi_rows.append(base_sparse + get_sparse_vec7(neg))
            Xc_rows.append(base_dense + get_dense_vec2(neg) + [0.0])
            Y_rows.append([0]); Q_rows.append([qid]); negs += 1
    return Xi_rows, Xc_rows, Y_rows, Q_rows


def build_rows_pairwise(pair):
    """One positive + **one** sampled negative ⇒ a pair"""
    uid, wish_list = pair
    user_hist = ratings_by_user.get(uid, [])[:MAX_HIST_LEN]
    wish_list = wish_list[:MAX_HIST_LEN]

    rated_s, rated_d = _hist_vectors(uid, user_hist)
    wish_s , wish_d  = _hist_vectors(uid, wish_list)

    base_sparse = rated_s + wish_s
    base_dense  = rated_d + wish_d

    Xi_rows, Xc_rows, Y_rows, Q_rows = [], [], [], []
    q_counter = 0
    for pos_bid in wish_list:
        qid = (uid << 16) + q_counter; q_counter += 1
        # positive row
        Xi_rows.append(base_sparse + get_sparse_vec7(pos_bid))
        Xc_rows.append(base_dense  + get_dense_vec2(pos_bid) + [1.0])
        Y_rows.append([1]); Q_rows.append([qid])
        # sample ONE negative
        while True:
            neg = int(np.random.choice(all_books_np))
            if neg in user_hist or neg in wish_list or neg == PAD_BOOK: continue
            Xi_rows.append(base_sparse + get_sparse_vec7(neg))
            Xc_rows.append(base_dense + get_dense_vec2(neg) + [0.0])
            Y_rows.append([0]); Q_rows.append([qid])
            break
    return Xi_rows, Xc_rows, Y_rows, Q_rows


# ------- swap‑in row‑builder depending on LOSS_TYPE ------------------------
ROW_BUILDER = {
    "soft_ndcg": build_rows_listwise,
    "mse":       build_rows_listwise,   # point‑wise rows still ok
    "bpr":       build_rows_pairwise,
    "listnet": build_rows_listwise
}[LOSS_TYPE]

# ════════════════ NPZ generation ════════════════════════════════════
def regenerate_npz():
    global book_author, book_lang, book_dense, top_tags, ratings_by_user, ratings_full, all_books_np
    global num_authors, num_langs, num_tags

    books      = pd.read_csv(PATHS['books'])
    ratings_df = pd.read_csv(PATHS['ratings'])
    wish_train = pd.read_csv(PATHS['wish_train'])
    wish_test  = pd.read_csv(PATHS['wish_test'])
    book_tags  = pd.read_csv(PATHS['book_tags'])
    tags_df    = pd.read_csv(PATHS['tags'])

    # create full rating map
    ratings_full = ratings_df.groupby('user_id').apply(lambda df: dict(zip(df.book_id, df.rating))).to_dict()
    ratings_by_user = ratings_df.groupby('user_id').book_id.apply(list).to_dict()
    train_wish = wish_train.groupby('user_id').book_id.apply(list).to_dict()
    test_wish  = wish_test.groupby('user_id').book_id.apply(list).to_dict()

    # build book feature dicts
    author2idx = {a:i+1 for i,a in enumerate(sorted(books.authors.unique()))}
    lang2idx   = {l:i+1 for i,l in enumerate(books.language_code.fillna('unk').unique())}
    num_authors = len(author2idx)+1
    num_langs   = len(lang2idx)+1
    num_tags    = int(tags_df.tag_id.max())+1

    top_tags = {}
    for bid, grp in tqdm(book_tags.groupby('book_id'), desc='Aggregating Tags'):
        sorted_tags = grp.sort_values('count',ascending=False).tag_id.tolist()
        top_tags[bid] = (sorted_tags + [0]*5)[:5]

    book_author = books.set_index('book_id').authors.map(author2idx).fillna(0).astype(int).to_dict()
    book_lang   = books.set_index('book_id').language_code.fillna('unk').map(lang2idx).fillna(0).astype(int).to_dict()
    book_dense  = books.set_index('book_id')[['ratings_count','average_rating']].astype(float).to_dict('index')
    all_books_np = books.book_id.values.astype(np.int32)

    # define counts
    single_counts = [num_authors, num_langs] + [num_tags]*5
    counts = np.array(single_counts * (MAX_HIST_LEN*2+1), dtype=np.int64)

    pool_globals = dict(
        book_author=book_author, book_lang=book_lang, book_dense=book_dense,
        top_tags=top_tags, ratings_by_user=ratings_by_user, ratings_full=ratings_full,
        all_books_np=all_books_np, K_NEG=K_NEG, MAX_HIST_LEN=MAX_HIST_LEN,
        PAD_BOOK=PAD_BOOK, PAD_SPARSE_VEC=PAD_SPARSE_VEC, PAD_DENSE_VEC=PAD_DENSE_VEC
    )

    def _build(wish_map, label):
        Xi_all, Xc_all, Y_all, Q_all = [], [], [], []
        with mp.Pool(CPU_WORKERS, initializer=_init_pool, initargs=(pool_globals,)) as pool:
            for Xi_u, Xc_u, Y_u, Q_u in tqdm(
                    pool.imap_unordered(ROW_BUILDER, wish_map.items()),
                    total=len(wish_map), desc=f'Generating {label}'
            ):

                Xi_all.extend(Xi_u); Xc_all.extend(Xc_u)
                Y_all.extend(Y_u);  Q_all.extend(Q_u)

        return (np.asarray(Xi_all, np.int32),
                np.asarray(Xc_all, np.float32),
                np.asarray(Y_all , np.float32),
                np.asarray(Q_all , np.int64))


    X_int_tr, X_dense_tr, y_tr, q_tr = _build(train_wish, 'train')
    X_int_te, X_dense_te, y_te, q_te = _build(test_wish, 'test')

    os.makedirs(os.path.dirname(PATHS['train_npz']), exist_ok=True)
    np.savez_compressed(PATHS['train_npz'],
                        X_int=X_int_tr, X_dense=X_dense_tr, y=y_tr, qid=q_tr, counts=counts)
    np.savez_compressed(PATHS['test_npz'],
                        X_int=X_int_te,  X_dense=X_dense_te,  y=y_te, qid=q_te, counts=counts)


# ═══════════════ Dataset / Loader ═══════════════════════════════════
class GoodreadsNPZ(Dataset):
    def __init__(self, path, counts_ref=None):
        print(f"[Data] Loading NPZ: {path}")
        try:
            f = np.load(path)
            self.X_int  = f["X_int"]      # Sparse features (indices)
            self.X_dense = f["X_dense"]   # Dense features (continuous values)
            self.y      = f["y"]
            self.qid     = f["qid"]
            self.counts = f["counts"] if "counts" in f.files else counts_ref # Cardinalities for sparse features
            print(f"[Data] Loaded {len(self.y)} samples.")
            print(f"[Data] Sparse shape: {self.X_int.shape}, Dense shape: {self.X_dense.shape}")

            self.num_sparse_features = self.X_int.shape[1]
            self.num_dense_features = self.X_dense.shape[1]
            expected_sparse = (MAX_HIST_LEN * 2 + 1) * 7
            expected_dense = (MAX_HIST_LEN * 2 + 1) * 3
            assert self.num_sparse_features == expected_sparse, f"Expected {expected_sparse} sparse features, found {self.num_sparse_features}"
            assert self.num_dense_features == expected_dense, f"Expected {expected_dense} dense features, found {self.num_dense_features}"
            if self.counts is not None:
                 assert len(self.counts) == self.num_sparse_features, f"Counts length ({len(self.counts)}) != num sparse features ({self.num_sparse_features})"
            else:
                 print("[Data] Warning: Counts not loaded directly from NPZ, using reference.")

        except FileNotFoundError:
            print(f"Error: NPZ file not found at {path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading NPZ file {path}: {e}")
            sys.exit(1)


    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_int[idx].astype(np.int64),   # Sparse indices
            self.X_dense[idx].astype(np.float32), # Dense values
            self.y[idx].astype(np.float32),      # Target label
            self.qid[idx].astype(np.int64),
        )


def collate(batch):
    sparse_idx = torch.tensor([b[0] for b in batch], dtype=torch.long)
    dense_feat = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    labels     = torch.tensor([b[2] for b in batch], dtype=torch.float32).view(-1,1)
    qids       = torch.tensor([b[3] for b in batch], dtype=torch.long)

    lS_i = [sparse_idx[:, i] for i in range(sparse_idx.shape[1])]
    lS_o = [torch.arange(dense_feat.size(0), dtype=torch.long)] * len(lS_i)
    return dense_feat, lS_o, lS_i, labels, qids


# ═══════════════ main execution ═════════════════════════════════════
if __name__ == "__main__":
    if REGEN_NPZ or not (os.path.exists(PATHS["train_npz"]) and os.path.exists(PATHS["test_npz"])):
        print("[Main] NPZ files not found or REGEN_NPZ=True. Regenerating...")
        regenerate_npz()
        gc.collect() # Clean up memory after generation
    else:
        print("[Main] Using cached NPZs — skip generation.")

    # --- Load Data ---
    try:
        train_ds = GoodreadsNPZ(PATHS["train_npz"])
        # Load counts from train_ds if test NPZ doesn't have them (shouldn't happen with current code)
        test_ds  = GoodreadsNPZ(PATHS["test_npz"], counts_ref=train_ds.counts)
        gc.collect()
    except Exception as e:
        print(f"Fatal Error loading datasets: {e}")
        sys.exit(1)


    # --- DataLoader ---
    num_loader_workers = 0 if sys.platform == "win32" else CPU_WORKERS // 2
    pin_mem = (NUM_GPUS > 0 and torch.cuda.is_available())
    print(f"[Main] DataLoader workers: {num_loader_workers}, Pin memory: {pin_mem}")

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=False,  collate_fn=collate, num_workers=num_loader_workers, pin_memory=pin_mem, persistent_workers=num_loader_workers > 0)
    test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=num_loader_workers, pin_memory=pin_mem, persistent_workers=num_loader_workers > 0)


    # --- Model Configuration ---
    ln_emb = train_ds.counts # Cardinalities of sparse features
    if ln_emb is None:
        print("Error: Embedding cardinalities (counts) are missing.")
        sys.exit(1)

    num_sparse_features = train_ds.num_sparse_features
    num_dense_features  = train_ds.num_dense_features

    # Bottom MLP: first layer size must match number of dense features
    ln_bot = np.array([num_dense_features, 128, 64, EMBED_DIM]) # Adjusted bot MLP size

    n_feat_for_interaction = num_sparse_features + 1 # +1 for the output of ln_bot
    interact_dim_dot = EMBED_DIM + (n_feat_for_interaction * (n_feat_for_interaction - 1) // 2)

    # Top MLP: input dim depends on interaction op
    # Using dot interaction as in the original code
    ln_top = np.array([interact_dim_dot, 256, 128, 1]) # Adjusted top MLP size

    print(f"\n[DLRM Config]")
    print(f"  Sparse features: {num_sparse_features}")
    print(f"  Dense features: {num_dense_features}")
    print(f"  Embedding Dim: {EMBED_DIM}")
    print(f"  ln_emb (first 10): {ln_emb[:10]}... (total {len(ln_emb)})")
    print(f"  ln_bot: {ln_bot}")
    print(f"  ln_top: {ln_top} (using 'dot' interaction)")


    # --- Device Setup ---
    if NUM_GPUS > 0 and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache() # Clear cache before allocation
        print(f"[Main] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Main] Using CPU")


    # --- Instantiate Model ---
    try:
        model = DLRM_Net(
            m_spa=EMBED_DIM,
            ln_emb=ln_emb,
            ln_bot=ln_bot,
            ln_top=ln_top,
            arch_interaction_op="dot", # Keep dot interaction
            sigmoid_bot=-1,           # No sigmoid in bot MLP
            sigmoid_top=len(ln_top) - 2, # Sigmoid before the final layer
            loss_function="bce",      # Let's use BCE loss directly
            ndevices=NUM_GPUS if device.type == "cuda" else -1,
        )
        model = model.to(device) # Move model to device
        print(f"[Main] DLRM Model instantiated successfully.")
        # Optional: Print model summary (can be very long)
        # print(model)
    except Exception as e:
         print(f"Fatal Error instantiating DLRM model: {e}")
         # This often happens due to memory constraints, especially with large embedding tables
         if device.type == "cuda":
              print("Try reducing BATCH_SIZE, EMBED_DIM, or MLP layer sizes if this is an OOM error.")
         sys.exit(1)

    def listnet_loss(scores, labels, qids, tau=1.0, eps=1e-8):
        """
        ListNet: cross‐entropy between the true and predicted rank distributions
        over each slate of (1 pos + 9 neg).
        """
        # flatten
        scores = scores.view(-1)
        labels = labels.view(-1)
        qids   = qids.view(-1)

        # group by slate
        unique_q, inv = torch.unique(qids, return_inverse=True)
        num_q = unique_q.size(0)
        max_len = inv.bincount().max().item()  # should be 10

        pred_dist = scores.new_full((num_q, max_len), -1e9)
        true_dist = scores.new_zeros((num_q, max_len))
        mask = torch.zeros_like(true_dist, dtype=torch.bool)

        # fill in
        for idx in range(scores.size(0)):
            q = inv[idx].item()
            slot = (mask[q] == 0).nonzero(as_tuple=False)[0]
            pred_dist[q, slot] = scores[idx] / tau
            true_dist[q, slot] = labels[idx]
            mask[q, slot] = 1

        # softmax over valid slots
        P_pred = torch.softmax(pred_dist, dim=1) * mask.float()
        # ground‐truth distribution: exp(label)/sum(exp(label))
        # since labels are 1 for pos, 0 for neg → it's a one‐hot dist
        P_true = torch.softmax(true_dist / tau, dim=1) * mask.float()

        # cross‐entropy per slate
        ce = -(P_true * torch.log(P_pred + eps)).sum(dim=1)
        # average
        return ce.mean()


    # --- Loss and Optimizer ---
    neg_weight = torch.tensor([20], device=device) # Weight for positive samples
    def soft_ndcg_loss(scores, labels, qids, eps=1e-8, tau=0.5):
        """
        Vectorized differentiable Soft-NDCG loss over batch.

        Args:
            scores: (B,1) or (B,) model outputs
            labels: (B,1) or (B,) binary labels
            qids: (B,1) or (B,) query IDs
            eps: numerical stability
            tau: softmax temperature
        """
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        qids = qids.reshape(-1)

        # Group examples by qid
        unique_qids, qid_inv = torch.unique(qids, return_inverse=True)
        num_q = unique_qids.size(0)

        # Build slate tensors
        max_len = (qid_inv.bincount()).max().item()

        slates_scores = scores.new_full((num_q, max_len), fill_value=-float('inf'))  # (q, max_len)
        slates_labels = labels.new_zeros((num_q, max_len))                           # (q, max_len)
        mask = torch.zeros((num_q, max_len), dtype=torch.bool, device=scores.device)

        for idx in range(scores.size(0)):
            q_idx = qid_inv[idx]
            slot = (mask[q_idx] == 0).nonzero(as_tuple=False)[0]
            slates_scores[q_idx, slot] = scores[idx]
            slates_labels[q_idx, slot] = labels[idx]
            mask[q_idx, slot] = 1

        # Apply softmax over valid parts only
        logits = slates_scores / tau
        logits = logits.masked_fill(~mask, -1e9)  # ignore padded
        P = torch.softmax(logits, dim=1) * mask   # (q, max_len)

        gains = (2.0 ** slates_labels - 1.0) * mask  # (q, max_len)
        discounts = 1.0 / torch.log2(torch.arange(2, 2 + max_len, device=scores.device).float())  # (max_len,)
        discounts = discounts.unsqueeze(0)  # (1, max_len)

        dcg_soft = (gains * P * discounts).sum(dim=1)  # (q,)
        # ideal DCG
        _, sort_idx = torch.sort(slates_labels, descending=True, dim=1)
        sorted_gains = gains.gather(1, sort_idx)
        dcg_ideal = (sorted_gains * discounts).sum(dim=1) + eps  # (q,)

        ndcg = dcg_soft / dcg_ideal

        return (1.0 - ndcg).mean()

    mse_loss_fn  = nn.MSELoss()

    def pairwise_bpr_loss(scores, labels, qids, eps=1e-8):
        """Bayesian Personalized Ranking (logistic) over 2‑row slates."""
        scores = scores.reshape(-1); labels = labels.reshape(-1); qids = qids.reshape(-1)
        unique_qids, inv = torch.unique(qids, return_inverse=True)

        # We assume exactly **2 rows** per qid (pos, neg)
        pos_scores = scores.new_empty(unique_qids.size(0))
        neg_scores = scores.new_empty(unique_qids.size(0))

        for i in range(scores.size(0)):
            if labels[i] == 1:
                pos_scores[inv[i]] = scores[i]
            else:
                neg_scores[inv[i]] = scores[i]
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + eps).mean()
        return loss


    def compute_loss(scores, labels, qids=None):
        if LOSS_TYPE == "soft_ndcg":
            if qids is None: raise ValueError("soft_ndcg requires qids")
            return soft_ndcg_loss(scores, labels, qids)
        elif LOSS_TYPE == "bpr":
            if qids is None: raise ValueError("bpr requires qids")
            return pairwise_bpr_loss(scores, labels, qids)
        elif LOSS_TYPE == "mse":
            return mse_loss_fn(scores, labels)
        elif LOSS_TYPE == "listnet":
            return listnet_loss(scores, labels, qids)
        else:
            raise ValueError(f"Unknown LOSS_TYPE '{LOSS_TYPE}'.")
        
    # Consider AdamW or other optimizers as well
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("[Main] Optimizer: Adagrad")

    def run_epoch(epoch: int, train: bool):
        loader = train_dl if train else test_dl
        model.train(train)

        total_loss = total_correct = total_samples = 0
        true_pos = false_pos = false_neg = 0

        pbar = tqdm(loader, desc=f"{'Train' if train else 'Test '} {epoch}", ncols=90, leave=False)
        with torch.set_grad_enabled(train):
            for dense, lS_o, lS_i, y_true, q in pbar:
                dense, y_true = dense.to(device), y_true.to(device)
                lS_i = [t.to(device) for t in lS_i]
                lS_o = [t.to(device) for t in lS_o]

                # Only move qids to device if needed
                q_device = q.to(device) if LOSS_TYPE in ("soft_ndcg", "bpr", "listnet") else None
                y_pred = model(dense, lS_o, lS_i)
                loss = compute_loss(y_pred, y_true, q_device)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                bs = y_true.size(0)
                total_loss   += loss.item() * bs
                total_samples += bs

                preds = (y_pred > 0.5).float()
                total_correct += (preds == y_true).sum().item()
                true_pos  += ((preds == 1) & (y_true == 1)).sum().item()
                false_pos += ((preds == 1) & (y_true == 0)).sum().item()
                false_neg += ((preds == 0) & (y_true == 1)).sum().item()

                if pbar.n and pbar.n % LOG_EVERY == 0:
                    pbar.set_postfix(loss=f"{total_loss/total_samples:.4f}",
                                    acc=f"{100*total_correct/total_samples:.2f}%")

        prec = true_pos / (true_pos + false_pos + 1e-8)
        rec  = true_pos / (true_pos + false_neg + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        return total_loss/total_samples, total_correct/total_samples, prec, rec, f1



    # --- Main Training ---
    print("\n[Main] Starting Training...")
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = run_epoch(ep, train=True)
        gc.collect() # Collect garbage between train and test
        if device.type == 'cuda': torch.cuda.empty_cache()

        te_loss, te_acc, te_prec, te_rec, te_f1 = run_epoch(ep, train=False)
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()


        print(f"\nEpoch {ep}/{EPOCHS}:")
        print(f"  Train | Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.2f}%, Prec: {tr_prec:.4f}, Rec: {tr_rec:.4f}, F1: {tr_f1:.4f}")
        print(f"  Test  | Loss: {te_loss:.4f}, Acc: {te_acc*100:.2f}%, Prec: {te_prec:.4f}, Rec: {te_rec:.4f}, F1: {te_f1:.4f}")

    print("\n[Main] Training complete ✅")

    # --- Save Model ---
    EXPORT = "trained_dlrm_goodreads_features.pt"
    try:
        torch.save({
            "model_state_dict": model.state_dict(),
            "embedding_sizes": ln_emb.tolist(), # Save as list
            "bottom_mlp": ln_bot.tolist(),     # Save as list
            "top_mlp": ln_top.tolist(),         # Save as list
            "embed_dim": EMBED_DIM,
            "num_sparse_features": num_sparse_features,
            "num_dense_features": num_dense_features,
            "max_hist_len": MAX_HIST_LEN,
            # Include necessary info to reconstruct features if needed elsewhere
        }, EXPORT)
        print(f"✅ Model and config saved to {EXPORT}")
    except Exception as e:
        print(f"Error saving model: {e}")