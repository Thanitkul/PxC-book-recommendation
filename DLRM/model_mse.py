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
import gc # Import garbage collector

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ══════════════════ CONFIG ══════════════════════════════════════════
DATA_ROOT    = "../data-prep-EDA/clean"  # Adjust if your data is elsewhere
K_NEG        = 10   # negatives per positive
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
TAG_JSON = "tag_counts.json"     # where we’ll store the counts
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
PAD_BOOK        = 0
PAD_SPARSE_VEC  = [0] * 7   # Padding for sparse features of a book
PAD_DENSE_VEC   = [0.0] * 3 # Padding for dense + user-rating features (3 dims)

# globals hydrated during NPZ generation
global book_author, book_lang, book_dense, top_tags, ratings_by_user, ratings_full, all_books_np
book_author = book_lang = book_dense = top_tags = None
ratings_by_user = all_books_np = None
ratings_full = None
books_by_tag = None          # tag → [book_id]
tag_pos_freq = defaultdict(int)   # tag → count(label==1)
tag_neg_freq = defaultdict(int)   # tag → count(label==0)

# ══════════════════ helper functions ═══════════════════════════════
def get_sparse_vec7(bid:int):
    if bid == PAD_BOOK: return PAD_SPARSE_VEC
    return [
        book_author.get(bid,0),
        book_lang.get(bid,0),
        *top_tags.get(bid,[0]*5)
    ]

def get_dense_vec3(uid:int,bid:int,max_rc:float):
    if bid == PAD_BOOK: return PAD_DENSE_VEC
    rec = book_dense.get(bid,{'ratings_count':0.0,'average_rating':0.0})
    r_norm = math.log1p(rec['ratings_count'])/math.log1p(max_rc)
    a_norm = (rec['average_rating']-1.0)/4.0
    u_norm = (ratings_full.get(uid,{}).get(bid,0.0)-1.0)/4.0 if ratings_full.get(uid) else 0.0
    return [max(0, min(1,r_norm)), max(0,min(1,a_norm)), max(0,min(1,u_norm))]

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


# ── tag-aware negative sampling  ───────────────────────────────────
def sample_tag_aware_negative(uid, pos_bid):
    """Try to draw a negative sharing at least one tag with the positive."""
    tags = [t for t in top_tags.get(pos_bid,[]) if t]
    random.shuffle(tags)
    for t in tags:
        cand = random.choice(books_by_tag[t])
        if cand!=PAD_BOOK and cand not in ratings_by_user.get(uid,[]) and cand not in top_tags:
            return cand
    # fallback: uniform
    return int(np.random.choice(all_books_np))

def build_rows(pair):
    uid, wish_list = pair
    wish_list = wish_list[:MAX_HIST_LEN]
    hist      = ratings_by_user.get(uid,[])[:MAX_HIST_LEN]

    max_rc = MAX_RATINGS_COUNT
    # history blocks
    hist_s,hist_d=[],[]
    for b in hist:
        hist_s += get_sparse_vec7(b)
        hist_d += get_dense_vec3(uid,b,max_rc)
    pad = MAX_HIST_LEN-len(hist)
    hist_s += PAD_SPARSE_VEC*pad; hist_d += PAD_DENSE_VEC*pad

    wish_s,wish_d=[],[]
    for b in wish_list:
        wish_s += get_sparse_vec7(b)
        wish_d += get_dense_vec3(uid,b,max_rc)
    pad = MAX_HIST_LEN-len(wish_list)
    wish_s += PAD_SPARSE_VEC*pad; wish_d += PAD_DENSE_VEC*pad

    base_s = hist_s+wish_s
    base_d = hist_d+wish_d

    Xi,Xc,Y = [],[],[]
    for pos in wish_list:
        Xi.append(base_s+get_sparse_vec7(pos))
        Xc.append(base_d+get_dense_vec3(uid,pos,max_rc)); Y.append([1])

        # count tags
        for t in top_tags.get(pos,[]):
            tag_pos_freq[t]+=1

        negs=0; attempts=0
        while negs<K_NEG and attempts<K_NEG*20:
            attempts+=1
            neg = sample_tag_aware_negative(uid,pos)
            if neg in hist or neg in wish_list or neg==PAD_BOOK:continue
            Xi.append(base_s+get_sparse_vec7(neg))
            Xc.append(base_d+get_dense_vec3(uid,neg,max_rc)); Y.append([0])
            for t in top_tags.get(neg,[]):
                tag_neg_freq[t]+=1
            negs+=1
    return np.asarray(Xi,np.int32),np.asarray(Xc,np.float32),np.asarray(Y,np.float32)

# ════════════════ NPZ generation ════════════════════════════════════
def regenerate_npz():
    global book_author,book_lang,book_dense,top_tags,ratings_by_user,ratings_full,all_books_np,books_by_tag
    books      = pd.read_csv(PATHS["books"])
    ratings_df = pd.read_csv(PATHS["ratings"])
    wish_tr    = pd.read_csv(PATHS["wish_train"])
    wish_te    = pd.read_csv(PATHS["wish_test"])
    book_tags  = pd.read_csv(PATHS["book_tags"])
    tags_df    = pd.read_csv(PATHS["tags"])

    # build dictionaries
    author2idx = {a:i+1 for i,a in enumerate(sorted(books.authors.unique()))}
    lang2idx   = {l:i+1 for i,l in enumerate(books.language_code.fillna('unk').unique())}
    book_author = books.set_index('book_id').authors.map(author2idx).fillna(0).astype(int).to_dict()
    book_lang   = books.set_index('book_id').language_code.fillna('unk').map(lang2idx).fillna(0).astype(int).to_dict()
    book_dense  = books.set_index('book_id')[['ratings_count','average_rating']].astype(float).to_dict('index')
    all_books_np= books.book_id.values.astype(np.int32)

    top_tags = {}
    books_by_tag = defaultdict(list)
    for bid,grp in book_tags.groupby('book_id'):
        tg = (grp.sort_values('count',ascending=False).tag_id.tolist()+[0]*5)[:5]
        top_tags[bid]=tg
        for t in tg:
            if t: books_by_tag[t].append(bid)

    ratings_full = ratings_df.groupby('user_id').apply(lambda d: dict(zip(d.book_id,d.rating))).to_dict()
    ratings_by_user = ratings_df.groupby('user_id').book_id.apply(list).to_dict()
    wish_train = wish_tr.groupby('user_id').book_id.apply(list).to_dict()
    wish_test  = wish_te.groupby('user_id').book_id.apply(list).to_dict()

    num_authors = len(author2idx) + 1
    num_langs   = len(lang2idx)   + 1
    num_tags    = int(tags_df.tag_id.max()) + 1

    # correct 7-field pattern: [author, lang, tag×5]
    single_counts = [num_authors, num_langs] + [num_tags]*5
    counts = np.array(single_counts * (MAX_HIST_LEN*2 + 1), dtype=np.int64)


    glob= dict(book_author=book_author,book_lang=book_lang,book_dense=book_dense,
               top_tags=top_tags,ratings_by_user=ratings_by_user,ratings_full=ratings_full,
               all_books_np=all_books_np,books_by_tag=books_by_tag,
               K_NEG=K_NEG,MAX_HIST_LEN=MAX_HIST_LEN,
               PAD_BOOK=PAD_BOOK,PAD_SPARSE_VEC=PAD_SPARSE_VEC,PAD_DENSE_VEC=PAD_DENSE_VEC)

    def _build(wish_map,label):
        Xi,Xc,Y=[],[],[]
        with mp.Pool(CPU_WORKERS,initializer=_init_pool,initargs=(glob,)) as pool:
            for x_int,x_den,y in tqdm(pool.imap_unordered(build_rows,wish_map.items()),total=len(wish_map),desc=f"Building {label}"):
                Xi.append(x_int); Xc.append(x_den); Y.append(y)
        return np.vstack(Xi),np.vstack(Xc),np.vstack(Y)

    X_int_tr,X_den_tr,y_tr=_build(wish_train,'train')
    X_int_te,X_den_te,y_te=_build(wish_test ,'test' )

    os.makedirs(os.path.dirname(PATHS["train_npz"]),exist_ok=True)
    np.savez_compressed(PATHS["train_npz"],X_int=X_int_tr,X_dense=X_den_tr,y=y_tr,counts=counts)
    np.savez_compressed(PATHS["test_npz"] ,X_int=X_int_te,X_dense=X_den_te,y=y_te,counts=counts)

    # write tag counts json
    tag_counts_json = {str(t):{"0":int(tag_neg_freq[t]),"1":int(tag_pos_freq[t])}
                       for t in set(tag_neg_freq)|set(tag_pos_freq)}
    os.makedirs(os.path.dirname(PATHS["tag_json"]),exist_ok=True)
    with open(PATHS["tag_json"],"w",encoding="utf-8") as f:
        json.dump(tag_counts_json,f,indent=2)
    print(f"[Data] wrote tag counts → {PATHS['tag_json']}")

# ═══════════════ Dataset / Loader ═══════════════════════════════════
class GoodreadsNPZ(Dataset):
    def __init__(self,path,counts_ref=None):
        f=np.load(path)
        self.X_int=f["X_int"]; self.X_dense=f["X_dense"]; self.y=f["y"]
        self.counts=f["counts"] if "counts" in f.files else counts_ref
        self.num_sparse_features=self.X_int.shape[1]
        self.num_dense_features =self.X_dense.shape[1]
    def __len__(self): return len(self.y)
    def __getitem__(self,idx):
        return (self.X_int[idx].astype(np.int64),
                self.X_dense[idx].astype(np.float32),
                self.y[idx].astype(np.float32),
                )


def collate(batch):
    sp=torch.tensor([b[0] for b in batch],dtype=torch.long)
    de=torch.tensor([b[1] for b in batch],dtype=torch.float32)
    y =torch.tensor([b[2] for b in batch],dtype=torch.float32).view(-1,1)
    lS_i=[sp[:,i] for i in range(sp.shape[1])]
    lS_o=[torch.arange(de.size(0),dtype=torch.long)]*sp.shape[1]
    return de,lS_o,lS_i,y

# ═══════════════ main execution ═════════════════════════════════════
if __name__ == "__main__":
    if REGEN_NPZ or not (os.path.exists(PATHS["train_npz"]) and os.path.exists(PATHS["test_npz"])):
        regenerate_npz()

    train_ds=GoodreadsNPZ(PATHS["train_npz"])
    test_ds =GoodreadsNPZ(PATHS["test_npz"],counts_ref=train_ds.counts)
    num_sparse_features = train_ds.num_sparse_features
    num_dense_features  = train_ds.num_dense_features
    train_dl = DataLoader(train_ds,BATCH_SIZE,shuffle=True ,collate_fn=collate,num_workers=CPU_WORKERS//2)
    test_dl = DataLoader(test_ds ,BATCH_SIZE,shuffle=False,collate_fn=collate,num_workers=CPU_WORKERS//2)

    print("── Dataset shapes ──")
    print(f"train_ds   X_int : {train_ds.X_int.shape}  X_dense : {train_ds.X_dense.shape}  y : {train_ds.y.shape}")
    print(f"test_ds    X_int : {test_ds.X_int.shape}   X_dense : {test_ds.X_dense.shape}   y : {test_ds.y.shape}")


    ln_emb = train_ds.counts
    ln_bot = np.array([train_ds.num_dense_features,128,64,EMBED_DIM])
    nfeat = train_ds.num_sparse_features+1
    ln_top = np.array([EMBED_DIM+(nfeat*(nfeat-1)//2),256,128,1])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DLRM_Net(
        m_spa=EMBED_DIM,
        ln_emb=ln_emb,
        ln_bot=ln_bot,
        ln_top=ln_top,
        arch_interaction_op="dot",
        sigmoid_bot=-1,
        sigmoid_top=len(ln_top)-2,
        loss_function="bce",
        ndevices=1
    ).to(device)


    # ── sample-weight function (inverse pos freq) ────────────────────
    with open(PATHS["tag_json"],encoding="utf-8") as f:
        tag_json=json.load(f)
    pos_freq={int(t): max(1,v["1"]) for t,v in tag_json.items()}  # avoid zero

    def inv_pos_weight(bid_batch):
        w=[]
        for bid in bid_batch:
            tags=[t for t in top_tags.get(int(bid),[]) if t]
            inv=[1/pos_freq.get(t,1) for t in tags] or [1.0]
            w.append(sum(inv)/len(inv))
        return torch.tensor(w,dtype=torch.float32,device=device).view(-1,1)

    bce = nn.BCELoss()
    optimizer = torch.optim.Adagrad(model.parameters(),lr=LR)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("[Main] Optimizer: Adagrad")

    # --- Training Loop ---
    def run_epoch(epoch: int, train: bool):
        loader = train_dl if train else test_dl
        model.train(train) # Set model mode

        total_samples = 0
        total_loss = 0.0
        total_correct = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        pbar = tqdm(loader, desc=f"{'Train' if train else 'Test '} {epoch}", ncols=90, leave=False)

        with torch.set_grad_enabled(train): # Context manager for gradients
            for step, (dense_features, lS_o, lS_i, y_true) in enumerate(pbar):
                # Move data to device
                dense_features = dense_features.to(device)
                lS_i = [S_i.to(device) for S_i in lS_i]
                lS_o = [S_o.to(device) for S_o in lS_o] # Offsets might not need .to(device) if generated on CPU, check DLRM impl.
                y_true = y_true.to(device)

                # Forward pass
                y_pred = model(dense_features, lS_o, lS_i)

                # Calculate loss
                loss = bce(y_pred, y_true)

                if train:
                    optimizer.zero_grad(set_to_none=True) # More memory efficient
                    loss.backward()
                    optimizer.step()

                # --- Metrics Calculation ---
                batch_size = y_true.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                preds_binary = (y_pred > 0.5).float()
                total_correct += (preds_binary == y_true).sum().item()

                # F1 Score components
                tp = ((preds_binary == 1) & (y_true == 1)).sum().item()
                fp = ((preds_binary == 1) & (y_true == 0)).sum().item()
                fn = ((preds_binary == 0) & (y_true == 1)).sum().item()

                true_positives += tp
                false_positives += fp
                false_negatives += fn

                # Update progress bar
                if step > 0 and step % LOG_EVERY == 0:
                    avg_loss = total_loss / total_samples
                    avg_acc = total_correct / total_samples
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc*100:.2f}%")

        # --- Epoch End Metrics ---
        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        return epoch_loss, epoch_acc, precision, recall, f1_score

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
            "embedding_sizes":  ln_emb.tolist(),
            "bottom_mlp":       ln_bot.tolist(),
            "top_mlp":          ln_top.tolist(),
            "embed_dim":        EMBED_DIM,
            "num_sparse_features": num_sparse_features,
            "num_dense_features":  num_dense_features,
            "max_hist_len":        MAX_HIST_LEN,
        }, EXPORT)
        print(f"✅ Model and config saved to {EXPORT}")
    except Exception as e:
        print(f"Error saving model: {e}")