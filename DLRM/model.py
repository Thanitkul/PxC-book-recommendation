import os, sys, json, math, multiprocessing as mp
import numpy as np, pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ══════════════════ CONFIG ══════════════════════════════════════════
DATA_ROOT   = "../data-prep-EDA/clean"
K_NEG       = 10               # negatives per positive
EMBED_DIM   = 32
BATCH_SIZE  = 2048
EPOCHS      = 3
LR          = 3e-4
WEIGHT_DECAY= 1e-6
NUM_GPUS    = 1
CPU_WORKERS = 60
LOG_EVERY   = 50
REGEN_NPZ   = True           # force rebuild if True

PATHS = {
    "books"      : f"{DATA_ROOT}/books.csv",
    "ratings"    : f"{DATA_ROOT}/ratings.csv",
    "wish_train" : f"{DATA_ROOT}/to_read_train.csv",
    "wish_test"  : f"{DATA_ROOT}/to_read_test.csv",
    "book_tags"  : f"{DATA_ROOT}/book_tags.csv",
    "tags"       : f"{DATA_ROOT}/tags.csv",
    "train_npz"  : "data/goodreads_0_reordered.npz",
    "test_npz"   : "data/goodreads_1_reordered.npz",
}

# ═══════════════  bring FB DLRM  ════════════════════════════════════
sys.path.insert(0, "dlrm")
from dlrm_s_pytorch import DLRM_Net

# ──  derive dataset-level stats once ────────────────────────────────
_books_df         = pd.read_csv(PATHS["books"])
MAX_RATINGS_COUNT = float(_books_df["ratings_count"].max() or 1.0)
del _books_df

PAD_BOOK = 0
PAD_VEC7 = [0] * 7

# globals hydrated during NPZ generation
book_author = book_lang = book_dense = top_tags = None

# ══════════════════ helper functions ═══════════════════════════════
def sparse_vec7(bid: int):
    if bid == 0:
        return PAD_VEC7
    return [
        book_author.get(bid, 0),
        book_lang.get(bid, 0),
        *top_tags.get(bid, [0, 0, 0, 0, 0])[:5],
    ]


def dense_vec2(bid: int):
    if bid == 0:
        return [0.0, 0.0]
    rec = book_dense.get(bid, {"ratings_count": 0.0, "average_rating": 0.0})
    ratings_norm = math.log1p(rec["ratings_count"]) / math.log1p(MAX_RATINGS_COUNT)
    rating_norm  = (rec["average_rating"] - 1.0) / 4.0      # 1-5 → 0-1
    return [min(1.0, ratings_norm), rating_norm]


# ── multiprocessing helpers ────────────────────────────────────────
def _init_pool(glob_dict):  # put shared dicts in worker globals
    globals().update(glob_dict)


def build_rows(pair):
    uid, wish = pair
    rated = ratings_by_user.get(uid, [])[:20]
    wish  = wish[:20]

    rated_ids = rated + [PAD_BOOK] * (20 - len(rated))
    wish_pad  = wish  + [PAD_BOOK] * (20 - len(wish))

    user_auth = {book_author[b] for b in rated + wish}
    user_tags = set().union(*[set(top_tags.get(b, [])) for b in rated + wish])

    Xi, Xc, Y = [], [], []
    for pos in wish:
        wish_ids = wish_pad[:20]
        base_sparse = rated_ids + wish_ids


        # POSITIVE
        Xi.append(base_sparse + [pos] + sparse_vec7(pos))
        Xc.append(dense_vec2(pos))
        Y.append([1])

        # k-NEGATIVES
        negs = 0
        while negs < K_NEG:
            neg = int(np.random.choice(all_books_np))
            if (
                (neg in rated) or (neg in wish) or
                (book_author[neg] in user_auth) or
                (not set(top_tags.get(neg, [])).isdisjoint(user_tags))
            ):
                continue
            Xi.append(base_sparse + [neg] + sparse_vec7(neg))
            Xc.append(dense_vec2(neg))
            Y.append([0])
            negs += 1
    return Xi, Xc, Y


# ════════════════ NPZ generation ════════════════════════════════════
def regenerate_npz():
    global book_author, book_lang, book_dense, top_tags
    print("\n[PRE] Loading CSVs …")
    books      = pd.read_csv(PATHS["books"])
    ratings    = pd.read_csv(PATHS["ratings"])
    wish_train = pd.read_csv(PATHS["wish_train"])
    wish_test  = pd.read_csv(PATHS["wish_test"])
    book_tags  = pd.read_csv(PATHS["book_tags"])
    tags_df    = pd.read_csv(PATHS["tags"])

    author2idx = {a: i + 1 for i, a in enumerate(sorted(books.authors.unique()))}
    lang2idx   = {l: i + 1 for i, l in enumerate(books.language_code.fillna("unk").unique())}
    num_items   = books.book_id.max() + 1
    num_authors = len(author2idx) + 1
    num_langs   = len(lang2idx) + 1
    num_tags    = int(tags_df.tag_id.max()) + 1

    top_tags = {
        bid: (grp.sort_values("count", ascending=False).tag_id.tolist() + [0] * 5)[:5]
        for bid, grp in book_tags.groupby("book_id")
    }

    book_author = books.set_index("book_id").authors.map(author2idx.get).fillna(0).astype(int).to_dict()
    book_lang   = books.set_index("book_id").language_code.fillna("unk").map(lang2idx.get).fillna(0).astype(int).to_dict()
    book_dense  = books.set_index("book_id")[["ratings_count", "average_rating"]].astype(float).to_dict("index")

    global ratings_by_user, all_books_np
    ratings_by_user   = ratings.groupby("user_id").book_id.apply(list).to_dict()
    train_wish_by_usr = wish_train.groupby("user_id").book_id.apply(list).to_dict()
    test_wish_by_usr  = wish_test.groupby("user_id").book_id.apply(list).to_dict()
    all_books_np      = books.book_id.values.astype(np.int32)

    pool_globals = dict(
        ratings_by_user=ratings_by_user,
        book_author=book_author,
        book_lang=book_lang,
        book_dense=book_dense,
        top_tags=top_tags,
        all_books_np=all_books_np,
        PAD_BOOK=PAD_BOOK,
        K_NEG=K_NEG,
    )

    def _build(wish_dict, label):
        with mp.Pool(CPU_WORKERS, initializer=_init_pool, initargs=(pool_globals,)) as pool:
            Xi, Xc, Y = [], [], []
            for a, b, c in tqdm(pool.imap_unordered(build_rows, wish_dict.items()),
                                total=len(wish_dict), desc=label):
                Xi.extend(a), Xc.extend(b), Y.extend(c)
        return np.asarray(Xi, np.int32), np.asarray(Xc, np.float32), np.asarray(Y, np.float32)

    print("[PRE] Generating TRAIN rows …")
    X_int_tr, X_cat_tr, y_tr = _build(train_wish_by_usr, "train-users")

    print("[PRE] Generating TEST rows …")
    X_int_te, X_cat_te, y_te = _build(test_wish_by_usr, "test-users")

    counts = np.array([num_items] * 41 + [num_authors, num_langs] + [num_tags] * 5, dtype=np.int64)
    assert len(counts) == 48, "counts length must equal #sparse features"

    os.makedirs(os.path.dirname(PATHS["train_npz"]), exist_ok=True)
    np.savez_compressed(PATHS["train_npz"], X_int=X_int_tr, X_cat=X_cat_tr, y=y_tr, counts=counts)
    np.savez_compressed(PATHS["test_npz"],  X_int=X_int_te, X_cat=X_cat_te, y=y_te, counts=counts)
    print(f"[PRE] NPZs written: {PATHS['train_npz']} ({len(y_tr)}) | {PATHS['test_npz']} ({len(y_te)})")


# ═══════════════ Dataset / Loader ═══════════════════════════════════
class GoodreadsNPZ(Dataset):
    def __init__(self, path, counts_ref=None):
        f           = np.load(path)
        self.X_int  = f["X_int"]
        self.X_cat  = f["X_cat"]
        self.y      = f["y"]
        self.counts = f["counts"] if "counts" in f.files else counts_ref
        assert self.X_int.shape[1] == 48, "expected 48 sparse features"

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_int[idx].astype(np.int64),
            self.X_cat[idx].astype(np.float32),
            self.y[idx].astype(np.float32),
        )


def collate(batch):
    sparse = torch.tensor([b[0] for b in batch], dtype=torch.long)
    dense  = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    y      = torch.tensor([b[2] for b in batch], dtype=torch.float32).view(-1, 1)
    lS_i   = [sparse[:, i] for i in range(48)]
    offs   = torch.arange(dense.size(0), dtype=torch.long)
    lS_o   = [offs] * 48
    return dense, lS_o, lS_i, y


# ═══════════════ main execution ═════════════════════════════════════
if __name__ == "__main__":
    if REGEN_NPZ or not (os.path.exists(PATHS["train_npz"]) and os.path.exists(PATHS["test_npz"])):
        regenerate_npz()
    else:
        print("[PRE] Using cached NPZs — skip generation.")

    train_ds = GoodreadsNPZ(PATHS["train_npz"])
    test_ds  = GoodreadsNPZ(PATHS["test_npz"], counts_ref=train_ds.counts)

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate, num_workers=0)
    test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)

    ln_emb = np.asarray(train_ds.counts, dtype=np.int64)
    ln_bot = np.array([2, 64, EMBED_DIM])

    nsparse = len(ln_emb)            # should be 48
    m       = nsparse + 1           # include bottom‐MLP as a feature
    interact_dim = EMBED_DIM + (m * (m - 1) // 2)  # = 32 + (49*48/2) = 1208

    ln_top = np.array([interact_dim, 64, 1])



    device = torch.device("cuda" if NUM_GPUS and torch.cuda.is_available() else "cpu")
    print(f"\n[DLRM] Device: {device}")

    model = DLRM_Net(
        m_spa           = EMBED_DIM,
        ln_emb          = ln_emb,
        ln_bot          = ln_bot,
        ln_top          = ln_top,
        arch_interaction_op = "dot",
        sigmoid_bot     = -1,
        sigmoid_top     = len(ln_top) - 2,
        ndevices        = NUM_GPUS if device.type == "cuda" else -1,
    ).to(device)

    positive_weight = 100  # not K_NEG, better simulate deployment

    def weighted_bce_loss(output, target):
        weight = torch.where(target == 1, positive_weight, 1.0)
        bce = nn.BCELoss(reduction='none')
        loss = bce(output, target)
        weighted_loss = (loss * weight).mean()
        return weighted_loss

    optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)

    def run_epoch(epoch: int, train: bool):
        loader = train_dl if train else test_dl
        model.train() if train else model.eval()
        total = correct = loss_sum = 0.0
        true_positives = false_positives = false_negatives = 0


        pbar = tqdm(loader, desc=f"{'Train' if train else 'Test '} {epoch}", ncols=100)
        with torch.set_grad_enabled(train):
            for step, (dense, lS_o, lS_i, y) in enumerate(pbar, 1):
                dense, y = dense.to(device), y.to(device)
                lS_i     = [t.to(device) for t in lS_i]
                lS_o     = [t.to(device) for t in lS_o]

                out = model(dense, lS_o, lS_i)
                loss = weighted_bce_loss(out, y)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                loss_sum += loss.item() * y.size(0)
                preds    = (out > 0.5).float()
                tp = ((preds == 1) & (y == 1)).sum().item()
                fp = ((preds == 1) & (y == 0)).sum().item()
                fn = ((preds == 0) & (y == 1)).sum().item()

                true_positives += tp
                false_positives += fp
                false_negatives += fn

                correct += (preds == y).sum().item()
                total   += y.size(0)

                if step % LOG_EVERY == 0:
                    pbar.set_postfix(loss=loss_sum / total,
                                     acc=100 * correct / total)

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall    = true_positives / (true_positives + false_negatives + 1e-8)
        f1_score  = 2 * (precision * recall) / (precision + recall + 1e-8)

        return loss_sum / total, correct / total, precision, recall, f1_score


    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = run_epoch(ep, train=True)
        te_loss, te_acc, te_prec, te_rec, te_f1 = run_epoch(ep, train=False)
        print(f"\nEpoch {ep}: "
            f"Train loss {tr_loss:.4f}, acc {tr_acc*100:.2f}%, prec {tr_prec:.4f}, rec {tr_rec:.4f}, f1 {tr_f1:.4f} │ "
            f"Test loss {te_loss:.4f}, acc {te_acc*100:.2f}%, prec {te_prec:.4f}, rec {te_rec:.4f}, f1 {te_f1:.4f}")


    print("\nTraining complete ✅")

    EXPORT = "trained_dlrm_goodreads.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "embedding_sizes": ln_emb.tolist(),
        "bottom_mlp": ln_bot.tolist(),
        "top_mlp": ln_top.tolist(),
        "embed_dim": EMBED_DIM,
        "device": str(device),
    }, EXPORT)
    print(f"✅ Model saved to {EXPORT}")
