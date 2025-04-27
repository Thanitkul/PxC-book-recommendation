#!/usr/bin/env python3
# train_goodreads_dlrm_fixed.py
# -------------------------------------------------------------
# Train / test DLRM on the Goodreads NPZs we just generated.
# -------------------------------------------------------------
import os, sys, math, time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ------------------------------------------------------------------ #
# 1.  Bring the Facebook model definition into scope                 #
# ------------------------------------------------------------------ #
sys.path.insert(0, "dlrm")      # folder that has dlrm_s_pytorch.py
from dlrm_s_pytorch import DLRM_Net

# ------------------------------------------------------------------ #
# 2.  A very small NPZ-backed Dataset                                #
# ------------------------------------------------------------------ #
class GoodreadsNPZ(Dataset):
    def __init__(self, npz_path, counts_ref=None):
        self.f = np.load(npz_path)
        self.X_int, self.X_cat, self.y = (
            self.f["X_int"], self.f["X_cat"], self.f["y"]
        )
        self.counts = self.f["counts"] if "counts" in self.f else counts_ref
        assert self.counts is not None, "counts array must come from first file"
        assert self.X_int.shape[1] == len(self.counts) == 47
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return (
            self.X_int[idx].astype(np.int64),
            self.X_cat[idx].astype(np.float32),
            self.y[idx].astype(np.float32)
        )

def collate(batch):
    sparse = torch.tensor([b[0] for b in batch], dtype=torch.long)
    dense  = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    y      = torch.tensor([b[2] for b in batch], dtype=torch.float32).view(-1,1)

    lS_i = [ sparse[:,i] for i in range(sparse.shape[1]) ]
    offs = torch.arange(dense.shape[0], dtype=torch.long)
    lS_o = [ offs for _ in range(sparse.shape[1]) ]
    return dense, lS_o, lS_i, y

# ------------------------------------------------------------------ #
# 3.  Fixed training parameters                                      #
# ------------------------------------------------------------------ #
TRAIN_NPZ_PATH = "goodreads_0_reordered.npz"
TEST_NPZ_PATH  = "goodreads_1_reordered.npz"

BATCH_SIZE = 2048
EPOCHS     = 3
LR         = 1e-3
NUM_GPUS   = 2      # 0 = CPU
EMBED_DIM  = 16     # m_spa
LOG_EVERY  = 50

device = torch.device("cuda" if (NUM_GPUS and torch.cuda.is_available()) else "cpu")
print(f"\n[DLRM] Training on {device}")

# ------------------------------------------------------------------ #
# 4.  Load datasets & dataloaders                                    #
# ------------------------------------------------------------------ #
train_ds = GoodreadsNPZ(TRAIN_NPZ_PATH)
test_ds  = GoodreadsNPZ(TEST_NPZ_PATH, counts_ref=train_ds.counts)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=0, collate_fn=collate, drop_last=False)
test_dl  = DataLoader(test_ds , batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, collate_fn=collate, drop_last=False)

ln_emb = np.asarray(train_ds.counts, dtype=np.int64)          
m_spa  = EMBED_DIM                                          
ln_bot = np.array([2, 64, m_spa])                             
ln_top = np.array([(47+1)*m_spa , 64, 1])

# ------------------------------------------------------------------ #
# 5.  Build the model                                                #
# ------------------------------------------------------------------ #
model = DLRM_Net(
    m_spa=m_spa,
    ln_emb=ln_emb,
    ln_bot=ln_bot,
    ln_top=ln_top,
    arch_interaction_op="cat",
    sigmoid_bot=-1,
    sigmoid_top=ln_top.size-2,
    ndevices=NUM_GPUS if device.type=="cuda" else -1
).to(device)

if NUM_GPUS > 1 and device.type=="cuda":
    model = torch.nn.DataParallel(model)

crit = nn.BCEWithLogitsLoss()
opt  = optim.Adam(model.parameters(), lr=LR)

# ------------------------------------------------------------------ #
# 6.  Train & Evaluate                                               #
# ------------------------------------------------------------------ #
def run_epoch(epoch, train=True):
    loader = train_dl if train else test_dl
    model.train() if train else model.eval()
    total, correct, running_loss = 0, 0, 0.0
    pbar = tqdm(loader, desc=f"{'Train' if train else 'Test'} E{epoch}", ncols=100)
    with torch.set_grad_enabled(train):
        for i, (dense,lS_o,lS_i,y) in enumerate(pbar):
            dense, y = dense.to(device), y.to(device)
            lS_i = [t.to(device) for t in lS_i]
            lS_o = [t.to(device) for t in lS_o]

            out = model(dense, lS_o, lS_i)
            loss = crit(out, y)

            if train:
                opt.zero_grad(); loss.backward(); opt.step()

            running_loss += loss.item() * y.size(0)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y).sum().item()
            total   += y.size(0)

            if (i+1) % LOG_EVERY == 0:
                pbar.set_postfix(loss=running_loss/total,
                                 acc=100.*correct/total)
    return running_loss/total, correct/total

for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc = run_epoch(ep, train=True)
    te_loss, te_acc = run_epoch(ep, train=False)
    print(f"\nEpoch {ep}: "
          f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
          f"test loss {te_loss:.4f} acc {te_acc*100:.2f}%\n")

print("Done ✅")
