import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Set visible device(s) as needed
import sys
sys.path.insert(0, "dlrm/")  # folder containing dlrm_s_pytorch.py

from dlrm_s_pytorch import DLRM_Net

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from tqdm import tqdm

# For BERT embedding from text:
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
# 1. Callback to Plot Training Loss
###############################################################################
class LossPlotterCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.steps = []
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        else:
            loss = outputs
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        global_step = trainer.global_step
        self.losses.append(loss)
        self.steps.append(global_step)
        if global_step % 50 == 0:
            print(f"Step {global_step}: Training Loss = {loss}")
    def on_train_end(self, trainer, pl_module):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(self.steps, self.losses, label="Training Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss over Steps")
        plt.legend()
        plt.savefig("training_loss.png")
        plt.close()
        print("Training loss plot saved as 'training_loss.png'.")


###############################################################################
# 2. BPR Dataset
#    - Builds user->pos pairs from top-5
#    - Randomly samples a negative item not in top-5
###############################################################################
class BPRDataset(Dataset):
    def __init__(self, ratings_df, user_top5_books, all_books, item_meta, user_books_count):
        """
        ratings_df: full DataFrame with user/book info
        user_top5_books: dict of user_id -> list of top-5 book_ids
        all_books: a list or set of all candidate book_ids
        item_meta: dict of book_id -> (average_rating, language_code)
        user_books_count: dict of user_id -> float(books_count)
        """
        super().__init__()
        self.ratings_df = ratings_df
        self.user_top5_books = user_top5_books
        self.all_books = list(all_books)  # convert to list for np.random.choice
        self.item_meta = item_meta
        self.user_books_count = user_books_count
        self.user_pos_pairs = []
        for user, top5_list in user_top5_books.items():
            for b in top5_list:
                self.user_pos_pairs.append((user, b))
    def __len__(self):
        return len(self.user_pos_pairs)
    def __getitem__(self, idx):
        user_id, pos_id = self.user_pos_pairs[idx]
        while True:
            neg_id = np.random.choice(self.all_books)
            if neg_id not in self.user_top5_books.get(user_id, []):
                break
        return user_id, pos_id, neg_id


###############################################################################
# 3. Collate Function for BPR
#    - Builds dense features for positive/negative items
###############################################################################
def bpr_collate_fn(batch):
    """
    Each element in batch is (user_id, pos_item_id, neg_item_id).
    Returns:
       user_ids, pos_dense, pos_item_ids, neg_dense, neg_item_ids
    """
    user_ids = []
    pos_item_ids = []
    neg_item_ids = []
    pos_dense_list = []
    neg_dense_list = []
    for (u, i, j) in batch:
        user_ids.append(u)
        pos_item_ids.append(i)
        neg_item_ids.append(j)
        # Build 3-dim numeric features for positive item
        bc = user_books_count.get(u, 0.0)
        avg_rating_i, lang_i = item_meta.get(i, (3.5, "en"))
        lang_idx_i = float(lang2idx.get(lang_i, 0))
        pos_dense_list.append([bc, lang_idx_i, avg_rating_i])
        # Build 3-dim numeric features for negative item
        avg_rating_j, lang_j = item_meta.get(j, (3.5, "en"))
        lang_idx_j = float(lang2idx.get(lang_j, 0))
        neg_dense_list.append([bc, lang_idx_j, avg_rating_j])
    pos_dense = torch.tensor(pos_dense_list, dtype=torch.float32)
    neg_dense = torch.tensor(neg_dense_list, dtype=torch.float32)
    return user_ids, pos_dense, pos_item_ids, neg_dense, neg_item_ids


###############################################################################
# 4. BPR LightningModule
###############################################################################
class BPRLightningModule(pl.LightningModule):
    def __init__(self, dlrm_model, lr=1e-3):
        super().__init__()
        self.model = dlrm_model
        self.lr = lr
    def bpr_loss(self, pos_score, neg_score):
        diff = pos_score - neg_score
        return -torch.log(torch.sigmoid(diff) + 1e-10).mean()
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    def training_step(self, batch, batch_idx):
        user_ids, pos_dense, pos_item_ids, neg_dense, neg_item_ids = batch
        pos_dense = pos_dense.to(self.device)
        neg_dense = neg_dense.to(self.device)
        Xpos = make_dlrm_input_bpr(pos_dense, pos_item_ids, user_ids)
        Xneg = make_dlrm_input_bpr(neg_dense, neg_item_ids, user_ids)
        pos_logits = self.model.forward(Xpos, [], []).view(-1)
        neg_logits = self.model.forward(Xneg, [], []).view(-1)
        loss = self.bpr_loss(pos_logits, neg_logits)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        user_ids, pos_dense, pos_item_ids, neg_dense, neg_item_ids = batch
        pos_dense = pos_dense.to(self.device)
        neg_dense = neg_dense.to(self.device)
        Xpos = make_dlrm_input_bpr(pos_dense, pos_item_ids, user_ids)
        Xneg = make_dlrm_input_bpr(neg_dense, neg_item_ids, user_ids)
        pos_logits = self.model.forward(Xpos, [], []).view(-1)
        neg_logits = self.model.forward(Xneg, [], []).view(-1)
        loss = self.bpr_loss(pos_logits, neg_logits)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss


###############################################################################
# 5. DataModule for BPR
###############################################################################
class BPRDataModule(pl.LightningDataModule):
    def __init__(self, bpr_dataset, batch_size=2048, num_workers=0):
        super().__init__()
        self.bpr_dataset = bpr_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
    def setup(self, stage=None):
        dataset_size = len(self.bpr_dataset)
        val_size = int(0.1 * dataset_size)
        train_size = dataset_size - val_size
        indices = torch.randperm(dataset_size).tolist()
        self.train_dataset = torch.utils.data.Subset(self.bpr_dataset, indices[:train_size])
        self.val_dataset = torch.utils.data.Subset(self.bpr_dataset, indices[train_size:])
        print(f"BPR dataset split: {train_size} train, {val_size} val.")
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=True,
                          collate_fn=bpr_collate_fn)
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=bpr_collate_fn)


###############################################################################
# 6. Function to Build DLRM Input for BPR
###############################################################################
def make_dlrm_input_bpr(candidate_dense_batch, item_ids_batch, user_ids_batch):
    """
    Build a 2307-dim input vector:
      cat(
        candidate_dense [B, 3],
        item_embedding   [B, 768],
        rated_embedding  [B, 768],
        wishlist_embedding [B, 768]
      )
    Returns tensor of shape [B, 2307]
    """
    if isinstance(user_ids_batch, torch.Tensor):
        user_ids_batch = user_ids_batch.cpu().tolist()
    B = candidate_dense_batch.size(0)
    item_embeddings = []
    for bid in item_ids_batch:
        if bid not in item_cache:
            text = get_item_text(bid)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.cuda.amp.autocast(), torch.no_grad():
                outputs = bert_model(**inputs)
            item_cache[bid] = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        emb = item_cache[bid].to(device)
        item_embeddings.append(emb.unsqueeze(0))
    item_embeddings = torch.cat(item_embeddings, dim=0)
    rated_embeddings = []
    wishlist_embeddings = []
    for uid in user_ids_batch:
        re = rated_cache[uid].to(device)  # (768,)
        we = wishlist_cache[uid].to(device)  # (768,)
        rated_embeddings.append(re.unsqueeze(0))
        wishlist_embeddings.append(we.unsqueeze(0))
    rated_embeddings = torch.cat(rated_embeddings, dim=0)
    wishlist_embeddings = torch.cat(wishlist_embeddings, dim=0)
    X = torch.cat([candidate_dense_batch, item_embeddings, rated_embeddings, wishlist_embeddings], dim=1)
    return X


###############################################################################
# Main Script
###############################################################################
if __name__ == "__main__":
    ##############################################
    # Step A: Read CSV files
    ##############################################
    print("Reading CSV files...")
    books_df     = pd.read_csv("../data-prep-EDA/clean/books.csv")
    ratings_df   = pd.read_csv("../data-prep-EDA/clean/ratings_train.csv")
    tags_df      = pd.read_csv("../data-prep-EDA/clean/tags.csv")
    book_tags_df = pd.read_csv("../data-prep-EDA/clean/book_tags.csv")
    to_read_df   = pd.read_csv("../data-prep-EDA/clean/to_read.csv")
    print("Finished reading CSV files.\n")

    ##############################################
    # Step B: Build dictionaries (authors, lang, etc.)
    ##############################################
    print("Building dictionaries for candidate features...")
    unique_lang = books_df["language_code"].fillna("Unknown").unique().tolist()
    lang2idx = {l: i for i, l in enumerate(unique_lang)}
    print("Finished building dictionaries.\n")

    ##############################################
    # Step C: Identify top-5 books per user
    ##############################################
    ratings_df = ratings_df.sort_values(["user_id", "rating"], ascending=[True, False])
    ratings_df["rank_desc"] = ratings_df.groupby("user_id")["rating"].rank(method="first", ascending=False)
    top5_df = ratings_df[ratings_df["rank_desc"] <= 5]
    user_top5_books = top5_df.groupby("user_id")["book_id"].apply(list).to_dict()

    ##############################################
    # Step D: Build item_meta and user_books_count for BPR
    ##############################################
    meta_small = books_df[["book_id", "average_rating", "language_code"]].drop_duplicates("book_id")
    item_meta = {}
    for row in meta_small.itertuples():
        item_meta[row.book_id] = (float(row.average_rating), row.language_code)
    user_books_count = ratings_df.groupby("user_id")["book_id"].nunique().to_dict()
    for u in user_books_count:
        user_books_count[u] = float(user_books_count[u])
    all_books = books_df["book_id"].unique().tolist()

    ##############################################
    # Step E: Prepare user embeddings text (exclude top5)
    ##############################################
    def build_user_embedding_lists(ratings_df, to_read_df, user_top5_books):
        user_rated_dict = ratings_df.groupby("user_id")["book_id"].apply(list).to_dict()
        user_wishlist_dict = to_read_df.groupby("user_id")["book_id"].apply(list).to_dict()
        user_rated_books_embed = {}
        for user, books in user_rated_dict.items():
            top5 = set(user_top5_books.get(user, []))
            filtered = [b for b in books if b not in top5][:50]
            user_rated_books_embed[user] = filtered
        user_wishlist_books_embed = {}
        for user, books in user_wishlist_dict.items():
            top5 = set(user_top5_books.get(user, []))
            filtered = [b for b in books if b not in top5][:50]
            user_wishlist_books_embed[user] = filtered
        return user_rated_books_embed, user_wishlist_books_embed

    user_rated_books_embed, user_wishlist_books_embed = build_user_embedding_lists(ratings_df, to_read_df, user_top5_books)

    # Print BPR Dataset summary
    total_users = len(user_top5_books)
    total_positives = sum(len(top5) for top5 in user_top5_books.values())
    total_negatives = total_positives  # one negative per positive
    print("==== BPR Dataset Summary ====")
    print(f"Total users          : {total_users}")
    print(f"Total positive pairs : {total_positives}")
    print(f"Total negative pairs : {total_negatives}")
    print(f"Total samples        : {total_positives} (each with 1 negative)")
    print("==============================\n")

    ##############################################
    # Step F: Create BERT text dict for each book
    ##############################################
    tag_id2name = {row.tag_id: row.tag_name for row in tags_df.itertuples()}
    def get_book_tags_text(book_id, top_k=5):
        sub = book_tags_df[book_tags_df["book_id"] == book_id]
        if sub.empty:
            return ""
        sub = sub.sort_values("count", ascending=False)
        tag_ids = sub["tag_id"].tolist()[:top_k]
        tag_names = [tag_id2name.get(tid, "") for tid in tag_ids]
        return ", ".join(tag_names)
    book_text_dict = {}
    for row in books_df.itertuples():
        bid = row.book_id
        title = row.title
        author = row.authors
        language = row.language_code
        avg_rating = row.average_rating
        tags_text = get_book_tags_text(bid, top_k=5)
        text = f"Title: {title}. Author: {author}. Language: {language}. Average Rating: {avg_rating}. Tags: {tags_text}."
        book_text_dict[bid] = text
    def get_item_text(bid):
        return book_text_dict.get(bid, "")

    ##############################################
    # Step G: Setup BERT
    ##############################################
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    bert_model.eval()
    # Global caches (declared globally so functions can use them)
    item_cache = {}
    rated_cache = {}
    wishlist_cache = {}

    def get_user_rated_text(uid):
        rated_list = user_rated_books_embed.get(uid, [])
        texts = [book_text_dict.get(bid, "") for bid in rated_list]
        return " [SEP] ".join(texts)
    def get_user_wishlist_text(uid):
        wishlist_list = user_wishlist_books_embed.get(uid, [])
        texts = [book_text_dict.get(bid, "") for bid in wishlist_list]
        return " [SEP] ".join(texts)

    ##############################################
    # Step H: Precompute BERT Embeddings
    ##############################################
    print("Precomputing BERT embeddings for items and users...")
    # Items
    item_cache_file = "item_cache.pt"
    if os.path.exists(item_cache_file):
        item_cache = torch.load(item_cache_file)
        print("Loaded item embeddings from file.")
    else:
        for bid in tqdm(all_books, desc="Precompute Items"):
            text = get_item_text(bid)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.cuda.amp.autocast(), torch.no_grad():
                outputs = bert_model(**inputs)
            item_cache[bid] = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        torch.save(item_cache, item_cache_file)
        print("Saved item embeddings.\n")
    # Users
    rated_cache_file = "rated_cache.pt"
    wishlist_cache_file = "wishlist_cache.pt"
    all_users = sorted(set(ratings_df["user_id"].unique()))
    if os.path.exists(rated_cache_file) and os.path.exists(wishlist_cache_file):
        rated_cache = torch.load(rated_cache_file)
        wishlist_cache = torch.load(wishlist_cache_file)
        print("Loaded user rated/wishlist embeddings from file.")
    else:
        def batch_precompute_users(user_list, text_fn, cache_dict):
            for i in tqdm(range(0, len(user_list), 256), desc="Batch Precompute"):
                batch_uids = user_list[i : i+256]
                texts = [text_fn(u) for u in batch_uids]
                inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.cuda.amp.autocast(), torch.no_grad():
                    outputs = bert_model(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu()
                for j, uid in enumerate(batch_uids):
                    cache_dict[uid] = batch_emb[j]
        batch_precompute_users(all_users, get_user_rated_text, rated_cache)
        batch_precompute_users(all_users, get_user_wishlist_text, wishlist_cache)
        torch.save(rated_cache, rated_cache_file)
        torch.save(wishlist_cache, wishlist_cache_file)
        print("Saved user rated/wishlist embeddings.\n")
    print("Done precomputing embeddings.\n")

    ##############################################
    # Step I: Build the BPR Dataset & DataModule
    ##############################################
    bpr_dataset = BPRDataset(
        ratings_df=ratings_df,
        user_top5_books=user_top5_books,
        all_books=all_books,
        item_meta=item_meta,
        user_books_count=user_books_count,
    )
    bpr_datamodule = BPRDataModule(bpr_dataset, batch_size=2048, num_workers=4)

    ##############################################
    # Step J: Build DLRM
    ##############################################
    ln_emb = np.array([])  # No sparse features
    m_spa = 16
    ln_bot = np.array([2307, 512, 128])  # 2307 = 3 + 768 + 768 + 768
    ln_top = np.array([128, 64, 1])
    dlrm = DLRM_Net(
        m_spa=m_spa,
        ln_emb=ln_emb,
        ln_bot=ln_bot,
        ln_top=ln_top,
        arch_interaction_op="cat",
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        md_flag=False,
    ).to(device)

    ##############################################
    # Step K: Create the BPR LightningModule & Train
    ##############################################
    bpr_model = BPRLightningModule(dlrm, lr=1e-5)
    loss_plotter_callback = LossPlotterCallback()
    trainer = Trainer(
        max_epochs=100,   # adjust as needed
        accelerator="gpu",
        devices=2,      # or 1 if only one GPU
        callbacks=[loss_plotter_callback]
    )
    trainer.fit(bpr_model, datamodule=bpr_datamodule)
    trainer.save_checkpoint("dlrm_bpr_checkpoint.ckpt")
    print("Training complete. Model checkpoint saved as 'dlrm_bpr_checkpoint.ckpt'.")
