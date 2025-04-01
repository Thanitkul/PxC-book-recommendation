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
        if global_step % 10 == 0:
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
# 2. Dataset for Binary Classification using Wishlist as Label
#    - For each user, use wishlist books (from to_read.csv) as positive samples.
#    - For each positive sample, randomly select a negative sample from books not in the wishlist.
###############################################################################
class BinaryDataset(Dataset):
    def __init__(self, user_wishlist_books, all_books, item_meta, user_books_count, user_wishlist_books_set):
        """
        user_wishlist_books: dict of user_id -> list of wishlist book_ids (positive samples)
        all_books: list of all candidate book_ids
        item_meta: dict of book_id -> (average_rating, language_code)
        user_books_count: dict of user_id -> float (number of books rated)
        user_wishlist_books_set: dict of user_id -> set of wishlist book_ids for the user
        """
        super().__init__()
        self.user_wishlist_books = user_wishlist_books
        # Build list of positive samples: each element is (user, wishlist_book)
        self.user_pos_samples = []
        for user, books in user_wishlist_books.items():
            for b in books:
                self.user_pos_samples.append((user, b))
        self.all_books = all_books
        self.item_meta = item_meta
        self.user_books_count = user_books_count
        self.user_wishlist_books_set = user_wishlist_books_set

    def __len__(self):
        # Each positive sample produces one positive and one negative sample.
        return 2 * len(self.user_pos_samples)

    def __getitem__(self, idx):
        pos_idx = idx // 2  # index in self.user_pos_samples
        user, pos_book = self.user_pos_samples[pos_idx]
        if idx % 2 == 0:
            # Positive sample from wishlist
            label = 1
            item_id = pos_book
        else:
            # Negative sample: sample a book not in the user's wishlist.
            wishlist_set = self.user_wishlist_books_set.get(user, set())
            candidate_books = list(set(self.all_books) - wishlist_set)
            item_id = np.random.choice(candidate_books)
            label = 0
        # Build 3-dim dense feature vector for the item
        bc = self.user_books_count.get(user, 0.0)
        avg_rating, lang = self.item_meta.get(item_id, (3.5, "en"))
        lang_idx = float(lang2idx.get(lang, 0))
        dense_features = [bc, lang_idx, avg_rating]
        return user, dense_features, item_id, label

###############################################################################
# 3. Collate Function for Binary Dataset
###############################################################################
def binary_collate_fn(batch):
    """
    Each element in batch is (user_id, dense_features, item_id, label).
    Returns:
       user_ids, dense_features_tensor, item_ids, labels_tensor
    """
    user_ids = []
    dense_list = []
    item_ids = []
    labels = []
    for (u, dense_features, item_id, label) in batch:
        user_ids.append(u)
        dense_list.append(dense_features)
        item_ids.append(item_id)
        labels.append(label)
    dense_tensor = torch.tensor(dense_list, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return user_ids, dense_tensor, item_ids, labels_tensor

###############################################################################
# 4. Binary LightningModule (using BCE Loss)
#    Updated make_dlrm_input: Exclude wishlist embedding from input.
###############################################################################
def make_dlrm_input(candidate_dense_batch, item_ids_batch, user_ids_batch):
    """
    Build a 1539-dim input vector:
      cat(
        candidate_dense [B, 3],
        item_embedding   [B, 768],
        rated_embedding  [B, 768]
      )
    Returns tensor of shape [B, 1539]
    """
    if isinstance(user_ids_batch, torch.Tensor):
        user_ids_batch = user_ids_batch.cpu().tolist()
    B = candidate_dense_batch.size(0)
    
    # Build item embeddings
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
    
    # Build rated embeddings with fallback for missing user IDs
    rated_embeddings = []
    for uid in user_ids_batch:
        if uid in rated_cache:
            re = rated_cache[uid].to(device)  # (768,)
        else:
            # Fallback: use a zero vector if the user rated embedding is missing.
            re = torch.zeros(768, device=device)
        rated_embeddings.append(re.unsqueeze(0))
    rated_embeddings = torch.cat(rated_embeddings, dim=0)
    
    # Concatenate to build final input
    X = torch.cat([candidate_dense_batch, item_embeddings, rated_embeddings], dim=1)
    return X


class BinaryLightningModule(pl.LightningModule):
    def __init__(self, dlrm_model, lr=1e-3):
        super().__init__()
        self.model = dlrm_model
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    def training_step(self, batch, batch_idx):
        user_ids, dense_features, item_ids, labels = batch
        dense_features = dense_features.to(self.device)
        labels = labels.to(self.device)
        X = make_dlrm_input(dense_features, item_ids, user_ids)
        logits = self.model.forward(X, [], []).view(-1)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        user_ids, dense_features, item_ids, labels = batch
        dense_features = dense_features.to(self.device)
        labels = labels.to(self.device)
        X = make_dlrm_input(dense_features, item_ids, user_ids)
        logits = self.model.forward(X, [], []).view(-1)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

###############################################################################
# 5. DataModule for Binary Dataset
###############################################################################
class BinaryDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=2048, num_workers=0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
    def setup(self, stage=None):
        dataset_size = len(self.dataset)
        val_size = int(0.1 * dataset_size)
        train_size = dataset_size - val_size
        indices = torch.randperm(dataset_size).tolist()
        self.train_dataset = torch.utils.data.Subset(self.dataset, indices[:train_size])
        self.val_dataset = torch.utils.data.Subset(self.dataset, indices[train_size:])
        print(f"Dataset split: {train_size} train, {val_size} val.")
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=True,
                          collate_fn=binary_collate_fn)
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=binary_collate_fn)

###############################################################################
# Main Script
###############################################################################
if __name__ == "__main__":
    ##############################################
    # Step A: Read CSV files
    ##############################################
    print("Reading CSV files...")
    books_df     = pd.read_csv("../data-prep-EDA/clean/books.csv")
    ratings_df   = pd.read_csv("../data-prep-EDA/clean/ratings.csv")
    tags_df      = pd.read_csv("../data-prep-EDA/clean/tags.csv")
    book_tags_df = pd.read_csv("../data-prep-EDA/clean/book_tags.csv")
    to_read_df   = pd.read_csv("../data-prep-EDA/clean/to_read_train.csv")
    print("Finished reading CSV files.\n")

    ##############################################
    # Step B: Build dictionaries (authors, lang, etc.)
    ##############################################
    print("Building dictionaries for candidate features...")
    unique_lang = books_df["language_code"].fillna("Unknown").unique().tolist()
    lang2idx = {l: i for i, l in enumerate(unique_lang)}
    print("Finished building dictionaries.\n")

    ##############################################
    # Step C: Identify wishlist books per user and build wishlist set
    ##############################################
    # Use wishlist from to_read_df as positive labels
    user_wishlist_books = to_read_df.groupby("user_id")["book_id"].apply(list).to_dict()
    user_wishlist_books_set = {user: set(books) for user, books in user_wishlist_books.items()}

    ##############################################
    # Step D: Build item_meta and user_books_count for Binary Loss
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
    # Step E: Prepare user embeddings text (exclude wishlist books from input)
    ##############################################
    def build_user_embedding_lists(ratings_df, to_read_df, user_wishlist_books):
        user_rated_dict = ratings_df.groupby("user_id")["book_id"].apply(list).to_dict()
        # Exclude wishlist books from the rated list
        user_rated_books_embed = {}
        for user, books in user_rated_dict.items():
            wishlist = set(user_wishlist_books.get(user, []))
            filtered = [b for b in books if b not in wishlist][:50]
            user_rated_books_embed[user] = filtered
        # Since wishlist is now used as label, we do not use it in input.
        user_wishlist_books_embed = {user: [] for user in user_wishlist_books.keys()}
        return user_rated_books_embed, user_wishlist_books_embed

    user_rated_books_embed, _ = build_user_embedding_lists(ratings_df, to_read_df, user_wishlist_books)

    # Print Binary Dataset summary
    total_users = len(user_wishlist_books)
    total_positive_samples = sum(len(books) for books in user_wishlist_books.values())
    total_samples = total_positive_samples * 2  # each positive paired with a negative
    print("==== Binary Dataset Summary ====")
    print(f"Total users           : {total_users}")
    print(f"Total positive samples: {total_positive_samples}")
    print(f"Total negative samples: {total_positive_samples}")
    print(f"Total samples         : {total_samples} (1:1 ratio)")
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
    # Note: wishlist_cache is no longer used for input.
    def get_user_rated_text(uid):
        rated_list = user_rated_books_embed.get(uid, [])
        texts = [book_text_dict.get(bid, "") for bid in rated_list]
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
    # Users (only rated embeddings are needed)
    rated_cache_file = "rated_cache.pt"
    all_users = sorted(set(ratings_df["user_id"].unique()))
    if os.path.exists(rated_cache_file):
        rated_cache = torch.load(rated_cache_file)
        print("Loaded user rated embeddings from file.")
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
        torch.save(rated_cache, rated_cache_file)
        print("Saved user rated embeddings.\n")
    print("Done precomputing embeddings.\n")

    ##############################################
    # Step I: Build the Binary Dataset & DataModule
    ##############################################
    binary_dataset = BinaryDataset(
        user_wishlist_books=user_wishlist_books,
        all_books=all_books,
        item_meta=item_meta,
        user_books_count=user_books_count,
        user_wishlist_books_set=user_wishlist_books_set,
    )
    binary_datamodule = BinaryDataModule(binary_dataset, batch_size=2048, num_workers=4)

    ##############################################
    # Step J: Build DLRM Model
    ##############################################
    ln_emb = np.array([])  # No sparse features
    m_spa = 16
    # Update ln_bot dimension: 3 (dense) + 768 (item) + 768 (rated) = 1539
    ln_bot = np.array([1539, 512, 128])
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
    # Step K: Create the Binary LightningModule & Train
    ##############################################
    binary_model = BinaryLightningModule(dlrm, lr=1e-5)
    loss_plotter_callback = LossPlotterCallback()
    trainer = Trainer(
        max_epochs=1,   # adjust as needed
        accelerator="gpu",
        devices=2,      # or 1 if only one GPU
        callbacks=[loss_plotter_callback]
    )
    trainer.fit(binary_model, datamodule=binary_datamodule)
    trainer.save_checkpoint("dlrm_binary_checkpoint.ckpt")
    print("Training complete. Model checkpoint saved as 'dlrm_binary_checkpoint.ckpt'.")
