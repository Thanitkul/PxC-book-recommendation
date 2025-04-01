import os
os.makedirs("cache", exist_ok=True)  # Ensure the cache folder exists

import sys
sys.path.insert(0, "dlrm/")  # folder containing dlrm_s_pytorch.py

import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from tqdm import tqdm  # for progress bars

# Import your model definitions.
from model import BPRLightningModule, DLRM_Net

# For BERT embedding from text:
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
# nDCG Function: Graded relevance (1-5)
###############################################################################
def ndcg_at_k_graded(actual, predicted, k):
    """
    Computes nDCG@K using graded relevance scores.
    Only considers items with a non-zero actual rating.
    
    Parameters:
      actual (list or np.array): Ground truth relevance scores (e.g., 1 to 5; unrated=0).
      predicted (list or np.array): Predicted scores from the model.
      k (int): Rank cutoff.
      
    Returns:
      float: nDCG@K value.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    mask = actual > 0  # Only consider items that have been rated.
    if np.sum(mask) == 0:
        return 0.0
    actual = actual[mask]
    predicted = predicted[mask]
    order = np.argsort(-predicted)[:k]
    dcg = np.sum((2 ** actual[order] - 1) / np.log2(np.arange(2, len(order)+2)))
    ideal_order = np.argsort(-actual)[:k]
    idcg = np.sum((2 ** actual[ideal_order] - 1) / np.log2(np.arange(2, len(ideal_order)+2)))
    if idcg == 0:
        return 0.0
    return dcg / idcg

###############################################################################
# Global Objects and Helper Functions (same as during training)
###############################################################################

# Read CSV files.
books_df = pd.read_csv("../data-prep-EDA/clean/books.csv")
ratings_df = pd.read_csv("../data-prep-EDA/clean/ratings_test.csv")  # Use test ratings file here!
tags_df = pd.read_csv("../data-prep-EDA/clean/tags.csv")
book_tags_df = pd.read_csv("../data-prep-EDA/clean/book_tags.csv")
to_read_df = pd.read_csv("../data-prep-EDA/clean/to_read.csv")

# Build language dictionary.
unique_lang = books_df["language_code"].fillna("Unknown").unique().tolist()
lang2idx = {l: i for i, l in enumerate(unique_lang)}

# Build tag dictionary.
tag_id2name = {row.tag_id: row.tag_name for row in tags_df.itertuples()}

def get_book_tags_text(book_id, top_k=5):
    sub = book_tags_df[book_tags_df["book_id"] == book_id]
    if sub.empty:
        return ""
    sub = sub.sort_values("count", ascending=False)
    tag_ids = sub["tag_id"].tolist()[:top_k]
    tag_names = [tag_id2name.get(tid, "") for tid in tag_ids]
    return ", ".join(tag_names)

# Build candidate item text representations.
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

# Rebuild user behavior dictionaries (for test).
user_rated_books = ratings_df.groupby("user_id")["book_id"].apply(list).to_dict()
for user, books in user_rated_books.items():
    user_rated_books[user] = books[:50]

user_wishlist_books = to_read_df.groupby("user_id")["book_id"].apply(list).to_dict()
for user, books in user_wishlist_books.items():
    user_wishlist_books[user] = books[:50]

# Load BERT tokenizer and model.
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)
bert_model.eval()

# Set up test cache file paths.
cache_dir = "cache"
item_cache_file = os.path.join(cache_dir, "item_cache.pt")
rated_cache_file = os.path.join(cache_dir, "rated_cache_test.pt")
wishlist_cache_file = os.path.join(cache_dir, "wishlist_cache_test.pt")

# Global caches for embeddings.
if os.path.exists(item_cache_file):
    item_cache = torch.load(item_cache_file)
    print("Loaded item embeddings from cache.")
else:
    item_cache = {}

if os.path.exists(rated_cache_file):
    rated_cache = torch.load(rated_cache_file)
    print("Loaded rated embeddings for test from cache.")
else:
    rated_cache = {}

if os.path.exists(wishlist_cache_file):
    wishlist_cache = torch.load(wishlist_cache_file)
    print("Loaded wishlist embeddings for test from cache.")
else:
    wishlist_cache = {}

def get_user_rated_text(uid):
    rated_list = user_rated_books.get(uid, [])
    texts = [book_text_dict.get(bid, "") for bid in rated_list]
    return " [SEP] ".join(texts)

def get_user_wishlist_text(uid):
    wishlist_list = user_wishlist_books.get(uid, [])
    texts = [book_text_dict.get(bid, "") for bid in wishlist_list]
    return " [SEP] ".join(texts)

def get_item_text(bid):
    return book_text_dict.get(bid, "")

def get_candidate_item_embedding(bid):
    filename = os.path.join(cache_dir, f"item_{bid}.pt")
    if os.path.exists(filename):
        embedding = torch.load(filename)
        print(f"Loaded embedding for book {bid} from {filename}.")
        return embedding
    else:
        text = get_item_text(bid)
        if text == "":
            embedding = torch.zeros(768)
            print(f"No text for book {bid}; using zero vector.")
        else:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
            print(f"Computed embedding for book {bid}; saving to {filename}.")
        torch.save(embedding.cpu(), filename)
        return embedding

def make_batch_dlrm_format_bert(candidate_dense_batch, candidate_item_ids, user_ids_batch):
    """
    Constructs input X (shape [B, 2307]) by concatenating:
      - candidate_dense_batch: (B, 3)
      - item_embedding: (B, 768)
      - rated_embedding: (B, 768)
      - wishlist_embedding: (B, 768)
    Returns X and two empty lists.
    """
    if isinstance(user_ids_batch, torch.Tensor):
        user_ids_batch = user_ids_batch.cpu().tolist()
    
    B = candidate_dense_batch.size(0)
    
    global item_cache
    item_embeddings = []
    for bid in candidate_item_ids:
        bid_int = int(bid)
        if bid_int not in item_cache:
            item_cache[bid_int] = get_candidate_item_embedding(bid_int).cpu()
        emb = item_cache[bid_int].to(device)
        item_embeddings.append(emb.unsqueeze(0))
    item_embeddings = torch.cat(item_embeddings, dim=0)
    
    rated_embeddings = []
    wishlist_embeddings = []
    for uid in user_ids_batch:
        uid_int = int(uid)
        if uid_int not in rated_cache:
            inputs = tokenizer(get_user_rated_text(uid_int), return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = bert_model(**inputs)
            rated_cache[uid_int] = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
            print(f"Computed rated embedding for user {uid_int}.")
        if uid_int not in wishlist_cache:
            inputs = tokenizer(get_user_wishlist_text(uid_int), return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = bert_model(**inputs)
            wishlist_cache[uid_int] = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
            print(f"Computed wishlist embedding for user {uid_int}.")
        rated_emb = rated_cache[uid_int].to(device)
        wishlist_emb = wishlist_cache[uid_int].to(device)
        rated_embeddings.append(rated_emb.unsqueeze(0))
        wishlist_embeddings.append(wishlist_emb.unsqueeze(0))
    rated_embeddings = torch.cat(rated_embeddings, dim=0)
    wishlist_embeddings = torch.cat(wishlist_embeddings, dim=0)
    
    X = torch.cat([candidate_dense_batch, item_embeddings, rated_embeddings, wishlist_embeddings], dim=1)
    return X, [], []


###############################################################################
# Inference Code in Main
###############################################################################
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Instantiate the DLRM model with same architecture as during training.
    ln_emb = np.array([])  # No sparse features.
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
        sigmoid_top=-1,  # disable top sigmoid if needed.
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        md_flag=False,
    ).to(device)
    
    # Load the Lightning model checkpoint.
    checkpoint_path = "dlrm_lightning_checkpoint.ckpt"  # adjust if needed
    lightning_model = BPRLightningModule.load_from_checkpoint(
        checkpoint_path,
        dlrm_model=dlrm,
        lr=1e-3
    )
    lightning_model.eval()
    lightning_model.to(device)
    print("Lightning model loaded from checkpoint.\n")
    
    # -------------------------------------------------------------------------
    # Inference: Choose a test user.
    test_user_id = 61  # adjust as needed
    
    # For inference, use ratings_test.csv to compute the count of books the user rated.
    ratings_df_infer = pd.read_csv("../data-prep-EDA/clean/ratings_test.csv")
    if test_user_id in ratings_df_infer["user_id"].values:
        user_books_count = len(ratings_df_infer[ratings_df_infer["user_id"] == test_user_id])
    else:
        user_books_count = 0
    
    # For inference, sample candidate books. Here we randomly sample 100 candidates.
    candidate_books = books_df.sample(n=100, random_state=42).reset_index(drop=True)
    dense_features_list = []
    candidate_item_ids = []
    for _, row in candidate_books.iterrows():
        lang_numeric = float(lang2idx.get(row["language_code"], 0))
        avg_rating = float(row["average_rating"])
        candidate_dense = [float(user_books_count), lang_numeric, avg_rating]
        dense_features_list.append(candidate_dense)
        candidate_item_ids.append(row["book_id"])
    candidate_dense_batch = torch.tensor(dense_features_list, dtype=torch.float32).to(device)
    user_ids_batch = [test_user_id] * candidate_dense_batch.size(0)
    
    # Build the input X.
    X, lS_o, lS_i = make_batch_dlrm_format_bert(candidate_dense_batch, candidate_item_ids, user_ids_batch)
    
    with torch.no_grad():
        preds = lightning_model.model.forward(X, lS_o, lS_i)
    probs = torch.sigmoid(preds).view(-1).cpu().numpy()
    
    # Print user features in understandable text.
    print("\nUser Rated Books Text:")
    print(get_user_rated_text(test_user_id))
    print("\nUser Wishlist Books Text:")
    print(get_user_wishlist_text(test_user_id))
    
    print(f"\nCandidate recommendations for user {test_user_id}:")
    results = []
    for idx, row in candidate_books.iterrows():
        result = {
            "book_id": row["book_id"],
            "title": row["title"],
            "predicted_score": probs[idx]
        }
        # Add actual rating if available
        # Build dictionary of actual ratings for test user.
        # (Assume ratings_df_infer contains a "rating" column.)
        user_actual_ratings = ratings_df_infer[ratings_df_infer["user_id"] == test_user_id].set_index("book_id")["rating"].to_dict()
        result["actual_rating"] = user_actual_ratings.get(row["book_id"], np.nan)
        results.append(result)
        print(f"Book ID: {row['book_id']}, Title: {row['title']}, Predicted Score: {probs[idx]:.4f}, Actual Rating: {result['actual_rating']}")
    
    # Save evaluation results to CSV.
    eval_df = pd.DataFrame(results)
    eval_df.sort_values("predicted_score", ascending=False, inplace=True)
    eval_df.to_csv("evaluation_results.csv", index=False)
    print("\nEvaluation results saved to evaluation_results.csv.")
    
    # Now, compute nDCG@K using only books that the user has rated.
    user_actual_ratings = ratings_df_infer[ratings_df_infer["user_id"] == test_user_id].set_index("book_id")["rating"].to_dict()
    actual_list = []
    predicted_list = []
    for res in results:
        if not np.isnan(res["actual_rating"]):
            actual_list.append(res["actual_rating"])
            predicted_list.append(res["predicted_score"])
    
    k = 10
    ndcg_score_value = ndcg_at_k_graded(actual_list, predicted_list, k)
    print(f"\nnDCG@{k} (graded, only rated candidates): {ndcg_score_value:.4f}")
    
    # Save the updated test caches.
    torch.save(rated_cache, rated_cache_file)
    torch.save(wishlist_cache, wishlist_cache_file)
    print("Test caches saved.")
