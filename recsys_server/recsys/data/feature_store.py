import os
import psycopg2
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import time

from config import DB_URL  # Ensure DB_URL is set in your config

class FeatureStore:
    """
    A singleton class that loads static item data from the database,
    preprocesses it, and computes embeddings. This is shared across models.
    """
    def __init__(self):
        self.books_tagged = None         # DataFrame containing item metadata
        self.book_tag_embeddings = None    # Precomputed embeddings for each book
        self.model = None                  # SentenceTransformer model
        self.tag_id_to_name = {}

    def load_data(self):
        """
        Connect to the database and load static tables (books, book_tags, tags)
        then merge and aggregate the data.
        """
        for i in range(10):
            try:
                conn = psycopg2.connect(DB_URL)
                break
            except Exception as e:
                print(f"⏳ Retrying DB connect for feature store ({i+1}/10)...", e)
                time.sleep(3)
        else:
            raise RuntimeError("❌ Feature store: DB connection failed after retries.")

        print("✅ Feature store: DB connection established.")
        print("Loading books and tags...")

        books_query = """
            SELECT book_id, title, authors, average_rating, ratings_count,
                   original_publication_year, language_code
            FROM app.books
            WHERE is_visible = TRUE;
        """
        books_df = pd.read_sql(books_query, conn)

        book_tags_df = pd.read_sql("SELECT book_id, tag_id FROM app.book_tags;", conn)
        tags_df = pd.read_sql("SELECT tag_id, tag_name FROM app.tags;", conn)

        # ADD THIS LINE: Store mapping as {tag_id: tag_name}
        self.tag_id_to_name = dict(tags_df.values)

        # Merge and preprocess
        book_tags_merged = pd.merge(book_tags_df, tags_df, on="tag_id", how="inner")
        books_with_tags = pd.merge(books_df, book_tags_merged, on="book_id", how="inner")

        self.books_tagged = books_with_tags.groupby('book_id').agg({
            'title': 'first',
            'authors': 'first',
            'original_publication_year': 'first',
            'language_code': 'first',
            'tag_name': lambda x: ' '.join(set(x))
        }).reset_index()

        self.books_tagged['tag_text'] = self.books_tagged['tag_name']

        print("Feature store: Data loaded and preprocessed.")
        print(f"Number of books loaded: {len(self.books_tagged)}")

        conn.close()

    def load_model_and_compute_embeddings(self):
        """
        Load the SentenceTransformer model and precompute the embeddings for the
        tag_text field of all items.
        """
        print("Loading SentenceTransformer model...")
        print("Using GPU" if torch.cuda.is_available() else "Using CPU")
        self.model = SentenceTransformer('BAAI/bge-m3',device='cuda' if torch.cuda.is_available() else 'cpu')
        self.book_tag_embeddings = self.model.encode(
            self.books_tagged['tag_text'].tolist(), convert_to_tensor=True
        )
        print("Feature store: Model loaded and embeddings computed.")
        print(f"Number of embeddings computed: {len(self.book_tag_embeddings)}")

    def initialize(self):
        """
        Perform all initialization steps: load data and precompute embeddings.
        """
        self.load_data()
        self.load_model_and_compute_embeddings()


# Create a global instance of FeatureStore to be used by all recommendation models.
feature_store = FeatureStore()

def init_feature_store():
    """A helper function to initialize the global feature store."""
    feature_store.initialize()
