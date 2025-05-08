import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from recsys.data.loader import get_ratings_matrix, get_user_history
import pandas as pd


class UserBasedCFOptimized:
    def __init__(self, ratings_df, books_df, k=40):
        self.books = books_df
        row = ratings_df.user_id.values - 1
        col = ratings_df.book_id.values - 1
        data = ratings_df.rating.values

        self.n_users = row.max() + 1
        self.n_items = col.max() + 1
        self.R_csr   = sparse.csr_matrix((data, (row, col)), shape=(self.n_users, self.n_items))

        nz_counts = (self.R_csr != 0).sum(1).A1
        self.mu   = self.R_csr.sum(1).A1 / np.maximum(nz_counts, 1)

        R_coo = self.R_csr.tocoo(copy=True)
        R_coo.data = R_coo.data - self.mu[R_coo.row]
        self.Rd = R_coo.tocsr()
        self.Rd.eliminate_zeros()

        knn_graph = NearestNeighbors(n_neighbors=k+1, metric="cosine", algorithm="brute", n_jobs=-1) \
            .fit(self.Rd).kneighbors_graph(self.Rd, mode="distance")

        knn_graph.setdiag(0)
        knn_graph.eliminate_zeros()
        knn_graph.data = 1.0 - knn_graph.data
        self.S = knn_graph.tocsr()

    def recommend(self, user_id: int, top_n: int = 10) -> list[int]:
        u = user_id - 1

        if u < 0 or u >= self.Rd.shape[0]:
            # New user not in similarity matrix — soft integration

            rated_books, _ = get_user_history(user_id)

            if not rated_books:
                # Completely cold user → fallback to popularity
                pop = self.R_csr.sum(0).A1
                idx = np.argsort(pop)[::-1][:top_n]
                return (idx + 1).tolist()

            # Step 1: Create temporary rating vector
            u_vec = np.zeros(self.n_items)
            for book_id, rating in rated_books:
                if 0 < book_id <= self.n_items:
                    u_vec[book_id - 1] = rating

            # Step 2: Normalize user vector
            seen = u_vec != 0
            if not seen.any():
                pop = self.R_csr.sum(0).A1
                idx = np.argsort(pop)[::-1][:top_n]
                return (idx + 1).tolist()

            mu = np.mean(u_vec[seen])
            u_vec[seen] -= mu

            # Step 3: Predict scores via dot with Rd
            sim_scores = self.Rd.dot(u_vec)
            den = np.linalg.norm(sim_scores) + 1e-9
            preds = self.mu + sim_scores / den
            preds[seen] = -np.inf

            top_idx = np.argpartition(preds, -top_n)[-top_n:]
            top_idx = top_idx[np.argsort(preds[top_idx])[::-1]]
            return (top_idx + 1).tolist()

        # Existing user — normal path
        u_vec = self.Rd[u, :].toarray().ravel()
        seen = u_vec != 0

        if not seen.any():
            pop = self.R_csr.sum(0).A1
            idx = np.argsort(pop)[::-1][:top_n]
            return (idx + 1).tolist()

        s_u   = self.S.getrow(u)
        num   = s_u.dot(self.Rd).toarray().ravel()
        den   = np.abs(s_u.data).sum() + 1e-9
        preds = self.mu[u] + num / den
        preds[seen] = -np.inf

        top_idx = np.argpartition(preds, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(preds[top_idx])[::-1]]
        return (top_idx + 1).tolist()

    
_model = None

def recommend_collaborative(user_id: int, top_n: int = 30) -> list[int]:
    global _model
    if _model is None:
        print("📥 Initializing collaborative model...")
        ratings, books = get_ratings_matrix()
        _model = UserBasedCFOptimized(ratings, books)
        print("✅ Collaborative model ready.")

    return _model.recommend(user_id=user_id, top_n=top_n)
