from sentence_transformers import util
from recsys.data.feature_store import feature_store
import torch


def recommend_by_genres(user_genres: str, top_n=10):
    """
    Compute recommendations for a cold user by comparing the user's input genres
    to the precomputed embeddings.
    """
    if (feature_store.model is None or
        feature_store.books_tagged is None or
        feature_store.book_tag_embeddings is None):
        raise Exception("Feature store not initialized. Run init_feature_store() first.")

    # Clean and split the input genres string
    genre_list = [g.strip() for g in user_genres.split(',') if g.strip()]
    if not genre_list:
        raise ValueError("Please input at least one genre!")
    
    # Encode user genres and compute the average embedding (user profile)
    genre_embeddings = feature_store.model.encode(genre_list, convert_to_tensor=True)
    user_embedding = genre_embeddings.mean(dim=0)
    
    # Compute cosine similarity between the user profile and all book embeddings
    noise_strength = 0.02  # tune between 0.01–0.05 for subtle randomness
    scores = util.pytorch_cos_sim(user_embedding, feature_store.book_tag_embeddings)[0]
    scores += noise_strength * torch.randn_like(scores)
    top_results = scores.topk(top_n)
    
    # Retrieve the top matching books
    results = feature_store.books_tagged.iloc[top_results.indices.cpu().numpy()].copy()
    results['similarity'] = top_results.values.cpu().numpy()
    return results

def recommend_book_ids_by_genres(user_genres: str, top_n=10):
    """
    Return only the list of recommended book IDs.
    """
    results = recommend_by_genres(user_genres, top_n)
    return results['book_id'].tolist()
