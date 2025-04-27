from typing import List, Optional, Tuple
import asyncio
import psycopg2
from config import DB_URL
from recsys.logic import cold_start

def get_user_activity_counts(user_id: int) -> Tuple[int, int]:
    """
    Synchronously queries the database to return the count of ratings
    and wishlist items for the given user.
    """
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM app.ratings WHERE user_id = %s", (user_id,))
            ratings_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM app.to_read WHERE user_id = %s", (user_id,))
            wishlist_count = cur.fetchone()[0]
    finally:
        conn.close()
    return ratings_count, wishlist_count

async def get_user_activity_counts_async(user_id: int) -> Tuple[int, int]:
    """
    Wraps the synchronous DB query in a thread so it doesn't block the event loop.
    """
    return await asyncio.to_thread(get_user_activity_counts, user_id)

async def recommend_books(user_id: int, user_genres: Optional[str] = None) -> List[int]:
    """
    Dispatch recommendation requests based on user's activity.
    
    - Users with fewer than 5 ratings OR fewer than 5 wishlist items use the cold‑start model.
    - Users with sufficient activity (>= 5 ratings and >= 5 wishlist items) receive combined results
      from DLRM and collaborative models (placeholders for now).
    """
    ratings_count, wishlist_count = await get_user_activity_counts_async(user_id)
    if ratings_count < 5 or wishlist_count < 5:
        return await cold_start_recommendation(user_id, user_genres)
    else:
        dlrm_results = await dlrm_recommendation(user_id)
        collab_results = await collaborative_recommendation(user_id)
        # Combine the results (here we concatenate and remove duplicates)
        combined = list(dict.fromkeys(dlrm_results + collab_results))
        return combined

async def cold_start_recommendation(user_id: int, user_genres: Optional[str] = None) -> List[int]:
    """
    For cold users, use the cold-start model.
    If no genres are provided, return a fallback list.
    """
    if not user_genres:
        return [1, 2, 3, 4, 5]
    recommended_books = cold_start.recommend_book_ids_by_genres(user_genres, top_n=10)
    return recommended_books

async def dlrm_recommendation(user_id: int) -> List[int]:
    # Placeholder for DLRM logic
    return [6, 7, 8, 9, 10]

async def collaborative_recommendation(user_id: int) -> List[int]:
    # Placeholder for collaborative filtering logic
    return [11, 12, 13, 14, 15]
