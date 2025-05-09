from typing import List, Optional, Tuple
import asyncio
import psycopg2
from config import DB_URL
from recsys.logic import cold_start
from recsys.logic.two_tower import recommend_two_tower
from recsys.logic.collaborative import recommend_collaborative
from recsys.data.feature_store import feature_store

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

async def recommend_books(user_id: int) -> List[int]:
    """
    Dispatch recommendation requests based on user's activity.
    
    - Users with fewer than 5 ratings OR fewer than 5 wishlist items use the cold‑start model.
    - Users with sufficient activity (>= 5 ratings and >= 5 wishlist items) receive combined results
      from two-tower and collaborative models (placeholders for now).
    """
    ratings_count, wishlist_count = await get_user_activity_counts_async(user_id)
    if ratings_count < 3:
        # query the database for user genres by query genre_ids INTEGER[] from app.users and map them to tag names in app.tags
        conn = psycopg2.connect(DB_URL)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT genre_ids FROM app.users WHERE id = %s", (user_id,))
                result = cur.fetchone()
                if result is None:
                    return await cold_start_recommendation(user_id)
                user_genres = result[0]

                # Map genre_ids to tag names
                user_genres = ' '.join([
            feature_store.tag_id_to_name.get(tag_id, '') for tag_id in user_genres
        ])
        finally:
            conn.close()
        return await cold_start_recommendation(user_id, user_genres)
    else:
        two_tower_results = await asyncio.to_thread(recommend_two_tower, user_id)
        collab_results = await asyncio.to_thread(recommend_collaborative, user_id)
        # Combine the results (here we concatenate and remove duplicates)
        # Total 100 books: 70 from two-tower, 30 from CF, exclude seen
        combined = list(dict.fromkeys(two_tower_results))[:100]
        return combined

async def cold_start_recommendation(user_id: int, user_genres: Optional[str] = None) -> List[int]:
    """
    For cold users, use the cold-start model.
    If no genres are provided, return a fallback list.
    """
    if not user_genres:
        return [1, 2, 3, 4, 5]
    recommended_books = cold_start.recommend_book_ids_by_genres(user_genres, top_n=100)
    return recommended_books

