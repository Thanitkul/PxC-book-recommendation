import psycopg2
import pandas as pd
from collections import defaultdict
from config import DB_URL

def get_user_seen_books(user_id: int) -> set[int]:
    """
    Return a set of book_ids the user has rated or wishlisted.
    """
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT book_id FROM app.ratings WHERE user_id = %s
                UNION
                SELECT book_id FROM app.to_read WHERE user_id = %s
            """, (user_id, user_id))
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

def load_static_book_features() -> dict:
    """
    Loads static metadata needed by the two-tower model:
    - Books
    - Book authors, languages, tag mappings
    - Book dense stats (ratings count, average rating)
    """
    conn = psycopg2.connect(DB_URL)
    try:
        books_query = """
            SELECT book_id, authors, language_code,
                   ratings_count, average_rating, is_visible
            FROM app.books
            WHERE is_visible = TRUE;
        """
        books_df = pd.read_sql(books_query, conn)

        # Build author2idx and lang2idx
        author2idx = {a: i+1 for i, a in enumerate(sorted(books_df['authors'].dropna().unique()))}
        lang2idx   = {l: i+1 for i, l in enumerate(sorted(books_df['language_code'].fillna("unk").unique()))}

        # Index maps for embedding lookup
        book_author = books_df.set_index("book_id")["authors"].map(author2idx).fillna(0).astype(int).to_dict()
        book_lang   = books_df.set_index("book_id")["language_code"].fillna("unk").map(lang2idx).fillna(0).astype(int).to_dict()

        # Book dense features
        book_dense = books_df.set_index("book_id")[["ratings_count", "average_rating"]].to_dict("index")
        max_rc     = float(books_df["ratings_count"].max() or 1.0)

        # Book tags
        book_tags_df = pd.read_sql("SELECT book_id, tag_id, count FROM app.book_tags;", conn)
        tags_df      = pd.read_sql("SELECT tag_id, tag_name FROM app.tags;", conn)
        tag_id2name  = dict(tags_df.values)

        # Top-5 tags per book by frequency
        tag_counts = defaultdict(list)
        for row in book_tags_df.itertuples(index=False):
            tag_counts[row.book_id].append((row.tag_id, row.count))
        
        top_tags = {
            bid: [tag for tag, _ in sorted(lst, key=lambda x: -x[1])[:5]]
            for bid, lst in tag_counts.items()
        }

        # Pad missing with zeros
        for bid in books_df["book_id"]:
            if bid not in top_tags:
                top_tags[bid] = [0]*5
            elif len(top_tags[bid]) < 5:
                top_tags[bid] += [0]*(5 - len(top_tags[bid]))

        return {
            "books": books_df,
            "book_author": book_author,
            "book_lang": book_lang,
            "book_dense": book_dense,
            "max_rc": max_rc,
            "top_tags": top_tags,
            "tag_id2name": tag_id2name,
            "author2idx": author2idx,
            "lang2idx": lang2idx,
        }

    finally:
        conn.close()

def get_user_history(user_id: int) -> tuple[list[tuple[int, float]], list[int]]:
    """
    Returns:
      - rated_books:    List of (book_id, rating)
      - wishlist_books: List of book_id
    """
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT book_id, rating FROM app.ratings WHERE user_id = %s", (user_id,))
            rated_books = [(row[0], float(row[1])) for row in cur.fetchall()]

            cur.execute("SELECT book_id FROM app.to_read WHERE user_id = %s", (user_id,))
            wishlist_books = [row[0] for row in cur.fetchall()]
        
        return rated_books, wishlist_books
    finally:
        conn.close()

def get_ratings_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - ratings: pd.DataFrame with [user_id, book_id, rating]
      - books: pd.DataFrame with [book_id, title, authors]
    """
    conn = psycopg2.connect(DB_URL)
    try:
        ratings_df = pd.read_sql("SELECT user_id, book_id, rating FROM app.ratings;", conn)
        books_df   = pd.read_sql("SELECT book_id, title, authors FROM app.books WHERE is_visible = TRUE;", conn)
        return ratings_df, books_df
    finally:
        conn.close()
