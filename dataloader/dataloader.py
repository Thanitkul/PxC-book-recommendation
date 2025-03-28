#!/usr/bin/env python3
import os
import csv
import bcrypt
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from tqdm import tqdm

# ========================================
# 1. Load env & Connect to PostgreSQL
# ========================================
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

def get_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    conn.autocommit = False  # We'll handle transactions manually
    return conn

def truncate_tables(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE app.to_read, app.ratings, app.book_tags, app.tags, app.books RESTART IDENTITY CASCADE;")
    conn.commit()

# ========================================
# 2. SQL for Creating Schema & Tables
# ========================================
CREATE_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS app;
"""

CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS app.users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password TEXT NOT NULL,
    permission VARCHAR(100) NOT NULL DEFAULT 'user',
    selected_genre BOOLEAN NOT NULL DEFAULT FALSE,
    genre_ids INTEGER[] DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_id
    ON app.users (id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
    ON app.users (username);
"""

ADD_ADMIN_USER_SQL = """
INSERT INTO app.users (id, username, password, permission)
VALUES (0, 'admin', %s, 'admin')
ON CONFLICT (id) DO NOTHING;
"""

CREATE_BOOKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS app.books (
    book_id INTEGER PRIMARY KEY,
    authors TEXT,
    original_publication_year REAL,
    title TEXT,
    language_code VARCHAR(10),
    average_rating REAL,
    ratings_count INTEGER,
    ratings_1 INTEGER,
    ratings_2 INTEGER,
    ratings_3 INTEGER,
    ratings_4 INTEGER,
    ratings_5 INTEGER,
    image_url TEXT,
    small_image_url TEXT,
    is_visible BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_books_id
    ON app.books (book_id);
CREATE INDEX IF NOT EXISTS idx_books_title
    ON app.books (title);
"""

CREATE_TAGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS app.tags (
    tag_id INTEGER PRIMARY KEY,
    tag_name TEXT NOT NULL
);
"""

CREATE_BOOK_TAGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS app.book_tags (
    book_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    count INTEGER,
    PRIMARY KEY (book_id, tag_id)
);
"""

CREATE_RATINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS app.ratings (
    user_id INTEGER NOT NULL,
    book_id INTEGER NOT NULL,
    rating INTEGER NOT NULL,
    PRIMARY KEY (user_id, book_id)
);
"""

CREATE_TO_READ_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS app.to_read (
    user_id INTEGER NOT NULL,
    book_id INTEGER NOT NULL,
    PRIMARY KEY (user_id, book_id)
);
"""

def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_SCHEMA_SQL)
        cur.execute(CREATE_USERS_TABLE_SQL)
        cur.execute(CREATE_BOOKS_TABLE_SQL)
        cur.execute(CREATE_TAGS_TABLE_SQL)
        cur.execute(CREATE_BOOK_TAGS_TABLE_SQL)
        cur.execute(CREATE_RATINGS_TABLE_SQL)
        cur.execute(CREATE_TO_READ_TABLE_SQL)
        cur.execute(ADD_ADMIN_USER_SQL, (bcrypt.hashpw(b"admin", bcrypt.gensalt()).decode('utf-8'),))

    conn.commit()

# ========================================
# 3. Load CSV Data
# ========================================
def load_books(conn, csv_path):
    """
    CSV columns:
    book_id,authors,original_publication_year,title,language_code,
    average_rating,ratings_count,ratings_1,ratings_2,ratings_3,
    ratings_4,ratings_5,image_url,small_image_url
    is_visible defaults to TRUE
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in tqdm(reader, desc="Loading books", unit="row"):
            book_id = int(row["book_id"])
            authors = row["authors"]
            pub_year = float(row["original_publication_year"]) if row["original_publication_year"] else None
            title = row["title"]
            lang_code = row["language_code"]
            avg_rating = float(row["average_rating"]) if row["average_rating"] else None
            ratings_count = int(row["ratings_count"]) if row["ratings_count"] else None
            r1 = int(row["ratings_1"]) if row["ratings_1"] else 0
            r2 = int(row["ratings_2"]) if row["ratings_2"] else 0
            r3 = int(row["ratings_3"]) if row["ratings_3"] else 0
            r4 = int(row["ratings_4"]) if row["ratings_4"] else 0
            r5 = int(row["ratings_5"]) if row["ratings_5"] else 0
            img_url = row["image_url"]
            sm_img_url = row["small_image_url"]
            rows.append((
                book_id, authors, pub_year, title, lang_code, avg_rating,
                ratings_count, r1, r2, r3, r4, r5, img_url, sm_img_url
            ))

    insert_sql = """
        INSERT INTO app.books (
            book_id, authors, original_publication_year, title, language_code,
            average_rating, ratings_count, ratings_1, ratings_2, ratings_3,
            ratings_4, ratings_5, image_url, small_image_url
        )
        VALUES %s
        ON CONFLICT (book_id) DO NOTHING;
    """
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, rows)
    conn.commit()

def load_tags(conn, csv_path):
    """
    CSV columns: tag_id,tag_name
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(int(row["tag_id"]), row["tag_name"]) for row in tqdm(reader, desc="Loading tags", unit="row")]

    insert_sql = """
        INSERT INTO app.tags (tag_id, tag_name)
        VALUES %s
        ON CONFLICT (tag_id) DO NOTHING;
    """
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, rows)
    conn.commit()

def load_book_tags(conn, csv_path):
    """
    CSV columns: book_id,tag_id,count
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in tqdm(reader, desc="Loading book tags", unit="row"):
            book_id = int(row["book_id"])
            tag_id = int(row["tag_id"])
            count = int(row["count"])
            rows.append((book_id, tag_id, count))

    insert_sql = """
        INSERT INTO app.book_tags (book_id, tag_id, count)
        VALUES %s
        ON CONFLICT (book_id, tag_id) DO NOTHING;
    """
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, rows)
    conn.commit()

def load_ratings(conn, csv_path):
    """
    CSV columns: user_id,book_id,rating
    Must also ensure user accounts exist in app.users
      => password = "user_{user_id}" (hash)
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        user_ids = set()
        for row in tqdm(reader, desc="Loading ratings", unit="row"):
            uid = int(row["user_id"])
            bid = int(row["book_id"])
            rating = int(row["rating"])
            rows.append((uid, bid, rating))
            user_ids.add(uid)

    # 1) create missing user accounts
    create_missing_users(conn, user_ids)

    # 2) insert ratings
    insert_sql = """
        INSERT INTO app.ratings (user_id, book_id, rating)
        VALUES %s
        ON CONFLICT (user_id, book_id) DO NOTHING;
    """
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, rows)
    conn.commit()

def load_to_read(conn, csv_path):
    """
    CSV columns: user_id,book_id
    Also ensure user accounts exist
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        user_ids = set()
        for row in tqdm(reader, desc="Loading to-read list", unit="row"):
            uid = int(row["user_id"])
            bid = int(row["book_id"])
            rows.append((uid, bid))
            user_ids.add(uid)

    # create missing user accounts
    create_missing_users(conn, user_ids)

    insert_sql = """
        INSERT INTO app.to_read (user_id, book_id)
        VALUES %s
        ON CONFLICT (user_id, book_id) DO NOTHING;
    """
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, rows)
    conn.commit()

# ========================================
# 4. Create Missing User Accounts
# ========================================
def create_missing_users(conn, user_ids):
    if not user_ids:
        return

    # find existing user IDs
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id FROM app.users
            WHERE id = ANY(%s);
        """, (list(user_ids),))
        existing_ids = {row[0] for row in cur.fetchall()}

    missing = user_ids - existing_ids
    if not missing:
        return

        # for each missing user, password = "user_{user_id}"
    to_insert = []
    salt = bcrypt.gensalt(rounds=8)  # Use fixed salt with lower cost factor

    for uid in tqdm(missing, desc="Creating missing users", unit="user"):
        password_str = f"user_{uid}"  # e.g. "user_9"
        hashed_pw = bcrypt.hashpw(password_str.encode('utf-8'), salt).decode('utf-8')
        username = f"user_{uid}"
        to_insert.append((uid, username, hashed_pw, "user"))


    insert_sql = """
        INSERT INTO app.users (id, username, password, permission)
        VALUES %s
        ON CONFLICT (id) DO NOTHING;
    """
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, to_insert)
    conn.commit()

# ========================================
# 5. Main Execution
# ========================================
def main():
    conn = get_connection()
    try:
        create_tables(conn)
        truncate_tables(conn)

        # 2. load CSVs
        load_books(conn, "data/books.csv")
        load_tags(conn, "data/tags.csv")
        load_book_tags(conn, "data/book_tags.csv")
        load_ratings(conn, "data/ratings.csv")
        load_to_read(conn, "data/to_read.csv")

        def reset_user_id_sequence(conn):
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT setval('app.users_id_seq', (SELECT MAX(id) FROM app.users));
                """)
        reset_user_id_sequence(conn)

        conn.commit()

        print("✅ Data loading complete!")
    except Exception as e:
        conn.rollback()
        print("❌ Error during data load:", e)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
