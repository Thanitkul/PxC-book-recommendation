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

import time

def wait_for_db_ready(max_retries=15, delay=3):
    for i in range(max_retries):
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'app'
                        AND table_name = 'books'
                    );
                """)
                exists = cur.fetchone()[0]
                if exists:
                    print(f"✅ Schema and books table are ready (attempt {i+1})")
                    conn.close()
                    return
                else:
                    print(f"⏳ Table 'app.books' not found yet (attempt {i+1})")
            conn.close()
        except Exception as e:
            print(f"⏳ DB not ready (attempt {i+1}) - {e}")
        time.sleep(delay)
    raise Exception("❌ Database schema 'app.books' not found after retries.")


def get_connection(max_retries=10, delay=3):
    for i in range(max_retries):
        try:
            return psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
        except psycopg2.OperationalError as e:
            print(f"⏳ Waiting for DB... ({i+1}/{max_retries}) - {e}")
            time.sleep(delay)
    raise Exception("❌ Could not connect to the database after several attempts.")


def truncate_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            TRUNCATE app.to_read, app.ratings, app.book_tags, app.tags, app.books
            RESTART IDENTITY CASCADE;
        """)
    conn.commit()

def is_already_loaded(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'app'
                AND table_name = 'books'
            );
        """)
        return cur.fetchone()[0]

# ========================================
# 2. SQL Schema and Table Creation
# ========================================
CREATE_SCHEMA_AND_TABLES_SQL = """

DROP SCHEMA IF EXISTS app CASCADE;

CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE IF NOT EXISTS app.users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password TEXT NOT NULL,
    permission VARCHAR(100) NOT NULL DEFAULT 'user',
    selected_genre BOOLEAN NOT NULL DEFAULT FALSE,
    genre_ids INTEGER[] DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

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

CREATE TABLE IF NOT EXISTS app.tags (
    tag_id INTEGER PRIMARY KEY,
    tag_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app.book_tags (
    book_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    count INTEGER,
    PRIMARY KEY (book_id, tag_id)
);

CREATE TABLE IF NOT EXISTS app.ratings (
    user_id INTEGER NOT NULL,
    book_id INTEGER NOT NULL,
    rating INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, book_id)
);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = NOW();
   RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS trg_update_timestamp ON app.ratings;
CREATE TRIGGER trg_update_timestamp
BEFORE UPDATE ON app.ratings
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE IF NOT EXISTS app.to_read (
    user_id INTEGER NOT NULL,
    book_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, book_id)
);
"""

ADD_ADMIN_USER_SQL = """
INSERT INTO app.users (id, username, password, permission)
VALUES (0, 'admin', %s, 'admin')
ON CONFLICT (id) DO NOTHING;
"""

def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_SCHEMA_AND_TABLES_SQL)
        cur.execute(ADD_ADMIN_USER_SQL, (bcrypt.hashpw(b"admin", bcrypt.gensalt()).decode('utf-8'),))
    conn.commit()

# ========================================
# 3. Data Loaders
# ========================================
def load_csv_data(conn, path, table, transform_fn, conflict_keys):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [transform_fn(row) for row in tqdm(reader, desc=f"Loading {table}", unit="row")]

    columns = ', '.join(rows[0]._fields)
    placeholders = ', '.join([f"%s"] * len(rows[0]))
    sql = f"""
        INSERT INTO app.{table} ({columns})
        VALUES %s
        ON CONFLICT ({conflict_keys}) DO NOTHING;
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()

def create_missing_users(conn, user_ids):
    if not user_ids:
        return
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM app.users WHERE id = ANY(%s);", (list(user_ids),))
        existing_ids = {row[0] for row in cur.fetchall()}

    missing = user_ids - existing_ids
    if not missing:
        return

    salt = bcrypt.gensalt(rounds=8)
    to_insert = [
        (uid, f"user_{uid}", bcrypt.hashpw(f"user_{uid}".encode('utf-8'), salt).decode('utf-8'), "user")
        for uid in tqdm(missing, desc="Creating missing users", unit="user")
    ]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO app.users (id, username, password, permission)
            VALUES %s
            ON CONFLICT (id) DO NOTHING;
        """, to_insert)
    conn.commit()

# ========================================
# 4. Main Execution
# ========================================
def main():
    #wait_for_db_ready()
    print("✅ DB and schema ready.")
    conn = get_connection()
    try:
        if is_already_loaded(conn):
            print("✅ Data already loaded. Skipping.")
            return
    except Exception as e:
        print("❌ Error checking data load status:", e)
    try:
        create_tables(conn)
        truncate_tables(conn)

        from collections import namedtuple

        Book = namedtuple("Book", [
            "book_id", "authors", "original_publication_year", "title", "language_code",
            "average_rating", "ratings_count", "ratings_1", "ratings_2", "ratings_3",
            "ratings_4", "ratings_5", "image_url", "small_image_url"
        ])
        load_csv_data(conn, "data/books.csv", "books", lambda r: Book(
            int(r["book_id"]), r["authors"],
            float(r["original_publication_year"]) if r["original_publication_year"] else None,
            r["title"], r["language_code"], float(r["average_rating"]),
            int(r["ratings_count"]), int(r["ratings_1"]), int(r["ratings_2"]),
            int(r["ratings_3"]), int(r["ratings_4"]), int(r["ratings_5"]),
            r["image_url"], r["small_image_url"]
        ), "book_id")

        Tag = namedtuple("Tag", ["tag_id", "tag_name"])
        load_csv_data(conn, "data/tags.csv", "tags", lambda r: Tag(int(r["tag_id"]), r["tag_name"]), "tag_id")

        BookTag = namedtuple("BookTag", ["book_id", "tag_id", "count"])
        load_csv_data(conn, "data/book_tags.csv", "book_tags", lambda r: BookTag(
            int(r["book_id"]), int(r["tag_id"]), int(r["count"])
        ), "book_id, tag_id")

        Rating = namedtuple("Rating", ["user_id", "book_id", "rating"])
        with open("data/ratings.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ratings = []
            user_ids = set()
            for row in tqdm(reader, desc="Loading ratings", unit="row"):
                uid = int(row["user_id"])
                bid = int(row["book_id"])
                user_ids.add(uid)
                ratings.append((uid, bid, int(row["rating"])))
        create_missing_users(conn, user_ids)
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO app.ratings (user_id, book_id, rating)
                VALUES %s
                ON CONFLICT (user_id, book_id) DO NOTHING;
            """, ratings)
        conn.commit()

        ToRead = namedtuple("ToRead", ["user_id", "book_id"])
        with open("data/to_read.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            to_read = []
            user_ids = set()
            for row in tqdm(reader, desc="Loading to_read", unit="row"):
                uid = int(row["user_id"])
                user_ids.add(uid)
                to_read.append((uid, int(row["book_id"])))
        create_missing_users(conn, user_ids)
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO app.to_read (user_id, book_id)
                VALUES %s
                ON CONFLICT (user_id, book_id) DO NOTHING;
            """, to_read)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT setval('app.users_id_seq', (SELECT MAX(id) FROM app.users));")
        conn.commit()

        print("\u2705 Data loading complete!")

    except Exception as e:
        conn.rollback()
        print("\u274C Error during data load:", e)
    finally:
        conn.close()

if __name__ == "__main__":
    main()