const { pool } = require("./db");

// Helper: get book base info with optional user info
const getAllBooks = async (userId, sortBy = "popularity") => {
    const validSorts = {
        popularity: "b.ratings_count DESC",
        ratings: "b.average_rating DESC",
        wishlist: "CASE WHEN t.user_id IS NOT NULL THEN 0 ELSE 1 END, b.title ASC",
        title: "b.title ASC"
    };

    const orderClause = validSorts[sortBy] || validSorts["popularity"];

    const query = `
        SELECT b.book_id, b.title, b.authors, b.image_url,
               COALESCE(r.rating, NULL) AS rating,
               CASE WHEN t.user_id IS NOT NULL THEN true ELSE false END AS in_wishlist,
               b.ratings_count AS popularity
        FROM app.books b
        LEFT JOIN app.ratings r ON b.book_id = r.book_id AND r.user_id = $1
        LEFT JOIN app.to_read t ON b.book_id = t.book_id AND t.user_id = $1
        ORDER BY ${orderClause};
    `;

    const res = await pool.query(query, [userId]);
    return res.rows;
};

const editRating = async (userId, bookId, rating) => {
    const query = `
        INSERT INTO app.ratings (user_id, book_id, rating, created_at)
        VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id, book_id)
        DO UPDATE SET rating = EXCLUDED.rating, created_at = CURRENT_TIMESTAMP
        RETURNING book_id, rating;
    `;

    const res = await pool.query(query, [userId, bookId, rating]);
    return res.rows[0];
};

const deleteRating = async (userId, bookId) => {
    const query = `
        DELETE FROM app.ratings
        WHERE user_id = $1 AND book_id = $2
        RETURNING book_id;
    `;

    const res = await pool.query(query, [userId, bookId]);
    return res.rows[0];
};

const addToWishlist = async (userId, bookId) => {
    const query = `
        INSERT INTO app.to_read (user_id, book_id, created_at)
        VALUES ($1, $2, CURRENT_TIMESTAMP)
        ON CONFLICT DO NOTHING
        RETURNING book_id;
    `;

    const res = await pool.query(query, [userId, bookId]);
    return res.rows.length > 0 ? res.rows[0] : { book_id: bookId };
};

const removeFromWishlist = async (userId, bookId) => {
    const query = `
        DELETE FROM app.to_read
        WHERE user_id = $1 AND book_id = $2
        RETURNING book_id;
    `;

    const res = await pool.query(query, [userId, bookId]);
    return res.rows[0];
};

module.exports = {
    getAllBooks,
    editRating,
    deleteRating,
    addToWishlist,
    removeFromWishlist
};
