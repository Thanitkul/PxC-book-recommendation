const { pool } = require("./db");

const getRatedBooks = async (userId) => {
    const query = `
        SELECT r.book_id, r.rating, r.created_at, b.title, b.authors, b.image_url
        FROM app.ratings r
        JOIN app.books b ON r.book_id = b.book_id
        WHERE r.user_id = $1
        ORDER BY r.created_at DESC;
    `;
    const res = await pool.query(query, [userId]);
    return res.rows;
};

const getRatedBookById = async (userId, bookId) => {
    const query = `
        SELECT r.book_id, r.rating, r.created_at, b.title, b.authors, b.image_url
        FROM app.ratings r
        JOIN app.books b ON r.book_id = b.book_id
        WHERE r.user_id = $1 AND r.book_id = $2;
    `;
    const res = await pool.query(query, [userId, bookId]);
    return res.rows.length ? res.rows[0] : null;
};

const updateRating = async (userId, bookId, rating) => {
    const query = `
        INSERT INTO app.ratings (user_id, book_id, rating, created_at)
        VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id, book_id)
        DO UPDATE SET rating = EXCLUDED.rating, created_at = CURRENT_TIMESTAMP;
    `;
    await pool.query(query, [userId, bookId, rating]);
    return await getRatedBookById(userId, bookId);
};

const deleteRating = async (userId, bookId) => {
    const query = `DELETE FROM app.ratings WHERE user_id = $1 AND book_id = $2`;
    await pool.query(query, [userId, bookId]);
};

const getWishlist = async (userId) => {
    const query = `
        SELECT t.book_id, t.created_at, b.title, b.authors, b.image_url
        FROM app.to_read t
        JOIN app.books b ON t.book_id = b.book_id
        WHERE t.user_id = $1
        ORDER BY t.created_at DESC;
    `;
    const res = await pool.query(query, [userId]);
    return res.rows;
};

const addToWishlist = async (userId, bookId) => {
    const query = `
        INSERT INTO app.to_read (user_id, book_id, created_at)
        VALUES ($1, $2, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id, book_id)
        DO NOTHING;
    `;

    await pool.query(query, [userId, bookId]);
};



const removeFromWishlist = async (userId, bookId) => {
    const query = `DELETE FROM app.to_read WHERE user_id = $1 AND book_id = $2`;
    await pool.query(query, [userId, bookId]);
};

module.exports = {
    getRatedBooks,
    getRatedBookById,
    updateRating,
    deleteRating,
    getWishlist,
    addToWishlist,
    removeFromWishlist
};
