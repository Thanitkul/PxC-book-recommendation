const { pool } = require("./db");
const { parse } = require("json2csv");

const exportUserData = async (userId) => {
    const genresQuery = `
        SELECT unnest(genre_ids) AS genre_id
        FROM app.users WHERE id = $1;
    `;
    const ratedBooksQuery = `
        SELECT r.book_id, b.title, b.authors, r.rating
        FROM app.ratings r
        JOIN app.books b ON r.book_id = b.book_id
        WHERE r.user_id = $1;
    `;
    const wishlistQuery = `
        SELECT t.book_id, b.title, b.authors
        FROM app.to_read t
        JOIN app.books b ON t.book_id = b.book_id
        WHERE t.user_id = $1;
    `;

    const genres = await pool.query(genresQuery, [userId]);
    const ratings = await pool.query(ratedBooksQuery, [userId]);
    const wishlist = await pool.query(wishlistQuery, [userId]);

    const rows = [];

    for (const g of genres.rows) {
        rows.push({ Section: "Genres", "Book ID": g.genre_id });
    }

    for (const r of ratings.rows) {
        rows.push({
            Section: "Ratings",
            "Book ID": r.book_id,
            Title: r.title,
            Authors: r.authors,
            Rating: r.rating
        });
    }

    for (const w of wishlist.rows) {
        rows.push({
            Section: "Wishlist",
            "Book ID": w.book_id,
            Title: w.title,
            Authors: w.authors
        });
    }

    return parse(rows, { fields: ["Section", "Book ID", "Title", "Authors", "Rating"] });
};

const deleteUserAndData = async (userId) => {
    const client = await pool.connect();
    try {
        await client.query("BEGIN");

        await client.query(`DELETE FROM app.ratings WHERE user_id = $1`, [userId]);
        await client.query(`DELETE FROM app.to_read WHERE user_id = $1`, [userId]);
        await client.query(`DELETE FROM app.users WHERE id = $1`, [userId]);

        await client.query("COMMIT");
    } catch (err) {
        await client.query("ROLLBACK");
        throw err;
    } finally {
        client.release();
    }
};

module.exports = {
    exportUserData,
    deleteUserAndData
};
