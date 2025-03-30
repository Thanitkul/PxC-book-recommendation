// models/recommendModel.js
const { pool } = require("./db");
const { getRecommendationFromEngine } = require("../recsys/recsysClient");

const GENRE_TAG_MAP = {
  "Children": [441, 1437, 1420, 1438],
  "Classics": [467],
  "Comics/Graphic": [507, 510, 508, 511, 509],
  "Fantasy": [945, 1846],
  "Fiction": [52, 526, 1149, 1773, 2413],
  "Historical": [1239],
  "Horror": [2614, 1269, 2723, 1105, 1107],
  "Mystery/Thriller": [689, 2025, 598],
  "Non-Fiction": [1761, 1764, 214, 1592, 871],
  "Romance": [437, 353],
  "Sci-Fi": [772],
  "Young Adult": [2715, 267, 2488, 2490, 2489]
};

const getUserGenreChoices = async (userId) => {
  // return all keys of GENRE_TAG_MAP
  return Object.keys(GENRE_TAG_MAP);
};

const setUserGenres = async (userId, genreNames) => {
  const genreIds = genreNames.flatMap(name => GENRE_TAG_MAP[name] || []);
  const query = `
    UPDATE app.users
    SET genre_ids = $1, selected_genre = true
    WHERE id = $2
    RETURNING genre_ids
  `;
  const res = await pool.query(query, [genreIds, userId]);
  return res.rows[0].genre_ids;
};

const getRecommendations = async (userId) => {
  const recommendedBookIds = await getRecommendationFromEngine(userId); // expects [book_id, ...]
  const query = `
    SELECT
      b.book_id,
      b.title,
      b.authors,
      b.image_url,
      r.rating,
      CASE WHEN t.book_id IS NOT NULL THEN true ELSE false END AS wishlisted
    FROM app.books b
    LEFT JOIN app.ratings r ON b.book_id = r.book_id AND r.user_id = $1
    LEFT JOIN app.to_read t ON b.book_id = t.book_id AND t.user_id = $1
    WHERE b.book_id = ANY($2)
  `;
  const res = await pool.query(query, [userId, recommendedBookIds]);

  console.log("Recommended Book IDs:", recommendedBookIds);
  console.log("Query Result:", res.rows);

  const orderMap = Object.fromEntries(recommendedBookIds.map((id, i) => [id, i]));
  return res.rows.sort((a, b) => orderMap[a.book_id] - orderMap[b.book_id]);
};

module.exports = {
  getUserGenreChoices,
  setUserGenres,
  getRecommendations
};
