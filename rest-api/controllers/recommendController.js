// routes/recommendController.js
const express = require("express");
const router = express.Router();
const RouteProtection = require("../helpers/RouteProtection");
const { setUserGenres, getRecommendations, getUserGenreChoices } = require("../models/recommendModel");
const { updateRating, deleteRating, addToWishlist, removeFromWishlist } = require("../models/libraryModel");

// GET /api/recommend/genre-choices
// response: { ARRAY<STRING>: genre_choices } // e.g., ["Fantasy", "Sci-Fi"]
router.get("/genre-choices", RouteProtection.verify, async (req, res) => {
    const userId = req.user.userId;
    try {
      const groupNames = await getUserGenreChoices(userId);
      res.status(200).json(groupNames);
    } catch (err) {
      res.status(500).json({ message: "Failed to retrieve genre preferences", error: err.message });
    }
  });


// POST /api/recommend/select-genres
// request: { ARRAY<STRING>: selected_genres } // e.g., ["Fantasy", "Sci-Fi"]
// response: { STRING message, ARRAY<INT>: genre_ids }
router.post("/select-genres", RouteProtection.verify, async (req, res) => {
  const userId = req.user.userId;
  const { selected_genres } = req.body;
  if (!Array.isArray(selected_genres) || selected_genres.length === 0) {
    return res.status(400).json({ message: "selected_genres must be a non-empty array" });
  }
  try {
    const genreIds = await setUserGenres(userId, selected_genres);
    res.status(200).json({ message: "Genres updated", genre_ids: genreIds });
  } catch (err) {
    res.status(500).json({ message: "Failed to set genres", error: err.message });
  }
});

// GET /api/recommend/list
// response: [{ INT book_id, STRING title, STRING authors, STRING image_url, INT? rating, BOOLEAN? wishlisted }]
router.get("/list", RouteProtection.verify, async (req, res) => {
  const userId = req.user.userId;
  try {
    const books = await getRecommendations(userId);
    res.json(books);
  } catch (err) {
    res.status(500).json({ message: "Failed to get recommendations", error: err.message });
  }
});

// PATCH /api/recommend/edit-rating
// request: { INT book_id, INT rating }
// response: { STRING message, INT book_id, INT rating }
router.patch("/edit-rating", RouteProtection.verify, async (req, res) => {
  const userId = req.user.userId;
  const { book_id, rating } = req.body;
  if (!book_id || !rating || rating < 1 || rating > 5) {
    return res.status(400).json({ message: "Invalid book ID or rating (must be 1-5)." });
  }
  try {
    const result = await updateRating(userId, book_id, rating);
    res.status(200).json({ message: "Rating updated", ...result });
  } catch (err) {
    res.status(500).json({ message: "Failed to update rating", error: err.message });
  }
});

// DELETE /api/recommend/delete-rating
// request: { INT book_id }
// response: { STRING message, INT book_id }
router.delete("/delete-rating", RouteProtection.verify, async (req, res) => {
  const userId = req.user.userId;
  const { book_id } = req.body;
  if (!book_id) return res.status(400).json({ message: "book_id required" });
  try {
    await deleteRating(userId, book_id);
    res.status(200).json({ message: "Rating deleted", book_id });
  } catch (err) {
    res.status(500).json({ message: "Failed to delete rating", error: err.message });
  }
});

// POST /api/recommend/add-wishlist
// request: { INT book_id }
// response: { STRING message, INT book_id }
router.post("/add-wishlist", RouteProtection.verify, async (req, res) => {
  const userId = req.user.userId;
  const { book_id } = req.body;
  if (!book_id) return res.status(400).json({ message: "book_id required" });
  try {
    await addToWishlist(userId, book_id);
    res.status(200).json({ message: "Book added to wishlist", book_id });
  } catch (err) {
    res.status(500).json({ message: "Failed to add to wishlist", error: err.message });
  }
});

// DELETE /api/recommend/delete-wishlist
// request: { INT book_id }
// response: { STRING message, INT book_id }
router.delete("/delete-wishlist", RouteProtection.verify, async (req, res) => {
  const userId = req.user.userId;
  const { book_id } = req.body;
  if (!book_id) return res.status(400).json({ message: "book_id required" });
  try {
    await removeFromWishlist(userId, book_id);
    res.status(200).json({ message: "Book removed from wishlist", book_id });
  } catch (err) {
    res.status(500).json({ message: "Failed to remove from wishlist", error: err.message });
  }
});

module.exports = router;
