// browseController.js
const express = require("express");
const router = express.Router();
const RouteProtection = require("../helpers/RouteProtection");
const {
  getAllBooks,
  editRating,
  deleteRating,
  addToWishlist,
  removeFromWishlist
} = require("../models/browseModel");

// GET /api/browse/list-books
// request: (query param) STRING: sort ("popularity" | "rating" | "wishlist" | "title")
// response: [{ INT: book_id, STRING: title, STRING: authors, STRING: image_url, REAL: average_rating, INT: popularity, INT/null: user_rating, BOOLEAN: in_wishlist }]
router.get("/list-books", RouteProtection.verify, async (req, res) => {
  const userId = req.user.userId;
  const sort = req.query.sort || "popularity";
  try {
    const books = await getAllBooks(userId, sort);
    res.json(books);
  } catch (err) {
    res.status(500).json({ message: "Failed to fetch books", error: err.message });
  }
});

// PATCH /api/browse/edit-rating
// request: { INT: book_id, INT: rating }
// response: { STRING: message, INT: book_id, INT: rating }
router.patch("/edit-rating", RouteProtection.verify, async (req, res) => {
  const userId = req.user.userId;
  const { book_id, rating } = req.body;
  if (!book_id || !rating || rating < 1 || rating > 5) {
    return res.status(400).json({ message: "Invalid book ID or rating (must be 1-5)." });
  }

  try {
    const result = await editRating(userId, book_id, rating);
    res.status(200).json({ message: "Rating updated", ...result });
  } catch (err) {
    res.status(500).json({ message: "Failed to update rating", error: err.message });
  }
});

// DELETE /api/browse/delete-rating
// request: { INT: book_id }
// response: { STRING: message, INT: book_id, deleted: true }
router.delete("/delete-rating", RouteProtection.verify, async (req, res) => {
  const userId = req.user.userId;
  const { book_id } = req.body;
  if (!book_id) return res.status(400).json({ message: "book_id required" });

  try {
    await deleteRating(userId, book_id);
    res.json({ message: "Rating deleted", book_id, deleted: true });
  } catch (err) {
    res.status(500).json({ message: "Failed to delete rating", error: err.message });
  }
});

// POST /api/browse/add-wishlist
// request: { INT: book_id }
// response: { STRING: message, INT: book_id, in_wishlist: true }
router.post("/add-wishlist", RouteProtection.verify, async (req, res) => {
  const userId = req.user.userId;
  const { book_id } = req.body;
  if (!book_id) return res.status(400).json({ message: "book_id required" });

  try {
    await addToWishlist(userId, book_id);
    res.json({ message: "Book added to wishlist", book_id, in_wishlist: true });
  } catch (err) {
    res.status(500).json({ message: "Failed to add to wishlist", error: err.message });
  }
});

// DELETE /api/browse/delete-wishlist
// request: { INT: book_id }
// response: { STRING: message, INT: book_id, in_wishlist: false }
router.delete("/delete-wishlist", RouteProtection.verify, async (req, res) => {
  const userId = req.user.userId;
  const { book_id } = req.body;
  if (!book_id) return res.status(400).json({ message: "book_id required" });

  try {
    await removeFromWishlist(userId, book_id);
    res.json({ message: "Book removed from wishlist", book_id, in_wishlist: false });
  } catch (err) {
    res.status(500).json({ message: "Failed to remove from wishlist", error: err.message });
  }
});

module.exports = router;
