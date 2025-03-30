const express = require("express");
const router = express.Router();
const RouteProtection = require("../helpers/RouteProtection");
const { getRatedBooks, updateRating, deleteRating, getWishlist, removeFromWishlist, getRatedBookById } = require("../models/libraryModel");

// GET /api/library/list-ratings
// response: [{ INT: book_id, INT: rating, STRING: created_at, STRING: title, STRING: authors, STRING: image_url }]
router.get("/list-ratings", RouteProtection.verify, async (req, res) => {
    const userId = req.user.userId;
    try {
        const books = await getRatedBooks(userId);
        res.json(books);
    } catch (err) {
        res.status(500).json({ message: "Failed to fetch ratings", error: err.message });
    }
});

// PATCH /api/library/edit-ratings
// request: { INT: book_id, INT: rating }
// response: { INT: book_id, INT: rating, STRING: created_at, STRING: title, STRING: authors, STRING: image_url }
router.patch("/edit-ratings", RouteProtection.verify, async (req, res) => {
    const userId = req.user.userId;
    const { book_id, rating } = req.body;
    if (!book_id || !rating || rating < 1 || rating > 5) {
        return res.status(400).json({ message: "Invalid book ID or rating (must be 1-5)." });
    }

    try {
        await updateRating(userId, book_id, rating);
        const updated = await getRatedBookById(userId, book_id);
        res.status(200).json(updated);
    } catch (err) {
        res.status(500).json({ message: "Failed to update rating", error: err.message });
    }
});

// DELETE /api/library/delete-ratings
// request: { INT: book_id }
// response: { STRING: message }
router.delete("/delete-ratings", RouteProtection.verify, async (req, res) => {
    const userId = req.user.userId;
    const { book_id } = req.body;
    if (!book_id) return res.status(400).json({ message: "book_id required" });

    try {
        await deleteRating(userId, book_id);
        res.json({ message: "Rating deleted and book removed from library" });
    } catch (err) {
        res.status(500).json({ message: "Failed to delete rating", error: err.message });
    }
});

// GET /api/library/list-wishlist
// response: [{ INT: book_id, STRING: created_at, STRING: title, STRING: authors, STRING: image_url }]
router.get("/list-wishlist", RouteProtection.verify, async (req, res) => {
    const userId = req.user.userId;
    try {
        const books = await getWishlist(userId);
        res.json(books);
    } catch (err) {
        res.status(500).json({ message: "Failed to fetch wishlist", error: err.message });
    }
});

// DELETE /api/library/delete-wishlist
// request: { INT: book_id }
// response: { STRING: message }
router.delete("/delete-wishlist", RouteProtection.verify, async (req, res) => {
    const userId = req.user.userId;
    const { book_id } = req.body;
    if (!book_id) return res.status(400).json({ message: "book_id required" });

    try {
        await removeFromWishlist(userId, book_id);
        res.json({ message: "Book removed from wishlist" });
    } catch (err) {
        res.status(500).json({ message: "Failed to remove from wishlist", error: err.message });
    }
});

module.exports = router;
