const express = require("express");
const router = express.Router();
const RouteProtection = require("../helpers/RouteProtection");
const PermissionRequire = require("../helpers/PermissionRequire");
const bcrypt = require("bcrypt");
const { getAllBooksForAdmin, updateBookVisibility } = require("../models/bookModel");
const { getAllAdmins, createUser, deleteUser, getUserByUsername } = require("../models/userModel");
const PasswordStrengthChecker = require("../helpers/passwordStrength");

// GET /api/admin/list-all-books
// response: { INT: book_id, STRING: title, STRING: authors, BOOLEAN: is_visible }
router.get("/list-all-books", RouteProtection.verify, PermissionRequire.verify, async (req, res) => {
  try {
    const books = await getAllBooksForAdmin();
    if (!books) {
      return res.status(404).json({ message: "No books found" });
    }

    return res.status(200).json(books);
  } catch (err) {
    return res.status(500).json({
      message: "Failed to fetch books",
      error: err.message,
    });
  }
});


// PATCH /api/admin/update-book-visibility
// request: { INT: book_id, BOOLEAN: is_visible }
// response: { STRING: message }
router.patch("/update-book-visibility", RouteProtection.verify, PermissionRequire.verify, async (req, res) => {
  const { book_id, is_visible } = req.body;

  if (typeof book_id !== "number" || typeof is_visible !== "boolean") {
    return res.status(400).json({
      message: "Invalid input. Book ID must be a number and visibility must be a boolean.",
    });
  }

  try {
    const updatedBook = await updateBookVisibility(book_id, is_visible);
    if (!updatedBook) {
      return res.status(404).json({ message: "Book not found" });
    }

    return res.status(200).json(updatedBook);
  } catch (err) {
    return res.status(500).json({
      message: "Failed to update book visibility",
      error: err.message,
    });
  }
});


// GET /api/admin/list-all-admins
// response: { INT: id, STRING: username }
router.get("/list-all-admins", RouteProtection.verify, PermissionRequire.verify, async (req, res) => {
  try {
    const admins = await getAllAdmins();
    if (!admins) {
      return res.status(404).json({ message: "No admins found" });
    }

    return res.status(200).json(admins);
  } catch (err) {
    return res.status(500).json({
      message: "Failed to fetch admins",
      error: err.message,
    });
  }
});


// POST /api/admin/add-admin
// request: { STRING: username, STRING: password }
// response: { INT id, STRING: username }
router.post("/add-admin", RouteProtection.verify, PermissionRequire.verify, async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({
      message: "Username and password are required.",
    });
  }

  try {
    // check if username already exists
    const existingAdmin = await getUserByUsername(username);
    if (existingAdmin) {
        return res.status(400).json({
            message: "Username already exists.",
        });
        }

    // check password strength
    if (!PasswordStrengthChecker.check(password)) {
      return res.status(400).json({
        message: "Password must be at least 6 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character.",
        });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newAdmin = await createUser({ username, hashedPassword, permission: "admin" });

    if (!newAdmin) {
      return res.status(500).json({ message: "Failed to create admin" });
    }

    return res.status(201).json(newAdmin);
  } catch (err) {
    return res.status(500).json({
      message: "Failed to create admin",
      error: err.message,
    });
  }
});

// DELETE /api/admin/delete-admin
// request: { INT: id }
// response: { STRING: message }
router.delete("/delete-admin", RouteProtection.verify, PermissionRequire.verify, async (req, res) => {
  const { id } = req.body;

  if (!id) {
    return res.status(400).json({
      message: "Admin ID is required.",
    });
  }

  try {
    const result = await deleteUser(id);
    if (!result) {
      return res.status(404).json({ message: "Admin not found" });
    }

    return res.status(200).json({ message: "Admin deleted successfully" });
  } catch (err) {
    return res.status(500).json({
      message: "Failed to delete admin",
      error: err.message,
    });
  }
});


module.exports = router;