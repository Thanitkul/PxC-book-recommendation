const express = require("express");
const router = express.Router();

const authRoutes = require("./authRoutes");
const adminRoutes = require("./adminRoutes");
const libraryRoutes = require("./libraryRoutes");

router.use("/auth", authRoutes);
router.use("/admin", adminRoutes);
router.use("/library", libraryRoutes);

module.exports = router;