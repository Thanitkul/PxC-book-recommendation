const express = require("express");
const router = express.Router();

const authRoutes = require("./authRoutes");
const adminRoutes = require("./adminRoutes");
const libraryRoutes = require("./libraryRoutes");
const browseRoutes = require("./browseRoutes");
const settingRoutes = require("./settingRoutes");

router.use("/auth", authRoutes);
router.use("/admin", adminRoutes);
router.use("/library", libraryRoutes);
router.use("/browse", browseRoutes);
router.use("/settings", settingRoutes);

module.exports = router;