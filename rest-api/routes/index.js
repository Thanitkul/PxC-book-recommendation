const express = require("express");
const router = express.Router();

const authRoutes = require("./authRoutes");
const adminRoutes = require("./adminRoutes");
const libraryRoutes = require("./libraryRoutes");
const browseRoutes = require("./browseRoutes");
const settingRoutes = require("./settingRoutes");
const recommendRoutes = require("./recommendRoutes");

router.use("/auth", authRoutes);
router.use("/admin", adminRoutes);
router.use("/library", libraryRoutes);
router.use("/browse", browseRoutes);
router.use("/settings", settingRoutes);
router.use("/recommend", recommendRoutes);

module.exports = router;