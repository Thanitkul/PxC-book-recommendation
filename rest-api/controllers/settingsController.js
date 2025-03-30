const express = require("express");
const router = express.Router();
const { exportUserData, deleteUserAndData } = require("../models/settingsModel");
const RouteProtection = require("../helpers/RouteProtection");

// GET /api/settings/export
// response: CSV file containing genres, rated books, and wishlist
router.get("/export", RouteProtection.verify, async (req, res) => {
    const userId = req.user.userId;
    try {
        const csvData = await exportUserData(userId);

        res.setHeader("Content-Type", "text/csv");
        res.setHeader("Content-Disposition", "attachment; filename=user_data.csv");
        res.send(csvData);
    } catch (err) {
        res.status(500).json({ message: "Failed to export user data", error: err.message });
    }
});

// DELETE /api/settings/delete-account
// request: {}
// response: { STRING message }
router.delete("/delete-account", RouteProtection.verify, async (req, res) => {
    const userId = req.user.userId;
    try {
        await deleteUserAndData(userId);
        res.clearCookie("token");  // assumes httpOnly cookie token
        res.status(200).json({ message: "Account deleted and logged out successfully." });
    } catch (err) {
        res.status(500).json({ message: "Failed to delete account", error: err.message });
    }
});

module.exports = router;
