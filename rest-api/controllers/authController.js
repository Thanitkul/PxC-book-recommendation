const express = require("express");
const router = express.Router();
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const { getUserByUsername, createUser, getUserInfo } = require("../models/userModel");
const RouteProtection = require("../helpers/RouteProtection");
const PasswordStrengthChecker = require("../helpers/passwordStrength");


// POST /api/auth/signup
// body = { STRING username, STRING password }
// response = { STRING token }
router.post("/signup", async (req, res) => {
    try {
        const { username, password } = req.body;

        const normalizedUsername = username.trim().toLowerCase();
        const normalizedPassword = password.trim();

        if (!normalizedUsername || !normalizedPassword) {
            return res.status(400).json({ message: "Missing fields" });
        }

        const existingUser = await getUserByUsername(normalizedUsername);
        if (existingUser) {
            return res.status(409).json({ message: "Username already exists" });
        }

        if (!PasswordStrengthChecker.check(normalizedPassword)) {
            return res.status(400).json({ message: "Password is too weak" });
        }

        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(normalizedPassword, salt);
        const newUser = await createUser({ username: normalizedUsername, hashedPassword: hashedPassword, permission: "user" });
        if (!newUser) {
            return res.status(500).json({ message: "Error creating user" });
        }

        const token = jwt.sign({ id: newUser.id, permission: "user" }, process.env.TOKENSECRET, {
            expiresIn: "24h",
        });

        res.cookie("token", token, {
            httpOnly: true,
            secure: false, // Set to true in production
            sameSite: 'none',
            maxAge: 24 * 60 * 60 * 1000 // 1 day in ms
        });
        res.status(200).json({ message: "Login successful" });
    } catch (err) {
        console.error(err);
        res.status(500).json({ message: "Internal server error" });
    }
});


// POST /api/auth/signin
// body = { STRING username, STRING password }
// response = { STRING token }
router.post("/signin", async (req, res) => {
    try {
        const { username, password } = req.body;

        const normalizedUsername = username.trim().toLowerCase();
        const normalizedPassword = password.trim();
        if (!normalizedUsername || !normalizedPassword) {
            return res.status(400).json({ message: "Missing fields" });
        }

        const user = await getUserByUsername(normalizedUsername);
        if (!user) {
            return res.status(401).json({ message: "Invalid credentials" });
        }

        const isMatch = await bcrypt.compare(normalizedPassword, user.password);
        if (!isMatch) {
            return res.status(401).json({ message: "Invalid credentials" });
        }

        const token = jwt.sign({ id: user.id, permission: user.permission }, process.env.TOKENSECRET, {
            expiresIn: "24h",
        });

        res.cookie("token", token, {
            httpOnly: true,
            secure: false, // Set to true in production
            sameSite: 'none',
            maxAge: 24 * 60 * 60 * 1000 // 1 day in ms
        });
        res.status(200).json({ message: "Login successful" });
    } catch (err) {
        console.error(err);
        res.status(500).json({ message: "Internal server error" });
    }
});


// GET /api/auth/signout
// response = { STRING message }
router.get("/signout", RouteProtection.verify, (req, res) => {
    try {
        res.clearCookie("token");
        res.status(200).json({ message: "Logout successful" });
    } catch (err) {
        console.error(err);
        res.status(500).json({ message: "Internal server error" });
    }
});

// GET /api/auth/user-info
// response = { STRING username, BOOLEAN selectedGenre }
router.get("/user-info", RouteProtection.verify, async (req, res) => {
    try {
        const userInfo = await getUserInfo(req.user.userId);
        if (!userInfo) {
            return res.status(404).json({ message: "User not found" });
        }
        res.status(200).json({ username: userInfo.username, selectedGenre: userInfo.selected_genre });
    } catch (err) {
        console.error(err);
        res.status(500).json({ message: "Internal server error" });
    }
});


module.exports = router;