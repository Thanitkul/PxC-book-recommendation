const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const cookieParser = require("cookie-parser");

const apiRoutes = require("./routes"); // loads routes from routes/index.js
const errorHandler = require("./middlewares/ErrorHandler");

const app = express();

// Load env vars
dotenv.config();

// Middleware stack
const corsOptions = {
  credentials: true
};

app.use(cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(cookieParser());


// Basic test endpoint
app.get("/", (req, res) => {
  res.status(200).send("📚 Books Recommendation REST API is running.");
});

app.use("/api", apiRoutes);

app.use(errorHandler);

module.exports = app;
