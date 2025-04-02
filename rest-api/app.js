// app.js
const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const cookieParser = require("cookie-parser");

const apiRoutes = require("./routes"); 
const errorHandler = require("./middlewares/ErrorHandler");

const app = express();
dotenv.config();

// // Create a list of allowed origins:
// const allowedOrigins = [
//   'https://192.168.1.130:4200',
//   'https://192.168.1.131:4200'
// ];

// const corsOptions = {
//   origin: function (origin, callback) {
//     // If there's no Origin (like Postman or curl), either allow or block:
//     if (!origin) {
//       return callback(null, true);
//     }
//     if (allowedOrigins.includes(origin)) {
//       callback(null, true);
//     } else {
//       console.log('Blocked by CORS. Origin:', origin);
//       callback(new Error('Origin not allowed by CORS'));
//     }
//   }
// };

// allow localhost:4200 and localhost:4201
const allowedOrigins = [
  "https://localhost:4200",
  "https://localhost:4201",
];

const corsOptions = {
  origin: function (origin, callback) {
    // If there's no Origin (like Postman or curl), either allow or block:
    if (!origin) {
      return callback(null, true);
    }
    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      console.log("Blocked by CORS. Origin:", origin);
      callback(new Error("Origin not allowed by CORS"));
    }
  },
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
