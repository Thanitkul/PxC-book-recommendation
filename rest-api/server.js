const PORT = process.env.PORT || 8080;
const app = require("./app");
const https = require("https");
const fs = require("fs");

const { testDbConnection } = require("./models/db.js");

// Load HTTPS credentials
const privateKey = fs.readFileSync("./ssl/key.pem", "utf8");
const certificate = fs.readFileSync("./ssl/cert.pem", "utf8");
const credentials = { key: privateKey, cert: certificate };

const server = https.createServer(credentials, app);

async function startup() {
  try {
    testDbConnection();
    console.log("Database connection successful");

    server.listen(PORT, "0.0.0.0", () => {
      console.log(`Service listening on port ${PORT}`);
    });

  } catch (error) {
    console.error("Database connection failed", error);
    process.exit(1); // Exit the process with a failure code
  }
}

startup();

process.on('uncaughtException', (err) => {
  console.error('There was an uncaught error', err);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});
