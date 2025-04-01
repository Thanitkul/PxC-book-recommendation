// recsys/recsysClient.js
const crypto = require("crypto");
const axios = require("axios");
require("dotenv").config();

const RECSYS_URL = process.env.RECSYS_URL || "http://localhost:8081";
const RECSYS_HMAC_SECRET = process.env.RECSYS_HMAC_SECRET;

function createHMACSignature({ method, path, timestamp, body }) {
  const payload = `${method}\n${path}\n${timestamp}\n${body}`;
  console.log("Payload for HMAC:", payload);
  return crypto.createHmac("sha256", RECSYS_HMAC_SECRET).update(payload).digest("hex");
}

async function getRecommendationFromEngine(userId) {
  const path = "/api/recsys/recommend";
  const url = RECSYS_URL + path;
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const bodyObj = { user_id: userId };
  const bodyStr = JSON.stringify(bodyObj);

  console.log("Request URL:", url);

  const signature = createHMACSignature({
    method: "POST",
    path,
    timestamp,
    body: bodyStr,
  });

  try {
    const response = await axios.post(url, bodyObj, {
      headers: {
        "X-Timestamp": timestamp,
        "X-Signature": signature,
        "Content-Type": "application/json",
      },
    });

    return response.data.recommendations;
  } catch (err) {
    console.error("Error fetching recommendations:", err.message);
    return [];
  }
}

module.exports = {
  getRecommendationFromEngine,
};
