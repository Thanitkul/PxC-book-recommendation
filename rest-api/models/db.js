const { Pool } = require("pg");
require("dotenv").config();

const pool = new Pool({
  host: process.env.DB_HOST || "localhost",
  port: process.env.DB_PORT || 5432,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  max: 20,
  idleTimeoutMillis: 60000,
});

async function testDbConnection() {
  const res = await pool.query("SELECT NOW()");
  return res;
}

module.exports = {
  pool,
  testDbConnection
};
