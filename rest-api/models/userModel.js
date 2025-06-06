const { pool } = require("./db");

// Get user by username or ID
const getUserByUsername = async (username) => {
  const query = `
    SELECT id, username, password, permission
    FROM app.users
    WHERE username = $1
  `;
  const res = await pool.query(query, [username]);
  if (res.rows.length === 0) {
    return null;
  }
  return res.rows[0];
};

// Create new user
const createUser = async ({ username, hashedPassword, permission }) => {
  
  const query = `
    INSERT INTO app.users (username, password, permission)
    VALUES ($1, $2, $3)
    RETURNING id, username, permission
  `;
  const res = await pool.query(query, [username, hashedPassword, permission]);

  if (res.rows.length === 0) {
    return null;
  }
  return res.rows[0];
};

const getUserInfo = async (userId) => {
    const query = `
    SELECT username, selected_genre
    FROM app.users
    WHERE id = $1
    `;

    const res = await pool.query(query, [userId]);
    console.log(res.rows);
    if (res.rows.length === 0) {
        return null;
    }
    return res.rows[0];
}

const getAllAdmins = async () => {
    const query = `
    SELECT id, username
    FROM app.users
    WHERE permission = 'admin'
    `;

    const res = await pool.query(query);

    if (res.rows.length === 0) {
        return null;
    }

    return res.rows;
}

const deleteUser = async (userId) => {
    const query = `
    DELETE FROM app.users
    WHERE id = $1
    `;

    // TODO delete all user data from all tables

    const res = await pool.query(query, [userId]);

    if (res.rowCount === 0) {
        return null;
    }

    return res.rowCount;
}



module.exports = {
    getUserByUsername,
    createUser,
    getUserInfo,
    getAllAdmins,
    deleteUser
};


