const pg = require('pg');

function errorHandler(err, req, res, next) {
    console.error('Unhandled error:', err);
  
    // Customize the error response based on the error type
    // database error
    if (err instanceof pg.DatabaseError) {
      // Database error
      return res.status(500).json({ message: 'Database error', details: err.message });
    } else if (err instanceof SomeCustomError) {
      // Your custom errors
      return res.status(err.statusCode).json({ message: err.message });
    } else {
      // General or unknown error
      return res.status(500).json({ message: 'Internal server error' });
    }
  }
  
  module.exports = errorHandler;
  