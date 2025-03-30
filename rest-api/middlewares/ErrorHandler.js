const pg = require('pg');

function errorHandler(err, req, res, next) {
  console.error('Unhandled error:', err);

  // Database error
  if (err instanceof pg.DatabaseError) {
    return res.status(500).json({
      message: 'Database error',
      details: err.message
    });
  }

  // Remove or comment out the custom error unless you define it:
  // else if (err instanceof SomeCustomError) {
  //   return res.status(err.statusCode).json({ message: err.message });
  // }

  // General or unknown error
  return res.status(500).json({ message: 'Internal server error' });
}

module.exports = errorHandler;
