const express = require('express');
const router = express.Router();
const libraryController = require('../controllers/libraryController');

router.use('/', libraryController);

module.exports = router;