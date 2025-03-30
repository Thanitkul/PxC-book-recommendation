const express = require('express');
const router = express.Router();
const settingsController = require('../controllers/settingsController');

router.use('/', settingsController);

module.exports = router;