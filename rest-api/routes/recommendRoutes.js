const express = require('express');
const router = express.Router();
const recommendController = require('../controllers/recommendController');

router.use('/', recommendController);

module.exports = router;