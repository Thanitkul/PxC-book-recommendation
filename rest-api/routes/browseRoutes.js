const express = require('express');
const router = express.Router();
const browseController = require('../controllers/browseController');

router.use('/', browseController);

module.exports = router;