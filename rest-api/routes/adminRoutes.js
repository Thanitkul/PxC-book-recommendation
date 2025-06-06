const express = require('express');
const router = express.Router();
const adminController = require('../controllers/adminController');

router.use('/', adminController);

module.exports = router;