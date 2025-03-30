const jwt = require("jsonwebtoken");

//verify authorization token
class RouteProtection {
  static verify(req, res, next) {
    try {
      const token = req.headers.authorization.split(" ")[1];
      if (!token) {
        return res.status(401).json({ message: "Unauthorized" }).end();
      }
      // Verify the token
      const decoded = jwt.verify(token, process.env.TOKENSECRET);
      req.user = { userId: decoded.id, permission: decoded.permission };
      return next();
    } catch (error) {
      console.log(error);
      res.status(401).json({ message: "Unauthorized" }).end();
    }
  }
}

module.exports = RouteProtection;
