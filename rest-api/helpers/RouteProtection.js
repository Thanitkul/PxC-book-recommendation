const jwt = require("jsonwebtoken");

//verify authorization token
class RouteProtection {
  static verify(req, res, next) {
    try {
      console.log("Verifying token...");
      console.log(req.headers);
      const token = req.headers.cookie
        .split("; ")
        .find((row) => row.startsWith("token="))
        .split("=")[1];
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
