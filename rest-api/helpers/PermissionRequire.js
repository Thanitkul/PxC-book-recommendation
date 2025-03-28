class PermissionRequire {
    static verify(req, res, next) {
      try {
        const userPermission = req.user.permission
        if (userPermission === "admin") {
          return next();
        } else {
          return res.status(403).json({
            message: "Insufficient permission. Admin access required.",
          });
        }
      } catch (err) {
        return res.status(500).json({
          message: "Permission check failed",
          error: err.message,
        });
      }
    }
  }
  
module.exports = PermissionRequire;
  