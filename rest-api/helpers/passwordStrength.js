class PasswordStrengthChecker {
    // password must be at least 6 characters long, contain at least one uppercase letter, one lowercase letter, one number, and one special character
    static check(password) {
        if (password.length < 6) {
            return false;
        }
        if (!/[A-Z]/.test(password)) {
            return false;
        }
        if (!/[a-z]/.test(password)) {
            return false;
        }
        if (!/[0-9]/.test(password)) {
            return false;
        }
        if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
            return false;
        }
        return true;
    }
}

module.exports = PasswordStrengthChecker;
// Usage example

// const PasswordStrengthChecker = require('./path/to/PasswordStrengthChecker');
// const password = "Password123!";
// if (PasswordStrengthChecker.check(password)) {
