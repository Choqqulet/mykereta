// ESM module
import { Router } from "express";
import passport from "passport";

const router = Router();

/**
 * GET /api/auth/google/start
 * Optional: ?redirect=/dashboard (defaults to /dashboard)
 * Stores the redirect target in the session and kicks off Google OAuth.
 */
router.get(
  "/google/start",
  (req, _res, next) => {
    // remember where to go after login
    req.session.redirectTo = (req.query.redirect && String(req.query.redirect)) || "/dashboard";
    next();
  },
  passport.authenticate("google", {
    scope: ["profile", "email"],
    prompt: "select_account",
    session: true, // we’re using express-session
  })
);

/**
 * GET /api/auth/google/callback
 * On success: redirects to FRONTEND_URL + redirectTo
 * On failure: goes back to /signin
 */
router.get(
  "/google/callback",
  passport.authenticate("google", {
    failureRedirect: `${process.env.FRONTEND_URL || "http://localhost:5173"}/signin`,
    session: true,
  }),
  (req, res) => {
    const to = req.session.redirectTo || "/dashboard";
    // (optional) you can also set a custom cookie here if your frontend expects it
    // res.cookie("token", req.user?.jwt, { httpOnly: true, sameSite: "none", secure: true });

    res.redirect(`${process.env.FRONTEND_URL || "http://localhost:5173"}${to}`);
  }
);

/**
 * POST /api/auth/logout
 * Clears session and auth cookies.
 */
router.post("/logout", (req, res) => {
  req.logout?.(() => {});        // passport
  req.session?.destroy(() => {}); // express-session
  res.clearCookie("connect.sid"); // default session cookie name
  res.clearCookie("token");       // if set a custom token
  res.status(200).json({ ok: true });
});

export default router;