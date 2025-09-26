import { Router } from "express";
import passport from "passport";

const router = Router();

// Start OAuth; optionally store redirect target in session
router.get(
  "/google/start",
  (req, _res, next) => {
    req.session.redirectTo = req.query.redirect || "/dashboard";
    next();
  },
  passport.authenticate("google", {
    scope: ["profile", "email"],
    prompt: "select_account",
    session: true,
  })
);

// Callback; on success, redirect to FE and (optionally) set cookie in controller
router.get(
  "/google/callback",
  passport.authenticate("google", {
    failureRedirect: `${process.env.FRONTEND_URL || "http://localhost:5173"}/signin`,
    session: true,
  }),
  (req, res) => {
    const to = req.session.redirectTo || "/dashboard";
    res.redirect(`${process.env.FRONTEND_URL || "http://localhost:5173"}${to}`);
  }
);

router.post("/logout", (req, res) => {
  req.logout?.(() => {});
  req.session?.destroy(() => {});
  res.clearCookie("token");
  res.status(200).json({ ok: true });
});

export default router;