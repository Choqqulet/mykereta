// Provides canonical and alias endpoints:
//   GET /api/auth/google           -> starts the flow
//   GET /api/auth/google/start     -> starts the flow (alias)
//   GET /api/auth/google/callback  -> callback from Google
import express from "express";
import passport from "passport";
import { initGooglePassport, startGoogle, googleCallback } from "../controllers/authController.js";
import { me } from "../controllers/authController.js";

const router = express.Router();
initGooglePassport();
router.use(passport.initialize());

router.get("/google", startGoogle);       // alias
router.get("/google/start", startGoogle); // canonical
router.get("/google/callback", ...googleCallback);
router.get("/me", me);

export default router;