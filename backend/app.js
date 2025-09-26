import "dotenv/config";
import express from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import cookieParser from "cookie-parser";
import session from "express-session";
import passport from "passport";

import authRoutes from "./src/routes/auth.js";
import { initGooglePassport } from "./src/controllers/authController.js";

const app = express();

// Heroku sits behind a proxy (needed for secure cookies)
app.set("trust proxy", 1);

app.use(
  cors({
    origin: (_origin, cb) => cb(null, true),
    credentials: true,
  })
);

app.use(
  helmet({
    crossOriginResourcePolicy: { policy: "same-origin" },
  })
);

app.use(morgan("tiny"));
app.use(express.json());
app.use(cookieParser());

// Minimal session for passport (required for OAuth redirect round-trip)
app.use(
  session({
    secret: process.env.SESSION_SECRET || "dev",
    resave: false,
    saveUninitialized: false,
    cookie: {
      httpOnly: true,
      sameSite: "none",
      secure: true, // Heroku is HTTPS
    },
  })
);

app.use(passport.initialize());
app.use(passport.session());

// IMPORTANT: init strategy before mounting routes
initGooglePassport();

// Health first
app.get("/health", (_req, res) => res.json({ ok: true }));

// Auth routes under /api/auth (so /api/auth/google/start, /api/auth/google/callback)
app.use("/api/auth", authRoutes);

export default app;