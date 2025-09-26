import "dotenv/config";
import express from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import authRoutes from "./src/routes/auth.js";

const app = express();

const CORS_ORIGINS = (process.env.CORS_ORIGINS || "").split(",").map(s => s.trim()).filter(Boolean);
// Allow Vercel + localhost to send/receive cookies
app.use(cors({ origin: CORS_ORIGINS, credentials: true }));

app.use(cookieParser());
app.use(express.json());
app.get("/health", (_req, res) => res.json({ ok: true }));

app.use("/api", authRoutes); // <- /api/auth/google/start & /api/auth/google/callback

export default app;