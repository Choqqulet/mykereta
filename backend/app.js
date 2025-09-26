import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";

import authRouter from "./routes/auth.js";
import config from "./config/config.js";

const app = express();

// Heroku needs this for Secure cookies behind proxy
app.set("trust proxy", 1);

// CORS: allow your frontend and localhost
const origins = [
  config.frontendUrl,           // e.g. https://mykereta.vercel.app
  "http://localhost:5173",
];
app.use(cors({ origin: origins, credentials: true }));

app.use(express.json());
app.use(cookieParser());

// Health
app.get("/health", (_req, res) => res.status(200).json({ ok: true }));

// 🚀 Mount OAuth routes (this removes the 404)
app.use("/api/auth", authRouter);

export default app;