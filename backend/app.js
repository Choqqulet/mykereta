import cors from 'cors';
import express from 'express';

const app = express();

const allowed = new Set([
  process.env.FRONTEND_URL,           // https://mykereta.vercel.app
  'http://localhost:5173',            // dev
  'http://127.0.0.1:5173'             // dev alt
].filter(Boolean));

app.use(cors({
  origin: (origin, cb) => {
    if (!origin || allowed.has(origin)) return cb(null, true);
    cb(new Error(`CORS blocked: ${origin}`));
  },
  credentials: true
}));

app.get('/health', (_req, res) => {
  res.status(200).json({ ok: true });
});

export default app;