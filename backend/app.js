import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import cookieParser from 'cookie-parser';

import authRouter from './src/routes/auth.js'; 
import apiRouter from './src/routes/api.js';       // vehicles/docs/expenses etc.

const app = express();

app.use(helmet());
app.use(morgan('tiny'));
app.use(express.json());
app.use(cookieParser());

// CORS
const allowed = (process.env.CORS_ORIGINS || '').split(',').map(s => s.trim()).filter(Boolean);
app.use(cors({
  origin: allowed.length ? allowed : true,
  credentials: true
}));

// health
app.get('/health', (_req, res) => res.json({ ok: true }));

// routes (note the /api prefix!)
app.use('/api/auth', authRouter);
app.use('/api', apiRouter);

export default app;