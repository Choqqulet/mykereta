import app from './app.js';
import { createProxyMiddleware } from "http-proxy-middleware";

const PORT = process.env.PORT || 3000;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`API listening on 0.0.0.0:${PORT}`);
});

app.use(
  "/api/ml",
  createProxyMiddleware({
    target: process.env.ML_API_BASE,  // e.g. https://mykereta-ml.herokuapp.com
    changeOrigin: true,
    pathRewrite: { "^/api/ml": "" },  // /api/ml/plate -> /plate
    timeout: 60_000,
  })
);