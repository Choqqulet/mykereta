import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Try common locations for the Express app module
const candidates = ['./app.js', './src/app.js', './dist/app.js'];

let appModule = null;
for (const rel of candidates) {
  const full = resolve(__dirname, rel);
  if (existsSync(full)) {
    appModule = await import(full);
    break;
  }
}

if (!appModule?.default) {
  throw new Error(`Could not find app module. Tried: ${candidates.join(', ')}`);
}

const app = appModule.default;

// Heroku: listen only on PORT; do not pass host
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`API listening on 0.0.0.0:${port}`);
});