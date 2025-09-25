import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

// Resolve current dir in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Try common locations for the Express app module
const candidates = ['./src/app.js', './app.js', './dist/app.js'];

let appModule = null;
for (const rel of candidates) {
  const full = resolve(__dirname, rel);
  if (existsSync(full)) {
    appModule = await import(full);
    break;
  }
}

if (!appModule?.default) {
  throw new Error(
    `Could not find app module. Tried: ${candidates.join(', ')}`
  );
}

const app = appModule.default;

// Heroku: must listen on process.env.PORT and bind to 0.0.0.0 (omit host)
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`API listening on 0.0.0.0:${port}`);
});