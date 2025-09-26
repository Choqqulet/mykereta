const BASE = import.meta.env.VITE_BACKEND_URL ?? ""; // Heroku in dev, "" in prod via Vercel rewrite

async function j<T>(p: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${p}`, { credentials: "include", ...(init ?? {}) });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<T>;
}

export async function mlHealth() {
  return j<{ ok: boolean }>("/api/ml/health");
}

export async function mlSummarize(text: string) {
  return j<{ summary: string }>("/api/ml/summarize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

export async function mlPlate(file: File) {
  const form = new FormData();
  form.append("file", file);
  return j<{ plate: string }>("/api/ml/plate", { method: "POST", body: form });
}