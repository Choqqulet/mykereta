// project/src/api/index.ts
const isProd = import.meta.env.PROD;

// In dev we hit the explicit backend URL.
// In prod we rely on Vercel rewrite and call relative /api/*.
const BASE = isProd
  ? "" // relative
  : (import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:3000");

type Json = Record<string, unknown>;

function jsonHeaders(token?: string): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

async function get<T = unknown>(path: string, token?: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "GET",
    headers: jsonHeaders(token),
    credentials: "include",
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

async function post<T = unknown>(path: string, body: Json, token?: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: jsonHeaders(token),
    credentials: "include",
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

async function del<T = unknown>(path: string, token?: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "DELETE",
    headers: jsonHeaders(token),
    credentials: "include",
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const api = { get, post, del };

// Convenience helpers for your existing routes:
export const VehiclesAPI = {
  list: () => api.get("/api/vehicles"),
  create: (data: Json) => api.post("/api/vehicles", data),
  remove: (id: string) => api.del(`/api/vehicles/${id}`),
};

export const DocumentsAPI = {
  list: () => api.get("/api/documents"),
  create: (data: Json) => api.post("/api/documents", data),
  remove: (id: string) => api.del(`/api/documents/${id}`),
};

export const ExpensesAPI = {
  list: () => api.get("/api/expenses"),
  create: (data: Json) => api.post("/api/expenses", data),
  remove: (id: string) => api.del(`/api/expenses/${id}`),
};

// Auth flow typically hits your backend directly:
export const AuthAPI = {
  startGoogle: () => `${BASE || ""}/api/auth/google/start`,
  me: () => api.get("/api/auth/me"),
  logout: () => api.post("/api/auth/logout", {}),
};