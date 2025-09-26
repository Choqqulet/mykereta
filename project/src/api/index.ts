import axios from "axios";

const isProd = import.meta.env.PROD;
const BASE = isProd ? "" : (import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:3000");

export const axiosClient = axios.create({
  baseURL: BASE || "",
  withCredentials: true,
  headers: { "Content-Type": "application/json" },
});

function headers(token?: string) {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (token) h.Authorization = `Bearer ${token}`;
  return h;
}

async function get<T = unknown>(path: string, token?: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { method: "GET", credentials: "include", headers: headers(token) });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}
async function post<T = unknown>(path: string, body?: any, token?: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    credentials: "include",
    headers: headers(token),
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}
async function del<T = unknown>(path: string, token?: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { method: "DELETE", credentials: "include", headers: headers(token) });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

const api = {
  get,
  post,
  del,
  // auth helpers
  me: () => get<{ user: any | null }>("/api/auth/me"),
  logout: () => post("/api/auth/logout", {}),
  startGoogle: (state = "/dashboard") => `${BASE}/api/auth/google/start?state=${encodeURIComponent(state)}`,
};

export default api;