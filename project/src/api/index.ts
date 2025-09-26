import axios from "axios";

/**
 * Runtime base URL:
 *  - PROD (Vercel):   ''  (relative -> handled by vercel.json rewrites)
 *  - DEV (local):     VITE_BACKEND_URL (e.g. http://127.0.0.1:3000)
 */
const isProd = import.meta.env.PROD;
const ORIGIN =
  isProd ? "" : (import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:3000");

const join = (path: string) => `${ORIGIN}${path}`;

/** Axios client (use when you want interceptors, etc.) */
export const http = axios.create({
  baseURL: ORIGIN,                      // '' in prod, absolute in dev
  withCredentials: true,                // send/receive cookies
  headers: { "Content-Type": "application/json" },
});
export default http;                    // optional default export

/** Lightweight fetch wrappers (handy for simple JSON calls) */
type Json = Record<string, unknown>;

function jsonHeaders(token?: string): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

async function get<T = unknown>(path: string, token?: string): Promise<T> {
  const res = await fetch(join(path), {
    method: "GET",
    headers: jsonHeaders(token),
    credentials: "include",
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

async function post<T = unknown>(path: string, body: Json, token?: string): Promise<T> {
  const res = await fetch(join(path), {
    method: "POST",
    headers: jsonHeaders(token),
    credentials: "include",
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

async function del<T = unknown>(path: string, token?: string): Promise<T> {
  const res = await fetch(join(path), {
    method: "DELETE",
    headers: jsonHeaders(token),
    credentials: "include",
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

/** Export a simple REST helper object.
 *  (Named `api` to match previous imports like: `import { api } from '@/api'`)
 */
export const api = { get, post, del };

/** Domain helpers — adjust paths to match your backend routes */
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

/** Auth */
export const AuthAPI = {
  /** Start Google OAuth (optionally pass desired post-login path) */
  startGoogle: (redirect = "/dashboard") =>
    `${join("/api/auth/google/start")}?redirect=${encodeURIComponent(redirect)}`,

  /** Who am I (session check) */
  me: () => api.get("/api/auth/me"),

  /** Logout and clear cookie */
  logout: () => api.post("/api/auth/logout", {}),
};