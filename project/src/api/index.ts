import axios from "axios";

const isProd = import.meta.env.PROD;
const DEV_BASE = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:3000";
export const BASE = isProd ? "" : DEV_BASE;

export const http = axios.create({
  baseURL: BASE || undefined,
  withCredentials: true,
  headers: { "Content-Type": "application/json" },
});

// 👇 add this line so legacy imports `import { api } from "./index"`
export { http as api };           // <— NEW
export default http;              // default export still available

type Json = Record<string, unknown>;

export async function get<T = unknown>(path: string): Promise<T> {
  const { data } = await http.get<T>(path);
  return data;
}
export async function post<T = unknown>(path: string, body?: Json): Promise<T> {
  const { data } = await http.post<T>(path, body ?? {});
  return data;
}
export async function del<T = unknown>(path: string): Promise<T> {
  const { data } = await http.delete<T>(path);
  return data;
}

export const VehiclesAPI = {
  list: () => get("/api/vehicles"),
  create: (payload: Json) => post("/api/vehicles", payload),
  remove: (id: string) => del(`/api/vehicles/${id}`),
};

export const DocumentsAPI = {
  list: () => get("/api/documents"),
  create: (payload: Json) => post("/api/documents", payload),
  remove: (id: string) => del(`/api/documents/${id}`),
};

export const ExpensesAPI = {
  list: () => get("/api/expenses"),
  create: (payload: Json) => post("/api/expenses", payload),
  remove: (id: string) => del(`/api/expenses/${id}`),
};

export const AuthAPI = {
  me: () => get("/api/auth/me"),
  logout: () => post("/api/auth/logout"),
  startGoogle: (redirect = "/dashboard") =>
    `${BASE || ""}/api/auth/google/start?redirect=${encodeURIComponent(redirect)}`,
};