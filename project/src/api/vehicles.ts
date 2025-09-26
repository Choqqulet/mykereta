import api from "./index";
import type { Vehicle } from "../types";

// READ
export async function listVehicles(): Promise<Vehicle[]> {
  const data = await api.get<Vehicle[]>("/api/vehicles");
  return data ?? [];
}

// CREATE
export async function createVehicle(payload: Partial<Vehicle>): Promise<Vehicle> {
  return api.post<Vehicle>("/api/vehicles", payload);
}

// UPDATE
export async function updateVehicle(id: string, patch: Partial<Vehicle>): Promise<Vehicle> {
  // If your backend expects PUT/PATCH, add a `put` helper in api/index.ts, otherwise POST to /update
  return api.post<Vehicle>(`/api/vehicles/${id}`, patch);
}

// DELETE
export async function deleteVehicle(id: string): Promise<void> {
  await api.del(`/api/vehicles/${id}`);
}