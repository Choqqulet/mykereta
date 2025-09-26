import { useEffect, useState } from "react";
import { VehiclesAPI, AuthAPI } from "../api";
import { useNavigate } from "react-router-dom";

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [vehicles, setVehicles] = useState<any[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    let alive = true;
    (async () => {
      // Require auth; if missing, go to /signin and remember where we came from
      try {
        await AuthAPI.me();
      } catch {
        navigate("/signin", {
          replace: true,
          state: { from: { pathname: "/dashboard" } },
        });
        return;
      }

      try {
        const list = (await VehiclesAPI.list()) as any[];
        if (alive) setVehicles(list ?? []);
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [navigate]);

  if (loading) {
    return (
      <div className="center">
        <div className="card">Loading…</div>
      </div>
    );
  }

  return (
    <div className="page">
      <h1>Dashboard</h1>
      <pre>{JSON.stringify(vehicles, null, 2)}</pre>
    </div>
  );
}