import { useState } from "react";

export default function PingML() {
  const [out, setOut] = useState("…");

  async function ping() {
    try {
      const base = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:3000";
      const res = await fetch(`${base}/api/ml/health`, { credentials: "include" });
      setOut(await res.text());
    } catch (e: any) {
      setOut(e?.message ?? String(e));
    }
  }

  return (
    <div style={{ padding: 16 }}>
      <button onClick={ping}>Ping ML</button>
      <pre>{out}</pre>
    </div>
  );
}