import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import api from "../api";

export default function SignInPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const target = (location.state as any)?.from ?? "/dashboard";

  const [checking, setChecking] = useState(true);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        await api.me();
        if (!alive) return;
        navigate(target, { replace: true });
      } catch {
        // not signed in → show button
      } finally {
        if (alive) setChecking(false);
      }
    })();
    return () => { alive = false; };
  }, [navigate, target]);

  return (
    <div className="min-h-screen grid place-items-center bg-slate-950 text-white p-6">
      <div className="w-full max-w-xl rounded-2xl bg-slate-900/70 border border-white/10 p-8">
        <h1 className="text-4xl font-bold">
          Welcome to <span className="text-indigo-300">MyKereta</span>
        </h1>
        <p className="text-slate-300 mt-2">Sign in to continue to your dashboard.</p>

        <div className="mt-8">
          {checking ? (
            <div className="text-slate-400">Checking session…</div>
          ) : (
            <button
              onClick={() => (window.location.href = api.startGoogle(target))}
              className="w-full h-12 rounded-xl bg-white text-slate-900 font-semibold hover:bg-slate-100 transition"
            >
              Continue with Google
            </button>
          )}
        </div>

        <p className="text-xs text-slate-400 mt-6">
          By continuing you consent to cookies for sign-in.
        </p>
      </div>
    </div>
  );
}