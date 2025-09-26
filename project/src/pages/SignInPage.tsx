import React, { useEffect, useState } from "react";

type MeResponse =
  | { ok: true; user: { id: string; email?: string; name?: string } }
  | { ok: false; error?: string };

const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL?.replace(/\/+$/, "") ||
  window.location.origin; // fallback for local dev

const REDIRECT_AFTER_LOGIN = "/dashboard"; // change if the route differs

export default function SignInPage() {
  const [checking, setChecking] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // If already signed in (session cookie present), go straight to dashboard
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/auth/me`, {
          method: "GET",
          credentials: "include",
        });
        const data = (await res.json()) as MeResponse;
        if (!cancelled) {
          if (res.ok && "ok" in data && data.ok && data.user?.id) {
            window.location.assign(REDIRECT_AFTER_LOGIN);
            return;
          }
        }
      } catch (e) {
        // swallow; we’ll just show the sign-in UI
        if (!cancelled) setErr(null);
      } finally {
        if (!cancelled) setChecking(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleGoogle = async () => {
    // Backend will handle Google OAuth and redirect back using its configured FRONTEND_URL
    window.location.href = `${BACKEND_URL}/api/auth/google/start`;
  };

  if (checking) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-950 to-black text-white">
        <div className="rounded-2xl border border-white/15 bg-white/10 backdrop-blur-md px-6 py-4">
          Checking session…
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-950 to-black flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="rounded-3xl border border-white/15 bg-white/10 backdrop-blur-xl shadow-2xl p-8 text-white">
          <div className="mb-6">
            <h1 className="text-2xl font-semibold tracking-tight">Welcome back</h1>
            <p className="text-sm text-white/70 mt-1">
              Sign in to continue.
            </p>
          </div>

          {err && (
            <div className="mb-4 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm">
              {err}
            </div>
          )}

          <button
            onClick={handleGoogle}
            className="w-full rounded-xl px-4 py-3 font-medium
                       bg-white/15 hover:bg-white/25 active:bg-white/30
                       border border-white/20 transition
                       backdrop-blur-md"
          >
            Use Google
          </button>

          <div className="mt-6 text-xs text-white/60">
            By continuing, you agree to our Terms &amp; Privacy Policy.
          </div>
        </div>

        {/* Optional footer note */}
        <div className="mt-4 text-center text-xs text-white/50">
          Having trouble? Ensure your pop-up/redirects aren’t blocked.
        </div>
      </div>
    </div>
  );
}