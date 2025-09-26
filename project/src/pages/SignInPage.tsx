import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

type MeResponse =
  | { user: { sub: string; email?: string; name?: string; picture?: string } }
  | { user: null; error?: string };

const BACKEND =
  (import.meta.env.VITE_BACKEND_URL as string | undefined)?.replace(/\/+$/, "") ||
  "http://127.0.0.1:3000";

export default function SignInPage() {
  const nav = useNavigate();
  const [checking, setChecking] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // If already authenticated (cookie present), go straight to dashboard
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${BACKEND}/api/auth/me`, {
          method: "GET",
          credentials: "include",
        });
        if (!res.ok) throw new Error(`me ${res.status}`);
        const data = (await res.json()) as MeResponse;
        if (!cancelled) {
          if ("user" in data && data.user) {
            nav("/dashboard", { replace: true });
          } else {
            setChecking(false);
          }
        }
      } catch (e: any) {
        if (!cancelled) {
          setChecking(false);
          setErr("Unable to check session. Please try again.");
          // still allow sign-in UI to render
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [nav]);

  const startGoogle = () => {
    // Let Google redirect back to: FRONTEND_URL + /dashboard
    window.location.href = `${BACKEND}/api/auth/google/start?redirect=/dashboard`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-900 via-neutral-950 to-black flex items-center justify-center p-6">
      <div
        className="w-full max-w-md rounded-2xl backdrop-blur-xl"
        style={{
          background:
            "linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06))",
          border: "1px solid rgba(255,255,255,0.18)",
          boxShadow:
            "0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06)",
        }}
      >
        <div className="px-6 pt-6 text-center">
          <h1 className="text-2xl font-semibold text-white">Welcome to MyKereta</h1>
          <p className="text-sm text-neutral-300 mt-1">
            Sign in to continue to your dashboard
          </p>
        </div>

        <div className="p-6">
          {checking ? (
            <div className="flex items-center justify-center gap-2 text-neutral-200">
              <svg
                className="animate-spin h-5 w-5"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-90"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                />
              </svg>
              Checking session…
            </div>
          ) : (
            <>
              {err && (
                <div className="mb-3 text-sm text-red-300 bg-red-900/30 border border-red-500/30 rounded-lg px-3 py-2">
                  {err}
                </div>
              )}

              <button
                onClick={startGoogle}
                className="w-full h-11 mt-1 rounded-xl font-medium text-black bg-white hover:bg-neutral-100 active:bg-neutral-200 transition-colors"
              >
                Use Google
              </button>

              {/* If keeping email/password later, slot here */}
            </>
          )}
        </div>

        <div className="px-6 pb-6 text-center">
          <p className="text-xs text-neutral-400">
            By continuing you consent to cookies for sign-in.
          </p>
        </div>
      </div>
    </div>
  );
}