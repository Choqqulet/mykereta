import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

/** Build API URLs safely for both dev and prod */
function apiUrl(path: string) {
  const base =
    import.meta.env.PROD
      ? "" // Vercel rewrite -> relative /api/*
      : (import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:3000");
  return `${base}${path}`;
}

export default function SignInPage() {
  const navigate = useNavigate();
  const [checking, setChecking] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 1) Check session; if already signed-in, jump to /dashboard
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(apiUrl("/api/auth/me"), {
          credentials: "include",
        });
        if (res.ok) {
          navigate("/dashboard", { replace: true });
          return;
        }
      } catch (e) {
        setError("Unable to check session. Please try again.");
      } finally {
        setChecking(false);
      }
    })();
  }, [navigate]);

  // 2) Start Google OAuth (backend sets cookie + redirects back)
  const startGoogle = () => {
    const start = apiUrl("/api/auth/google/start");
    // Send the desired post-login URL (your callback already reads & uses ?state=)
    window.location.href = `${start}?redirect=/dashboard`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 text-slate-100 flex items-center justify-center p-6">
      <div className="w-full max-w-md">
        <div className="rounded-2xl bg-slate-900/60 border border-white/10 shadow-2xl shadow-black/40 backdrop-blur p-8">
          <h1 className="text-3xl font-semibold tracking-tight">Welcome to <span className="text-indigo-300">MyKereta</span></h1>
          <p className="mt-2 text-slate-300">Sign in to continue to your dashboard.</p>

          <div className="mt-8">
            {checking ? (
              <div className="flex items-center gap-3 text-slate-300">
                <span className="inline-block h-5 w-5 rounded-full border-2 border-slate-300 border-t-transparent animate-spin" />
                <span>Checking session…</span>
              </div>
            ) : (
              <button
                onClick={startGoogle}
                className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-white text-slate-900 font-medium py-3 px-4 hover:bg-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 transition"
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" className="h-5 w-5">
                  <path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12 s5.373-12,12-12c3.059,0,5.842,1.156,7.961,3.039l5.657-5.657C33.64,6.053,29.084,4,24,4C12.955,4,4,12.955,4,24 s8.955,20,20,20s20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z"/>
                  <path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,16.246,18.961,13,24,13c3.059,0,5.842,1.156,7.961,3.039l5.657-5.657 C33.64,6.053,29.084,4,24,4C16.318,4,9.689,8.337,6.306,14.691z"/>
                  <path fill="#4CAF50" d="M24,44c5.137,0,9.728-1.957,13.177-5.146l-6.083-4.999C29.042,35.51,26.651,36,24,36 c-5.198,0-9.594-3.317-11.229-7.946l-6.536,5.036C9.568,39.556,16.222,44,24,44z"/>
                  <path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-0.793,2.24-2.31,4.141-4.206,5.522c0.001-0.001,0,0,0,0l6.083,4.999 C35.01,39.654,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z"/>
                </svg>
                Continue with Google
              </button>
            )}

            {error && <p className="mt-3 text-sm text-rose-300">{error}</p>}

            <p className="mt-6 text-xs text-slate-400">
              By continuing you consent to cookies for sign-in.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}