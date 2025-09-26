import { useEffect, useState } from "react";
import { AuthAPI } from "../api";
import { useNavigate, useLocation } from "react-router-dom";

export default function SignInPage() {
  const [checking, setChecking] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        await AuthAPI.me(); // if cookie is valid, go straight in
        if (!alive) return;
        const dest =
          (location.state as any)?.from?.pathname ||
          new URLSearchParams(window.location.search).get("redirect") ||
          "/dashboard";
        navigate(dest, { replace: true });
      } catch {
        // not logged in → show the button
      } finally {
        if (alive) setChecking(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [navigate, location.state]);

  if (checking) {
    return (
      <div className="center">
        <div className="card">Checking session…</div>
      </div>
    );
  }

  const href = AuthAPI.startGoogle("/dashboard");

  return (
    <div className="center">
      <div className="card">
        <h1>Welcome to MyKereta</h1>
        <p>Sign in to continue to your dashboard</p>
        <a className="btn" href={href}>Continue with Google</a>
        <p className="hint">By continuing you consent to cookies for sign-in.</p>
      </div>
    </div>
  );
}