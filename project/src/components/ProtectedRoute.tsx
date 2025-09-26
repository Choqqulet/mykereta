import { ReactNode, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

type Props = {
  children: ReactNode;
  redirectTo?: string;
};

export default function ProtectedRoute({ children, redirectTo = "/signin" }: Props) {
  const { user, loading } = useAuth();
  const nav = useNavigate();
  const loc = useLocation();

  useEffect(() => {
    if (!loading && !user) {
      nav(redirectTo, {
        replace: true,
        state: { from: loc.pathname + loc.search },
      });
    }
  }, [loading, user, nav, redirectTo, loc]);

  if (loading) {
    return (
      <div className="min-h-[60vh] grid place-items-center text-slate-300">
        Checking session…
      </div>
    );
  }

  if (!user) return null; // we'll navigate away

  return <>{children}</>;
}