import ProtectedRoute from "../components/ProtectedRoute";
import DashboardPage from "./DashboardPage";

export default function Dashboard() {
  // NOTE: wrap DashboardPage here so no need to wrap again in App.tsx
  return (
    <ProtectedRoute>
      <DashboardPage />
    </ProtectedRoute>
  );
}