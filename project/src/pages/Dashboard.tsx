import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from '../api';

export default function Dashboard() {
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();
  
    useEffect(() => {
      api.get('/api/auth/me')
        .then(() => setLoading(false))    // ok, render dashboard
        .catch(() => navigate('/signin')); // not logged in
    }, [navigate]);
  
    if (loading) return <div>Loading…</div>;
  
    return (
      <div>{/* your dashboard UI */}</div>
    );
  }