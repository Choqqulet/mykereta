import jwt from "jsonwebtoken";

const {
  GOOGLE_CLIENT_ID,
  GOOGLE_CLIENT_SECRET,
  GOOGLE_CALLBACK_URL,
  FRONTEND_URL,
  JWT_SECRET = "dev-secret",
} = process.env;

function assertEnv() {
  const miss = [];
  if (!GOOGLE_CLIENT_ID) miss.push("GOOGLE_CLIENT_ID");
  if (!GOOGLE_CLIENT_SECRET) miss.push("GOOGLE_CLIENT_SECRET");
  if (!GOOGLE_CALLBACK_URL) miss.push("GOOGLE_CALLBACK_URL");
  if (!FRONTEND_URL) miss.push("FRONTEND_URL");
  if (miss.length) {
    console.error("[OAuth] Missing env:", miss.join(", "));
    throw new Error("Server misconfigured");
  }
}
assertEnv();

export async function startGoogle(req, res) {
  try {
    const params = new URLSearchParams({
      client_id: GOOGLE_CLIENT_ID,
      redirect_uri: GOOGLE_CALLBACK_URL,
      response_type: "code",
      scope: "profile email",
      prompt: "select_account",
      access_type: "offline",
    });
    // optional redirect back path (e.g. /dashboard)
    const { redirect } = req.query;
    if (redirect) params.set("state", encodeURIComponent(redirect));

    const url = `https://accounts.google.com/o/oauth2/v2/auth?${params.toString()}`;
    return res.redirect(302, url);
  } catch (err) {
    console.error("[OAuth/start] error:", err);
    return res.status(500).json({ error: "OAuth start failed" });
  }
}

export async function googleCallback(req, res) {
  const t0 = Date.now();
  try {
    const { code, state } = req.query;
    if (!code) return res.status(400).json({ error: "Missing code" });

    // 1) Exchange code for tokens
    const body = new URLSearchParams({
      code,
      client_id: GOOGLE_CLIENT_ID,
      client_secret: GOOGLE_CLIENT_SECRET,
      redirect_uri: GOOGLE_CALLBACK_URL,
      grant_type: "authorization_code",
    });

    const tokenResp = await fetch("https://oauth2.googleapis.com/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
    });

    if (!tokenResp.ok) {
      const text = await tokenResp.text();
      console.error("[OAuth/callback] token exchange failed:", tokenResp.status, text);
      return res.status(400).json({ error: "OAuth token exchange failed" });
    }

    const tokens = await tokenResp.json();
    // 2) Fetch user info
    const userResp = await fetch("https://www.googleapis.com/oauth2/v2/userinfo", {
      headers: { Authorization: `Bearer ${tokens.access_token}` },
    });
    if (!userResp.ok) {
      const text = await userResp.text();
      console.error("[OAuth/callback] userinfo failed:", userResp.status, text);
      return res.status(400).json({ error: "Failed to fetch Google profile" });
    }
    const profile = await userResp.json();

    // 3) Upsert user in DB (replace with your prisma/sequelize call)
    // const user = await upsertUserFromGoogle(profile);

    // 4) Set session (JWT cookie)
    const jwtPayload = {
      sub: profile.id,
      email: profile.email,
      name: profile.name,
      picture: profile.picture,
    };
    const token = jwt.sign(jwtPayload, JWT_SECRET, { expiresIn: "7d" });

    // Secure cross-site cookie (Vercel <-> Heroku are different domains)
    res.cookie("session", token, {
      httpOnly: true,
      secure: true,
      sameSite: "none",
      path: "/",
      maxAge: 7 * 24 * 60 * 60 * 1000,
    });

    const redirectPath = state ? decodeURIComponent(state) : "/dashboard";
    console.log("[OAuth/callback] success in", Date.now() - t0, "ms ->", redirectPath);
    return res.redirect(302, `${FRONTEND_URL}${redirectPath}`);
  } catch (err) {
    // Log rich details for diagnosis
    const detail =
      err?.response
        ? { status: err.response.status, data: await err.response.text().catch(() => "") }
        : { message: err?.message };
    console.error("[OAuth/callback] unhandled error:", detail, err);
    return res.status(500).json({ error: "OAuth callback failed" });
  }
}