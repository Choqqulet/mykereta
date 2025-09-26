import jwt from 'jsonwebtoken';

const FRONTEND_URL = process.env.FRONTEND_URL || 'http://localhost:5173';
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
const GOOGLE_CALLBACK_URL = process.env.GOOGLE_CALLBACK_URL;
const JWT_SECRET = process.env.JWT_SECRET || 'dev-secret';

const isProd = process.env.NODE_ENV === 'production';
const cookieDomain = (() => {
  try { return new URL(FRONTEND_URL).hostname; } catch { return undefined; }
})();
const cookieOpts = {
  httpOnly: true,
  secure: isProd,          // required by Vercel + SameSite=None
  sameSite: 'none',
  domain: cookieDomain,    // e.g. "mykereta.vercel.app"
  path: '/',
  maxAge: 30 * 24 * 60 * 60 * 1000,
};

export function startGoogle(req, res) {
  const redirect = req.query.redirect || '/dashboard';
  const url = new URL('https://accounts.google.com/o/oauth2/v2/auth');
  url.searchParams.set('prompt', 'select_account');
  url.searchParams.set('response_type', 'code');
  url.searchParams.set('redirect_uri', GOOGLE_CALLBACK_URL);
  url.searchParams.set('scope', 'profile email');
  url.searchParams.set('client_id', GOOGLE_CLIENT_ID);
  // keep where to go after callback
  url.searchParams.set('state', encodeURIComponent(redirect));
  res.redirect(302, url.toString());
}

export async function googleCallback(req, res) {
  try {
    const { code, state } = req.query;
    if (!code) return res.redirect(`${FRONTEND_URL}/signin?error=missing_code`);

    // TODO: exchange "code" for tokens with Google.
    // For now, create a signed session so the UI can proceed.
    const payload = { sub: 'demo-user', email: 'demo@example.com' };
    const token = jwt.sign(payload, JWT_SECRET, { expiresIn: '30d' });

    res.cookie('session', token, cookieOpts);

    const nextUrl = `${FRONTEND_URL}${decodeURIComponent(state || '/dashboard')}`;
    return res.redirect(302, nextUrl);
  } catch (err) {
    console.error('callback error', err);
    return res.redirect(`${FRONTEND_URL}/signin?error=oauth_failed`);
  }
}

export function me(req, res) {
  const token = req.cookies?.session;
  if (!token) return res.status(401).json({ authenticated: false });

  try {
    const data = jwt.verify(token, JWT_SECRET);
    res.json({ authenticated: true, user: data });
  } catch {
    res.status(401).json({ authenticated: false });
  }
}

export function signout(_req, res) {
  res.clearCookie('session', { ...cookieOpts, maxAge: 0 });
  res.json({ ok: true });
}