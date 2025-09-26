import { Router } from 'express';
import passport from 'passport';

const router = Router();

// optional: keep classic /google
router.get('/google',
  (req, res, next) => {
    // allow ?redirect=https://… to decide where to go after login
    if (req.query.redirect) req.session.returnTo = req.query.redirect;
    next();
  },
  passport.authenticate('google', {
    scope: ['profile', 'email'],
    prompt: 'select_account',
    session: true,
  })
);

// alias used by your frontend: /google/start
router.get('/google/start',
  (req, res, next) => {
    if (req.query.redirect) req.session.returnTo = req.query.redirect;
    next();
  },
  passport.authenticate('google', {
    scope: ['profile', 'email'],
    prompt: 'select_account',
    session: true,
  })
);

// Google sends user back here
router.get('/google/callback',
  passport.authenticate('google', {
    failureRedirect: `${process.env.FRONTEND_URL}/signin`,
    session: true,
  }),
  (req, res) => {
    const to = req.session.returnTo || `${process.env.FRONTEND_URL}/dashboard`;
    delete req.session.returnTo;
    res.redirect(to);
  }
);

export default router;