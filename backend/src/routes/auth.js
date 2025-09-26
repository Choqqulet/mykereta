import { Router } from 'express';
import { startGoogle, googleCallback, me, signout } from '../controllers/authController.js';

const router = Router();

router.get('/google/start', startGoogle);
router.get('/google/callback', googleCallback);
router.get('/me', me);
router.post('/signout', signout);

export default router;